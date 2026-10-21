"""Integrated timing + exactness prototype of the one-clock hybrid: JAX cortex/thalamus/hippocampus partition,
BindsNET limbic core, SpikingJelly cerebellum (10 ms lookahead) and SpikingJelly Director prior (runs ahead),
exchanging spikes through one ring-buffer bus on a 1 ms master clock. Cross-partition delays >= 2 ms."""
import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/The-Voidlit-Tome/brain")
import torch
from concurrent.futures import ThreadPoolExecutor
torch.set_num_threads(1)
import jaxpath  # noqa
import jax, jax.numpy as jnp, numpy as np
import column2 as COL
from time_limbic2 import build as build_limbic
from spikingjelly.activation_based import neuron, functional
from voidlit.cortex.director import load_director

TURN = 200
D_MIN = 2

class Bus:
    """One ring buffer per source population, indexed by master tick; float payload (spikes or graded current)."""
    def __init__(self, sizes, horizon=64):
        self.h = horizon
        self.buf = {k: torch.zeros(horizon, n) for k, n in sizes.items()}
    def write(self, src, t0, x):
        for i in range(x.shape[0]):
            self.buf[src][(t0 + i) % self.h] = x[i]
    def read(self, src, t0, T, delay):
        idx = [(t0 + i - delay) % self.h for i in range(T)]
        return self.buf[src][idx]
    def state(self):
        return {k: v.clone() for k, v in self.buf.items()}

class Cerebellum(torch.nn.Module):
    def __init__(self, n_mf=96, n_gc=4096, n_out=15, per=8, seed=1):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        conn = torch.zeros(n_mf, n_gc)
        for j in range(n_gc):
            conn[torch.randperm(n_mf, generator=g)[:4], j] = 1.0
        self.register_buffer("mf_gc", conn * 0.6)
        self.gc = neuron.LIFNode(tau=5.0, v_threshold=1.0, step_mode="m", detach_reset=True)
        self.pc = neuron.LIFNode(tau=10.0, step_mode="m")
        self.register_buffer("pf_pc", torch.rand(n_gc, n_out * per, generator=g) * 0.02)
    def forward(self, mf):
        return self.pc(self.gc(mf @ self.mf_gc) @ self.pf_pc)

class Tome:
    def __init__(self, seed=0):
        self.jrun = COL.make(0.1, 10)
        self.jst, self.W, self.idx = COL.init(jax.random.PRNGKey(seed), 0.1)
        self.limbic = build_limbic(640, n_ctx=96)
        from bindsnet.network.monitors import Monitor
        self.lim_mon = Monitor(self.limbic.layers['limbic'], state_vars=('s',), time=None)
        self.limbic.add_monitor(self.lim_mon, 'lim')
        self.cb = Cerebellum()
        self.director = load_director(); functional.set_step_mode(self.director.net, "m")
        self.bus = Bus({"cortex": COL.N, "limbic": 640, "cb": 120, "prior": 8, "thal_in": COL.N})
        self.l5 = torch.arange(160, 224)  # stand-in: column 0 L5 cells -> 64 of the 96 ctx afferents
        g = torch.Generator().manual_seed(seed)
        self.w_bg_cx = (torch.rand(640, COL.N, generator=g) < 0.02).float() * 0.002
        self.t = 0
        self.pool = ThreadPoolExecutor(1)
        self.threaded = True
    def _jax_epoch(self, ext):
        st, sp = self.jrun(self.jst, self.W, self.idx, ext)
        sp.block_until_ready()
        return st, sp
    def turn(self, features, epoch=D_MIN, cb_epoch=10, seed=0):
        g = torch.Generator().manual_seed(seed)
        # external afferents for the whole turn, pre-generated (chunking-independent noise)
        thal = (torch.rand(TURN, COL.N, generator=g) < 0.02).float() * 0.8
        intero = (torch.rand(TURN, 1, 64, generator=g) < 0.2).float()
        mf_ext = (torch.rand(TURN, 1, 96, generator=g) < 0.2).float()
        with torch.no_grad():  # Director prior: no incoming bus links -> runs the whole turn ahead
            functional.reset_net(self.director.net)
            x = torch.as_tensor(features).unsqueeze(0)
            prior = self.director.net((torch.rand(TURN, 1, 18, generator=g) < x).float())[:, 0]
        self.bus.write("prior", self.t, prior)
        self.bus.write("thal_in", self.t, thal)
        t0 = self.t
        for k in range(0, TURN, epoch):
            t = t0 + k
            # JAX inputs: thalamic afferents + basal-ganglia output (delay 3 ms) -> dispatched asynchronously
            ext = self.bus.read("thal_in", t, epoch, 0) + self.bus.read("limbic", t, epoch, 3) @ self.w_bg_cx
            jext = jnp.from_dlpack(ext.contiguous())
            fut = self.pool.submit(self._jax_epoch, jext) if self.threaded else None
            if not self.threaded:
                self.jst, jsp = self._jax_epoch(jext)
            # BindsNET limbic core, same epoch, while XLA runs: cortical L5 (delay 4) + prior (delay 3)
            cx = self.bus.read("cortex", t, epoch, 4)[:, self.l5]
            pr = self.bus.read("prior", t, epoch, 3)
            ctx = torch.cat([cx, pr.repeat(1, 4)], 1).unsqueeze(1)
            self.lim_mon.reset_state_variables()
            self.limbic.run(inputs={"ctx": ctx, "intero": intero[k:k + epoch]}, time=epoch, reward=0.0)
            self.bus.write("limbic", t, self.lim_mon.get("s")[:, 0].float())
            # SpikingJelly cerebellum with 10 ms lookahead (pontine delay 10 ms)
            if (t - t0) % cb_epoch == 0:
                mf = mf_ext[k:k + cb_epoch].clone()
                mf[:, 0, :64] += self.bus.read("cortex", t, cb_epoch, 10)[:, self.l5]
                with torch.no_grad():
                    self.bus.write("cb", t, self.cb(mf)[:, 0])
            if self.threaded:
                self.jst, jsp = fut.result()
            self.bus.write("cortex", t, torch.from_dlpack(jsp))   # sync point with XLA
        self.t += TURN
        return self.bus.state()

if __name__ == "__main__":
    feats = torch.rand(18)
    tome = Tome()
    tome.turn(feats, seed=0)  # warm-up / compile
    ts = []
    for s in range(1, 6):
        t0 = time.perf_counter(); tome.turn(feats, seed=s); ts.append(time.perf_counter() - t0)
    print("integrated turn wall ms (JAX in worker thread):", [round(x * 1e3, 1) for x in ts])
    # exactness: same seed, chunked into 2-ms epochs vs lock-step 1-ms epochs -> identical bus contents
    a, b = Tome(seed=3), Tome(seed=3)
    for s in range(2):
        sa = a.turn(feats, epoch=2, cb_epoch=10, seed=100 + s)
        sb = b.turn(feats, epoch=1, cb_epoch=10, seed=100 + s)
    same = all(torch.equal(sa[k], sb[k]) for k in sa)
    print("epoch invariance (2 ms vs 1 ms epochs, 2 turns): bus identical =", same,
          "| cortex spikes/turn:", int(sa["cortex"].sum()), "| limbic spikes:", int(sa["limbic"].sum()),
          "| cerebellum spikes:", int((sa["cb"] > 0).sum()))
