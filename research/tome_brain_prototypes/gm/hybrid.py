"""One-clock hybrid prototype of the Tome's single network (timing, determinism, save/load).

Three partitions of ONE connectome on ONE 1-ms master clock, exchanging spikes through a ring-buffer bus:
  JAX   : thalamus (core/matrix/TRN), 8 cortical columns (L1 LAMP5, L2/3 4-compartment pyramids, PV/SST/VIP, L4, L5 ET/IT,
          L6 CT), hippocampus CA3/CA1, 8 astrocyte domains, energy budget.  L2/3 at dt=0.1 ms (Hines + Rush-Larsen).
  SJ    : striatum (D1/D2 MSNs per action channel, FSI, GPi) and cerebellum (mossy -> 2048 granule -> Purkinje -> DCN, IO).
  BN    : limbic core (hypothalamic need nuclei, BLA/CeA, VTA, LC, DRN), MSTDPET on the cortex->BLA projection.
Every cross-partition projection has delay >= D_MIN, so each partition can integrate D_MIN ticks on its own between
exchanges (conservative synchronisation: nothing it reads in the window can be produced inside the window).
"""
from __future__ import annotations

import io, sys, time, warnings
warnings.filterwarnings("ignore")
import jaxpath  # noqa: F401
import jax, jax.numpy as jnp
import numpy as np
import torch
from spikingjelly.activation_based import neuron, functional
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.learning import MSTDPET
import l23gm as L

torch.set_num_threads(1)
D_MIN = 4          # ms: minimum cross-partition delay = exchange epoch
BN_DT = 2          # limbic partition integrates every 2nd master tick (slow nuclei)
TURN_MS = 252      # 63 epochs
H = 32             # bus horizon (ticks)

# ------------------------------------------------------------------------------------------------ populations
J_POINT = dict(core=64, matrix=32, trn=32, l4=128, l5et=64, l5it=64, l6ct=64, pv=48, sst=32, vip=32, lamp5=24,
               ca3=256, ca1=128)
N_L23 = 192
N_ACT = 14         # 8 beats + disclose, withhold, suggest, corrupt_npc, bend_player, address
SJ_POP = dict(d1=N_ACT * 12, d2=N_ACT * 12, fsi=16, gpi=N_ACT * 4, gc=2048, pc=64, dcn=16, io=16)
BN_POP = dict(need=64, bla=64, cea=16, vta=32, lc=16, drn=16)

# cross-partition projections: (src_part, src_pop, dst_part, dst_pop, delay_ms)
PROJ = [("J", "l5it", "S", "d1", 4), ("J", "l5it", "S", "d2", 4), ("J", "l5et", "S", "mossy", 6),
        ("J", "core", "S", "mossy", 6), ("S", "gpi", "J", "matrix", 4), ("S", "dcn", "J", "core", 4),
        ("J", "l5et", "B", "bla", 6), ("J", "ca1", "B", "bla", 8), ("B", "bla", "J", "l5it", 6),
        ("B", "need", "J", "matrix", 8), ("B", "vta", "S", "d1", 4), ("B", "lc", "J", "pv", 6),
        ("B", "cea", "S", "io", 6), ("S", "pc", "B", "lc", 6)]
assert all(p[4] >= D_MIN for p in PROJ)


class Bus:
    """Ring buffer per source population, indexed by master tick. One payload type: float spikes."""

    def __init__(self, sizes):
        self.buf = {k: torch.zeros(H, n) for k, n in sizes.items()}

    def write(self, src, t0, x):                       # x [D_MIN, n]
        idx = [(t0 + i) % H for i in range(x.shape[0])]
        self.buf[src][idx] = torch.as_tensor(np.asarray(x), dtype=torch.float32)

    def read(self, src, t0, delay):                    # spikes emitted at t-delay for t in [t0, t0+D_MIN)
        idx = [(t0 + i - delay) % H for i in range(D_MIN)]
        return self.buf[src][idx]


# ------------------------------------------------------------------------------------------------ JAX partition
def j_build(key):
    names = list(J_POINT)
    off, o = {}, 0
    for k in names:
        off[k] = (o, o + J_POINT[k]); o += J_POINT[k]
    n = o
    k1, k2, k3 = jax.random.split(key, 3)
    inhib = jnp.zeros(n)
    for k in ("trn", "pv", "sst", "vip", "lamp5"):
        a, b = off[k]; inhib = inhib.at[a:b].set(1.0)
    W = (jax.random.uniform(k1, (n, n)) < 0.06) * jax.random.uniform(k2, (n, n), minval=0.2, maxval=0.8)
    W = W * jnp.where(inhib[:, None] > 0, -4.0, 0.15)  # rows = presynaptic; E weak, I strong (balanced)
    W_l23_in = (jax.random.uniform(k3, (n, N_L23, 2)) < 0.05) * 3.0 * (1 - inhib)[:, None, None]  # E -> basal/tuft (nS)
    W_l23_inh = (jax.random.uniform(k2, (n, N_L23, 2)) < 0.25) * 4.0 * inhib[:, None, None]       # PV->soma, SST->tuft
    W_l23_out = (jax.random.uniform(k1, (N_L23, n)) < 0.06) * 0.25
    ext_in = {"matrix": 0, "core": 0, "l5it": 0, "pv": 0}
    return dict(n=n, off=off, W=W, W_in=W_l23_in, W_inh=W_l23_inh, W_out=W_l23_out, inhib=inhib)


def j_init(G):
    return dict(v=jnp.zeros(G["n"]), ref=jnp.zeros(G["n"]), cell=L.init(N_L23),
                astro_ip3=jnp.zeros(8), astro_ca=jnp.full(8, 0.1), astro_g=jnp.zeros(8), energy=jnp.array(1.0),
                last=jnp.zeros(G["n"]), last23=jnp.zeros(N_L23))


def make_j_epoch(G):
    n, W, W_in, W_inh, W_out = G["n"], G["W"], G["W_in"], G["W_inh"], G["W_out"]
    tau, dt_sub, nsub = 10.0, 0.1, 10
    decay = np.exp(-1.0 / tau)

    def tick(c, inp):
        st, key = c
        ext, = inp                                       # [n] external drive this ms (bus + game input)
        key, sub = jax.random.split(key)
        rec = st["last"] @ W + st["last23"] @ W_out
        noise = 0.10 * jax.random.normal(sub, (n,))
        gain = 1.0 + 0.25 * st["astro_g"].mean()         # astrocytic precision on cortical gain
        v = st["v"] * decay + (rec + ext) * gain + noise
        v = jnp.where(st["ref"] > 0, 0.0, v)
        s = (v > 1.0).astype(jnp.float32)
        v = jnp.where(s > 0, 0.0, v)
        ref = jnp.where(s > 0, 2.0, jnp.maximum(st["ref"] - 1.0, 0.0))
        # L2/3 4-compartment cells, 10 substeps of 0.1 ms; synaptic increments delivered at substep 0
        drive = jnp.einsum("p,pcj->cj", s, W_in)        # [N_L23, 2]
        wa0 = jnp.zeros((N_L23, 4)).at[:, 1].set(drive[:, 0]).at[:, 3].set(drive[:, 1])
        inh = jnp.einsum("p,pcj->cj", s, W_inh)
        wi0 = jnp.zeros((N_L23, 4)).at[:, 0].set(inh[:, 0]).at[:, 3].set(inh[:, 1])
        def sub_step(cs, k):
            wa = jnp.where(k == 0, wa0, 0.0)
            wi = jnp.where(k == 0, wi0, 0.0)
            cs, sp = L.step(cs, wa, 0.5 * wa, wi, jnp.zeros(N_L23), dt_sub)
            return cs, sp
        cell, sp23 = jax.lax.scan(sub_step, st["cell"], jnp.arange(nsub))
        s23 = sp23.any(0).astype(jnp.float32)
        # De Pitta-style astrocyte (one per column) driven by local L2/3 activity, 1 ms step (exponential Euler)
        z = s23.reshape(8, -1).mean(1)
        ip3 = st["astro_ip3"] + (1.0 - jnp.exp(-1.0 / 100.0)) * (jnp.tanh(50 * z) - st["astro_ip3"])
        ca = st["astro_ca"] + (1.0 - jnp.exp(-1.0 / 500.0)) * (0.1 + 1.5 * ip3 - st["astro_ca"])
        g = st["astro_g"] + (1.0 - jnp.exp(-1.0 / 250.0)) * (jax.nn.sigmoid(10 * (ca - 0.4)) - st["astro_g"])
        energy = jnp.clip(st["energy"] - 2e-6 * (s.sum() + 5 * s23.sum()) + 1e-4 * (1 - st["energy"]), 0.0, 1.0)
        st = dict(v=v, ref=ref, cell=cell, astro_ip3=ip3, astro_ca=ca, astro_g=g, energy=energy, last=s, last23=s23)
        return (st, key), (s, s23)

    @jax.jit
    def epoch(st, key, ext):                            # ext [D_MIN, n]
        (st, key), (s, s23) = jax.lax.scan(tick, (st, key), (ext,))
        return st, key, s, s23
    return epoch


# ------------------------------------------------------------------------------------------------ SJ partition
class SJPart(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        P = SJ_POP
        self.n_mossy = J_POINT["l5et"] + J_POINT["core"]
        conn = torch.zeros(self.n_mossy, P["gc"])
        for j in range(P["gc"]):
            conn[torch.randperm(self.n_mossy, generator=g)[:4], j] = 1.0   # 4 mossy inputs per granule cell
        self.register_buffer("mf_gc", conn * 0.35)
        self.register_buffer("gc_pc", torch.rand(P["gc"], P["pc"], generator=g) * 0.01)   # PF-PC (plastic)
        self.register_buffer("ctx_d1", (torch.rand(J_POINT["l5it"], P["d1"], generator=g) < 0.1) * 0.4)
        self.register_buffer("ctx_d2", (torch.rand(J_POINT["l5it"], P["d2"], generator=g) < 0.1) * 0.4)
        self.register_buffer("d1_gpi", -torch.kron(torch.eye(N_ACT), torch.ones(12, 4)) * 0.3)
        self.register_buffer("d2_gpi", torch.kron(torch.eye(N_ACT), torch.ones(12, 4)) * 0.15)
        self.register_buffer("pc_dcn", -torch.kron(torch.eye(16), torch.ones(4, 1)) * 0.5)
        mk = lambda tau: neuron.LIFNode(tau=tau, v_threshold=1.0, step_mode="m", detach_reset=True)
        self.gc, self.pc, self.dcn, self.d1, self.d2, self.gpi = mk(5.0), mk(10.0), mk(10.0), mk(20.0), mk(20.0), mk(5.0)

    @torch.no_grad()
    def epoch(self, l5it, mossy, da, io_drive):
        gc = self.gc(mossy @ self.mf_gc)
        pc = self.pc(gc @ self.gc_pc + 0.3)
        dcn = self.dcn(pc @ self.pc_dcn + 0.4)
        d1 = self.d1(l5it @ self.ctx_d1 * (1 + da.mean(-1, keepdim=True)))
        d2 = self.d2(l5it @ self.ctx_d2 * (1 - 0.5 * da.mean(-1, keepdim=True)))
        gpi = self.gpi(d1 @ self.d1_gpi + d2 @ self.d2_gpi + 0.6)
        # Marr-Albus-Ito: PF->PC LTD on climbing-fibre error, slow LTP otherwise (per epoch, eligibility = gc trace)
        cf = io_drive.mean()
        self.gc_pc.add_(gc.mean(0).unsqueeze(1) * (1e-5 - 1e-3 * cf)).clamp_(0.0, 0.05)
        return dict(gpi=gpi, dcn=dcn, pc=pc)


# ------------------------------------------------------------------------------------------------ BindsNET partition
def bn_build(seed):
    torch.manual_seed(seed)
    net = Network(dt=float(BN_DT), batch_size=1, learning=True)
    ins = dict(in_l5et=J_POINT["l5et"], in_ca1=J_POINT["ca1"], in_pc=SJ_POP["pc"], in_game=32)
    for k, n in ins.items():
        net.add_layer(Input(n=n, traces=True), name=k)
    for k, n in BN_POP.items():
        net.add_layer(LIFNodes(n=n, thresh=-52.0, rest=-65.0, reset=-65.0, refrac=2, tc_decay=20.0, traces=True), name=k)
    def C(a, b, scale, p=0.2, rule=None):
        w = (torch.rand(net.layers[a].n, net.layers[b].n) < p).float() * scale
        kw = dict(update_rule=MSTDPET, nu=0.01, wmin=0.0, wmax=scale * 2) if rule else {}
        c = Connection(net.layers[a], net.layers[b], w=w, **kw)
        net.add_connection(c, a, b)
    C("in_l5et", "bla", 1.5, p=0.1, rule=True)
    C("in_ca1", "bla", 1.5, p=0.05, rule=True)
    C("in_game", "need", 8.0, p=0.3)
    C("need", "vta", 1.5, p=0.1); C("bla", "cea", 1.5, p=0.1); C("bla", "vta", 1.0, p=0.1); C("cea", "lc", 2.0, p=0.2)
    C("in_pc", "lc", 1.0, p=0.1); C("need", "drn", 2.0, p=0.1); C("drn", "bla", -2.0)
    mons = {}
    for k in ("bla", "cea", "vta", "lc", "need", "drn"):
        m = Monitor(net.layers[k], state_vars=("s",), time=D_MIN // BN_DT)
        net.add_monitor(m, k); mons[k] = m
    return net, mons


# ------------------------------------------------------------------------------------------------ the one network
class Tome:
    def __init__(self, seed=0, threads=False):
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)
        self.G = j_build(jax.random.fold_in(self.key, 1))
        self.jst = j_init(self.G)
        self.jepoch = make_j_epoch(self.G)
        self.sj = SJPart(seed + 2)
        self.bn, self.mons = bn_build(seed + 3)
        self.gen = torch.Generator().manual_seed(seed + 4)
        sizes = {**{f"J.{k}": v for k, v in J_POINT.items()}, **{f"S.{k}": v for k, v in SJ_POP.items()},
                 **{f"B.{k}": v for k, v in BN_POP.items()}}
        self.bus = Bus(sizes)
        self.t = 0
        self.threads = threads
        self.timing = {"J": 0.0, "S": 0.0, "B": 0.0, "bus": 0.0}

    def _jslice(self, k):
        return self.G["off"][k]

    def epoch(self, game_rates):
        t0, bus = self.t, self.bus
        tb = time.perf_counter()
        # ---- gather inbound (all emitted >= D_MIN ago)
        ext = torch.zeros(D_MIN, self.G["n"])
        a, b = self._jslice("matrix"); ext[:, a:b] += 0.08 * (bus.read("S.gpi", t0, 4) @ torch.ones(SJ_POP["gpi"], b - a)) * -1
        ext[:, a:b] += 0.05 * bus.read("B.need", t0, 8) @ torch.ones(BN_POP["need"], b - a) / 8
        a, b = self._jslice("core")
        game_sp = (torch.rand(D_MIN, 64, generator=self.gen) < 0.15 * game_rates[None, :64]).float()   # <= 45 Hz Poisson
        ext[:, a:b] += 0.3 * bus.read("S.dcn", t0, 4).repeat(1, 4) + 1.1 * game_sp
        a, b = self._jslice("l5it"); ext[:, a:b] += 0.1 * bus.read("B.bla", t0, 6)
        a, b = self._jslice("pv"); ext[:, a:b] += 0.2 * bus.read("B.lc", t0, 6).repeat(1, 3)
        l5it = bus.read("J.l5it", t0, 4)
        mossy = torch.cat([bus.read("J.l5et", t0, 6), bus.read("J.core", t0, 6)], 1)
        da = bus.read("B.vta", t0, 4)
        io = bus.read("B.cea", t0, 6)
        bn_in = {"in_l5et": bus.read("J.l5et", t0, 6).unsqueeze(1), "in_ca1": bus.read("J.ca1", t0, 8).unsqueeze(1),
                 "in_pc": bus.read("S.pc", t0, 6).unsqueeze(1),
                 "in_game": (torch.rand(D_MIN, 1, 32, generator=self.gen) < 0.08 * game_rates[None, None, 64:96]).float()}
        bn_in = {k: v.reshape(D_MIN // BN_DT, BN_DT, 1, -1).amax(1) for k, v in bn_in.items()}   # 2-ms bins
        self.timing["bus"] += time.perf_counter() - tb

        # ---- integrate the three partitions over [t0, t0 + D_MIN)
        def run_j():
            t = time.perf_counter()
            st, key, s, s23 = self.jepoch(self.jst, self.key, jnp.asarray(ext.numpy()))
            s = np.asarray(s); self._last23 = np.asarray(s23)
            self.timing["J"] += time.perf_counter() - t
            return st, key, s

        def run_s():
            t = time.perf_counter()
            out = self.sj.epoch(l5it, mossy, da, io)
            self.timing["S"] += time.perf_counter() - t
            return out

        def run_b():
            t = time.perf_counter()
            self.bn.run(inputs=bn_in, time=D_MIN, reward=float(da.mean()))
            out = {k: m.get("s")[:, 0].float().repeat_interleave(BN_DT, 0) * 0 for k, m in self.mons.items()}
            for k, m in self.mons.items():               # a spike is published on the first master tick of its bin
                out[k][::BN_DT] = m.get("s")[:, 0].float()
            self.timing["B"] += time.perf_counter() - t
            return out

        if self.threads:
            fj = POOL.submit(run_j); fs = POOL.submit(run_s); fb = POOL.submit(run_b)
            (st, key, s), so, bo = fj.result(), fs.result(), fb.result()
        else:
            st, key, s = run_j(); so = run_s(); bo = run_b()
        self.jst, self.key = st, key
        # ---- publish outbound
        tb = time.perf_counter()
        for k, (a, b) in self.G["off"].items():
            bus.write(f"J.{k}", t0, s[:, a:b])
        for k, v in so.items():
            bus.write(f"S.{k}", t0, v)
        for k, v in bo.items():
            bus.write(f"B.{k}", t0, v)
        self.timing["bus"] += time.perf_counter() - tb
        self.t += D_MIN
        return s, so, bo

    def turn(self, game_rates):
        counts = {}
        for _ in range(TURN_MS // D_MIN):
            s, so, bo = self.epoch(game_rates)
            for k, v in bo.items():
                counts[k] = counts.get(k, 0.0) + float(v.sum())
            counts["gpi"] = counts.get("gpi", 0.0) + float(so["gpi"].sum())
            counts["J"] = counts.get("J", 0.0) + float(s.sum())
            counts["l23"] = counts.get("l23", 0.0) + float(self._last23.sum())
        return counts

    # ---------------------------------------------------------------- whole-brain save/load
    def state_dict(self):
        bn_state = {k: v.clone() for k, v in self.bn.state_dict().items()}
        bn_dyn = {name: {a: getattr(l, a).clone() for a in ("v", "s", "x", "refrac_count") if hasattr(l, a)}
                  for name, l in self.bn.layers.items()}
        bn_elig = {f"{a}->{b}": {k: getattr(c.update_rule, k).clone() for k in ("eligibility", "eligibility_trace", "p_plus", "p_minus") if hasattr(c.update_rule, k)}
                   for (a, b), c in self.bn.connections.items()}
        sj_dyn = {name: m.v.clone() if isinstance(m.v, torch.Tensor) else m.v for name, m in self.sj.named_modules() if isinstance(m, neuron.LIFNode)}
        return dict(t=self.t, key=np.asarray(self.key), jst=jax.tree_util.tree_map(np.asarray, self.jst),
                    sj=self.sj.state_dict(), sj_dyn=sj_dyn, bn=bn_state, bn_dyn=bn_dyn, bn_elig=bn_elig,
                    bus={k: v.clone() for k, v in self.bus.buf.items()}, gen=self.gen.get_state(),
                    mons={k: [r.clone() for r in m.recording["s"]] for k, m in self.mons.items()})

    def load_state_dict(self, d):
        self.t, self.key = d["t"], jnp.asarray(d["key"])
        self.jst = jax.tree_util.tree_map(jnp.asarray, d["jst"])
        self.sj.load_state_dict(d["sj"])
        for name, m in self.sj.named_modules():
            if isinstance(m, neuron.LIFNode):
                m.v = d["sj_dyn"][name].clone() if isinstance(d["sj_dyn"][name], torch.Tensor) else d["sj_dyn"][name]
        self.bn.load_state_dict(d["bn"])
        for name, l in self.bn.layers.items():
            for a, v in d["bn_dyn"][name].items():
                setattr(l, a, v.clone())
        for (a, b), c in self.bn.connections.items():
            for k, v in d["bn_elig"][f"{a}->{b}"].items():
                setattr(c.update_rule, k, v.clone())
        for k, v in d["bus"].items():
            self.bus.buf[k] = v.clone()
        self.gen.set_state(d["gen"])
        for k, m in self.mons.items():
            m.recording["s"] = [r.clone() for r in d["mons"][k]]


from concurrent.futures import ThreadPoolExecutor
POOL = ThreadPoolExecutor(max_workers=3)


def game_rates(seed, turn):
    g = torch.Generator().manual_seed(seed * 1000 + turn)
    return torch.rand(96, generator=g) * 0.3


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "time"
    if mode == "time":
        for threads in (False, True):
            tm = Tome(seed=0, threads=threads)
            t0 = time.perf_counter(); tm.turn(game_rates(0, 0)); first = time.perf_counter() - t0
            tm.timing = {k: 0.0 for k in tm.timing}
            walls = []
            for k in range(1, 9):
                t0 = time.perf_counter(); c = tm.turn(game_rates(0, k)); walls.append(time.perf_counter() - t0)
            n = len(walls)
            print(f"threads={threads}: first turn (incl. JIT) {first:.2f}s; steady turns {np.mean(walls)*1000:.0f} ms "
                  f"mean, {np.max(walls)*1000:.0f} ms max; per-turn breakdown "
                  + ", ".join(f"{k} {v/n*1000:.0f} ms" for k, v in tm.timing.items()), flush=True)
            sizes = {**BN_POP, "gpi": SJ_POP["gpi"], "J": sum(J_POINT.values()), "l23": N_L23}
            print("   mean rates last turn (Hz):", {k: round(v / sizes[k] / (TURN_MS / 1000), 1) for k, v in c.items()})
    elif mode == "det":
        def play(seed, n, save_at=None):
            tm = Tome(seed=seed); out = []
            blob = None
            for k in range(n):
                if save_at is not None and k == save_at:
                    buf = io.BytesIO(); torch.save(tm.state_dict(), buf); blob = buf.getvalue()
                out.append(tm.turn(game_rates(seed, k)))
            return out, blob
        a, blob = play(7, 6, save_at=3)
        b, _ = play(7, 6)
        print("same seed identical over 6 turns:", a == b)
        tm = Tome(seed=7)
        tm.load_state_dict(torch.load(io.BytesIO(blob), weights_only=False))
        resumed = [tm.turn(game_rates(7, k)) for k in range(3, 6)]
        print("save at turn 3 -> load into a fresh Tome -> identical continuation:", resumed == a[3:])
        print("save blob size:", len(blob) // 1024, "KiB")
        c, _ = play(8, 2)
        print("different seed differs:", c != a[:2])
