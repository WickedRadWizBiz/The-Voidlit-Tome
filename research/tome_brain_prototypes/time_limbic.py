import time, warnings, sys
warnings.filterwarnings("ignore")
import torch
torch.set_num_threads(1)
import limbic_bn as LB
from spikingjelly.activation_based import neuron, layer, functional, surrogate
net = LB.build()
def run_epoch():
    net.run(inputs={"ctx": (torch.rand(2, 1, 256) < 0.03).float(), "intero": (torch.rand(2, 1, 64) < 0.05).float()},
            time=2, reward=0.0)
run_epoch()
t0 = time.perf_counter()
for _ in range(100): run_epoch()
print(f"BindsNET limbic core (8 layers, 13 conns, {sum(LB.SIZES.values())} units), 100 epochs x 2 ms: "
      f"{(time.perf_counter()-t0)*1e3:.1f} ms/turn")
# SpikingJelly cerebellum, multi-step mode per epoch
class Cerebellum(torch.nn.Module):
    def __init__(self, n_mf=96, n_gc=4096, n_out=15, per=8):
        super().__init__()
        g = torch.Generator().manual_seed(1)
        conn = torch.zeros(n_mf, n_gc)
        for j in range(n_gc):
            conn[torch.randperm(n_mf, generator=g)[:4], j] = 1.0  # each granule cell samples 4 mossy fibres
        self.register_buffer("mf_gc", conn * 0.6)
        self.gc = neuron.LIFNode(tau=5.0, v_threshold=1.0, step_mode="m", detach_reset=True)
        self.golgi = neuron.LIFNode(tau=10.0, step_mode="m")
        self.register_buffer("pf_pc", torch.rand(n_gc, n_out * per, generator=g) * 0.01)
        self.pc = neuron.LIFNode(tau=10.0, step_mode="m")
        self.register_buffer("golgi_w", torch.ones(n_gc, 64) / n_gc * 20)
        self.register_buffer("golgi_back", torch.ones(64, n_gc) * 0.05)
        self.register_buffer("g_prev", torch.zeros(1, 64))
    def forward(self, mf):  # mf [T, 1, n_mf]
        drive = mf @ self.mf_gc - self.g_prev @ self.golgi_back
        s_gc = self.gc(drive)
        s_go = self.golgi(s_gc @ self.golgi_w)
        self.g_prev = s_go[-1]
        return self.pc(s_gc @ self.pf_pc), s_gc
cb = Cerebellum()
with torch.no_grad():
    cb((torch.rand(2, 1, 96) < 0.05).float())
    t0 = time.perf_counter()
    for _ in range(100):
        out, sgc = cb((torch.rand(2, 1, 96) < 0.05).float())
print(f"SpikingJelly cerebellum (96 MF, 4096 GC, 120 PC), 100 epochs x 2 steps (multi-step): {(time.perf_counter()-t0)*1e3:.1f} ms/turn")
sys.path.insert(0, "/home/user/The-Voidlit-Tome/brain")
from voidlit.cortex.director import load_director
m = load_director()
functional.set_step_mode(m.net, "m"); functional.reset_net(m.net)
x = torch.rand(1, 18)
with torch.no_grad():
    t0 = time.perf_counter()
    for _ in range(10):
        functional.reset_net(m.net)
        out = m.net((torch.rand(200, 1, 18) < x).float()).mean(0)
print(f"SpikingJelly Director prior, one multi-step call of T=200 (runs ahead: no bus inputs): {(time.perf_counter()-t0)/10*1e3:.2f} ms/turn")
