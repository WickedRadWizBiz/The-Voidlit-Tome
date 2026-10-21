import time, warnings
warnings.filterwarnings("ignore")
import torch
torch.set_num_threads(1)
from bindsnet.learning import MSTDPET
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

def build(n_lim, n_ctx=256, n_int=64, learn_cols=None):
    g = torch.Generator().manual_seed(0)
    net = Network(dt=1.0, batch_size=1, learning=True)
    ctx, intero = Input(n=n_ctx), Input(n=n_int)
    lim = LIFNodes(n=n_lim, thresh=-52.0, rest=-65.0, reset=-65.0, tc_decay=20.0, refrac=2)
    for k, v in (("ctx", ctx), ("intero", intero), ("limbic", lim)):
        net.add_layer(v, name=k)
    kw = dict(update_rule=MSTDPET, nu=1e-3, tc_plus=20.0, tc_minus=20.0, tc_e_trace=200.0, wmin=0.0, wmax=4.0)
    net.add_connection(Connection(ctx, lim, w=torch.rand(n_ctx, n_lim, generator=g) * 0.5, **kw), "ctx", "limbic")
    net.add_connection(Connection(intero, lim, w=torch.rand(n_int, n_lim, generator=g) * 0.5), "intero", "limbic")
    rec = torch.randn(n_lim, n_lim, generator=g) * (torch.rand(n_lim, n_lim, generator=g) < 0.05)
    net.add_connection(Connection(lim, lim, w=rec), "limbic", "limbic")
    return net

for n in (() if __name__ != '__main__' else (764, 1584)):
    net = build(n)
    def ep():
        net.run(inputs={"ctx": (torch.rand(2, 1, 256) < 0.03).float(), "intero": (torch.rand(2, 1, 64) < 0.05).float()},
                time=2, reward=0.0)
    ep()
    t0 = time.perf_counter()
    for _ in range(100): ep()
    t_ep = time.perf_counter() - t0
    net2 = build(n)
    inp = {"ctx": (torch.rand(200, 1, 256) < 0.03).float(), "intero": (torch.rand(200, 1, 64) < 0.05).float()}
    net2.run(inputs={k: v[:2] for k, v in inp.items()}, time=2, reward=0.0)
    t0 = time.perf_counter(); net2.run(inputs=inp, time=200, reward=0.0); t_one = time.perf_counter() - t0
    print(f"BindsNET merged limbic n={n}: 100 epoch calls {t_ep*1e3:.1f} ms/turn; one 200-step call {t_one*1e3:.1f} ms")
