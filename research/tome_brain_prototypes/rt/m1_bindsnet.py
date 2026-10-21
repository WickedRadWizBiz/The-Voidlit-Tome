"""BindsNET per-step cost vs layers, connections, size, learning (dt = 1 ms)."""
import time, warnings, torch
warnings.filterwarnings("ignore")
torch.set_num_threads(1)
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, AdaptiveLIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.learning import MSTDPET

def build(n_layers, n, learn, dense=True):
    net = Network(dt=1.0, batch_size=1, learning=learn)
    net.add_layer(Input(n=64), name="in")
    names = []
    for i in range(n_layers):
        net.add_layer(LIFNodes(n=n, thresh=-52.0, rest=-65.0, reset=-65.0, tc_decay=20.0), name=f"L{i}")
        names.append(f"L{i}")
    g = torch.Generator().manual_seed(0)
    kw = dict(update_rule=MSTDPET, nu=1e-3, tc_plus=20., tc_minus=20., tc_e_trace=100.) if learn else {}
    net.add_connection(Connection(net.layers["in"], net.layers["L0"], w=torch.rand(64, n, generator=g) * 2, wmin=0., wmax=5., **kw), "in", "L0")
    nconn = 1
    for i in range(1, n_layers):
        net.add_connection(Connection(net.layers[names[i-1]], net.layers[names[i]], w=torch.rand(n, n, generator=g) * 0.5, wmin=0., wmax=5., **kw), names[i-1], names[i])
        nconn += 1
    for nm in names:
        net.add_monitor(Monitor(net.layers[nm], state_vars=("s",), time=None), nm)
    return net, nconn

def run(net, T):
    x = (torch.rand(T, 1, 64) < 0.05).float()
    t = time.perf_counter()
    net.run(inputs={"in": x}, time=T, one_step=True, reward=0.0)
    return (time.perf_counter() - t) / T * 1e6

print("layers neurons/layer learn  conns  us/step")
for n_layers in (1, 3, 6):
    for n in (64, 512, 2048):
        for learn in (False, True):
            net, nc = build(n_layers, n, learn)
            run(net, 20)
            us = min(run(net, 200) for _ in range(2))
            print(f"{n_layers:6d} {n:14d} {str(learn):5s} {nc:6d} {us:8.0f}")
