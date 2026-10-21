import time, torch, warnings
warnings.filterwarnings("ignore")
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, IzhikevichNodes
from bindsnet.network.topology import Connection
from bindsnet.learning import MSTDPET
torch.manual_seed(0)
for threads in (1, 2):
    torch.set_num_threads(threads)
    net = Network(dt=1.0, batch_size=1, learning=True)
    ctx = Input(n=160, traces=True); sens = Input(n=48, traces=True)
    STR = IzhikevichNodes(n=315, excitatory=1.0, traces=True)        # D1 150 + D2 150 + FSI 15 in one population
    BLA = LIFNodes(n=50, thresh=-52.0, rest=-65.0, reset=-65.0, tc_decay=20.0, traces=True)
    for nm, l in (("ctx", ctx), ("sens", sens), ("STR", STR), ("BLA", BLA)): net.add_layer(l, name=nm)
    kw = dict(update_rule=MSTDPET, nu=1.0, tc_plus=20.0, tc_minus=20.0, tc_e_trace=250.0, wmin=0.0, wmax=1.0)
    net.add_connection(Connection(ctx, STR, w=0.3*(torch.rand(160, 315) < 0.2).float(), **kw), "ctx", "STR")
    net.add_connection(Connection(STR, STR, w=-0.3*(torch.rand(315, 315) < 0.1).float()), "STR", "STR")
    net.add_connection(Connection(sens, BLA, w=0.4*(torch.rand(48, 50) < 0.3).float(), **kw), "sens", "BLA")
    net.add_connection(Connection(ctx, BLA, w=0.3*(torch.rand(160, 50) < 0.1).float()), "ctx", "BLA")
    net.add_connection(Connection(BLA, STR, w=0.2*(torch.rand(50, 315) < 0.1).float()), "BLA", "STR")
    T = 5
    inp = {"ctx": (torch.rand(T, 1, 160) < 0.02).float(), "sens": (torch.rand(T, 1, 48) < 0.02).float()}
    net.run(inputs=inp, time=T, reward=0.0)
    for rep in range(3):
        t0 = time.perf_counter()
        for k in range(50):
            net.run(inputs=inp, time=T, reward=0.0)
        if rep == 2: print(f"threads={threads}: lean BindsNET partition, epoch (50 x 5 ms) = {1e3*(time.perf_counter()-t0):.0f} ms")
