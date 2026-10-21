"""Equivalence: WindowedMSTDPET vs BindsNET MSTDPET (per-step), same spikes, reward constant within 4-step windows."""
import torch
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.learning import MSTDPET
from windowed_rule import WindowedMSTDPET
torch.manual_seed(0)
spk = (torch.rand(400, 1, 30) < 0.08).float()
def run(rule, **kw):
    net = Network(dt=1.0, learning=True)
    a, b = Input(n=30, traces=True), LIFNodes(n=20, thresh=-60.0, traces=True)
    net.add_layer(a, "a"); net.add_layer(b, "b")
    w0 = torch.rand(30, 20, generator=torch.Generator().manual_seed(3))
    c = Connection(a, b, w=w0.clone(), wmin=-10.0, wmax=10.0, update_rule=rule, nu=(0.01, 0.01), tc_plus=20.0,
                   tc_minus=20.0, tc_e_trace=50.0, **kw)
    net.add_connection(c, "a", "b")
    for k in range(100):
        r = float(torch.sin(torch.tensor(k * 0.7)))
        net.run(inputs={"a": spk[4 * k:4 * k + 4]}, time=4, reward=r)
    return c.w.clone(), w0
w_ref, w0 = run(MSTDPET)
w_win, _ = run(WindowedMSTDPET, window=4)
print("max |dw| (reference):", float((w_ref - w0).abs().max()))
print("max |w_windowed - w_reference|:", float((w_win - w_ref).abs().max()))
