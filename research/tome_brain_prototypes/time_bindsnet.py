# BindsNET partition timing: basal ganglia (3 loops, 25 channels), VTA, amygdala. dt = 1 ms, 5-ms sync windows.
import time, torch, warnings
warnings.filterwarnings("ignore")
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, IzhikevichNodes
from bindsnet.network.topology import Connection
from bindsnet.learning import MSTDPET
torch.manual_seed(0); torch.set_num_threads(1)
CH = 25
net = Network(dt=1.0, batch_size=1, learning=True)
L = {}
def add(name, layer): net.add_layer(layer, name=name); L[name] = layer
add("ctx", Input(n=120, traces=True)); add("hyp", Input(n=10, traces=True)); add("sens", Input(n=40, traces=True))
add("D1", IzhikevichNodes(n=6*CH, excitatory=1.0, traces=True)); add("D2", IzhikevichNodes(n=6*CH, excitatory=1.0, traces=True))
for nm, n in (("FSI", 15), ("ChI", 10), ("GPe", 2*CH), ("STN", 20), ("GPi", 2*CH), ("VAVL", 2*CH),
              ("DA", 20), ("VTAg", 10), ("BLA", 40), ("ITC", 10), ("CeA", 20)):
    add(nm, LIFNodes(n=n, thresh=-52.0, rest=-65.0, reset=-65.0, tc_decay=20.0, traces=True))
def C(a, b, p=0.2, w=0.5, rule=None):
    W = (torch.rand(L[a].n, L[b].n) < p).float() * w
    kw = dict(update_rule=MSTDPET, nu=1.0, tc_plus=20.0, tc_minus=20.0, tc_e_trace=250.0, wmin=0.0, wmax=1.0) if rule else {}
    net.add_connection(Connection(L[a], L[b], w=W, **kw), a, b)
C("ctx","D1",rule=1); C("ctx","D2",rule=1); C("ctx","FSI"); C("ctx","STN"); C("FSI","D1",w=-1); C("FSI","D2",w=-1)
C("ChI","D1",w=-0.3); C("ChI","D2",w=-0.3); C("D1","GPi",w=-2); C("D2","GPe",w=-2); C("GPe","GPi",w=-1); C("GPe","STN",w=-1)
C("STN","GPi",w=1); C("STN","GPe",w=1); C("GPi","VAVL",w=-2); C("D1","D1",p=0.1,w=-0.3); C("D2","D2",p=0.1,w=-0.3)
C("hyp","DA",w=2); C("VTAg","DA",w=-1); C("D1","VTAg",w=0.3); C("sens","BLA",rule=1); C("ctx","BLA"); C("BLA","CeA",w=1)
C("ITC","CeA",w=-1); C("ctx","ITC"); C("CeA","DA",w=-0.5); C("BLA","D1",w=0.3)
print("layers", len(net.layers), "connections", len(net.connections))
T = 5
inp = {"ctx": (torch.rand(T, 1, 120) < 0.02).float(), "hyp": (torch.rand(T, 1, 10) < 0.01).float(), "sens": (torch.rand(T, 1, 40) < 0.02).float()}
net.run(inputs=inp, time=T, reward=0.0)
for rep in range(3):
    t0 = time.perf_counter()
    for k in range(50):
        net.run(inputs=inp, time=T, reward=0.0)
    print(f"epoch (50 windows x 5 ms): {1e3*(time.perf_counter()-t0):.0f} ms")
