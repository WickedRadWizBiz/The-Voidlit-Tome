import warnings; warnings.filterwarnings("ignore")
import torch
from time_limbic2 import build
net = build(640, n_ctx=96)
from bindsnet.network.monitors import Monitor
m = Monitor(net.layers["limbic"], state_vars=("s", "v"), time=20); net.add_monitor(m, "m")
net.run(inputs={"ctx": torch.zeros(20, 1, 96), "intero": (torch.rand(20, 1, 64) < 0.3).float()}, time=20, reward=0.0)
print("spikes", int(m.get("s").sum()), "v range", float(m.get("v").min()), float(m.get("v").max()))
print(net.layers["limbic"].thresh, net.layers["limbic"].rest)
