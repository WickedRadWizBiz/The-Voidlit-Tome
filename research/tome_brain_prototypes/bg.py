"""Prototype BindsNET partition: striatum (D1 GO / D2 NOGO per action channel) + GPi/SNr + motor thalamus,
with MSTDPET on cortex->striatum. 24 action channels (8 beats + 10 disclosure/suggestion + 6 counter)."""
import torch
from bindsnet.learning import MSTDPET
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

A = 24          # action channels
PER = 8         # MSNs per channel per pathway
CTX = 192       # cortical L5 IT/PT afferents arriving via the bus

def build():
    net = Network(dt=1.0, batch_size=1, learning=True)
    ctx = Input(n=CTX)
    d1 = LIFNodes(n=A * PER, thresh=-52.0, rest=-80.0, reset=-80.0, tc_decay=20.0, refrac=2)  # MSNs rest near -80
    d2 = LIFNodes(n=A * PER, thresh=-52.0, rest=-80.0, reset=-80.0, tc_decay=20.0, refrac=2)
    gpi = LIFNodes(n=A * 4, thresh=-55.0, rest=-60.0, reset=-65.0, tc_decay=10.0, refrac=1)  # tonically active
    th = LIFNodes(n=A * 4, thresh=-55.0, rest=-65.0, reset=-70.0, tc_decay=10.0, refrac=2)
    for name, l in (("ctx", ctx), ("d1", d1), ("d2", d2), ("gpi", gpi), ("th", th)):
        net.add_layer(l, name=name)
    kw = dict(update_rule=MSTDPET, nu=1e-3, tc_plus=20.0, tc_minus=20.0, tc_e_trace=200.0, wmin=0.0, wmax=4.0)
    g = torch.Generator().manual_seed(0)
    c1 = Connection(ctx, d1, w=torch.rand(CTX, A * PER, generator=g) * 1.5, **kw)
    c2 = Connection(ctx, d2, w=torch.rand(CTX, A * PER, generator=g) * 1.5, **kw)
    blk = torch.block_diag(*[torch.ones(PER, 4)] * A)
    c3 = Connection(d1, gpi, w=-6.0 * blk)       # direct pathway: GO disinhibits
    c4 = Connection(d2, gpi, w=+3.0 * blk)       # indirect (collapsed via GPe/STN): NOGO excites GPi
    c5 = Connection(gpi, th, w=-8.0 * torch.block_diag(*[torch.ones(4, 4)] * A))
    for c, a, b in ((c1, "ctx", "d1"), (c2, "ctx", "d2"), (c3, "d1", "gpi"), (c4, "d2", "gpi"), (c5, "gpi", "th")):
        net.add_connection(c, a, b)
    return net
