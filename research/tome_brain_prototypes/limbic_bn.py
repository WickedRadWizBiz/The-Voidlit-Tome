"""BindsNET limbic-core prototype (timing): striatum (D1+D2), GPi/SNr, motor thalamus, BLA, hypothalamic need
nuclei, neuromodulator nuclei (VTA/LC/DRN/BF). 8 layers, 13 connections; MSTDPET on ctx->msn and ctx->bla."""
import torch
from bindsnet.learning import MSTDPET
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

CH, PER = 45, 8
SIZES = dict(ctx=256, msn=CH * PER * 2, gpi=CH * 4, vthal=CH * 4, bla=192, hyp=128, mod=104, intero=64)

def build(seed=0):
    g = torch.Generator().manual_seed(seed)
    net = Network(dt=1.0, batch_size=1, learning=True)
    L = {"ctx": Input(n=SIZES["ctx"]), "intero": Input(n=SIZES["intero"])}
    for name in ("msn", "gpi", "vthal", "bla", "hyp", "mod"):
        L[name] = LIFNodes(n=SIZES[name], thresh=-52.0, rest=-65.0, reset=-65.0, tc_decay=20.0, refrac=2)
    for k, v in L.items():
        net.add_layer(v, name=k)
    def rnd(a, b, scale, p=0.2):
        return torch.rand(SIZES[a], SIZES[b], generator=g) * (torch.rand(SIZES[a], SIZES[b], generator=g) < p) * scale
    kw = dict(update_rule=MSTDPET, nu=1e-3, tc_plus=20.0, tc_minus=20.0, tc_e_trace=200.0, wmin=0.0, wmax=4.0)
    conns = [("ctx", "msn", rnd("ctx", "msn", 2.0), kw), ("ctx", "bla", rnd("ctx", "bla", 2.0), kw),
             ("msn", "gpi", -rnd("msn", "gpi", 3.0), {}), ("gpi", "vthal", -rnd("gpi", "vthal", 4.0), {}),
             ("bla", "msn", rnd("bla", "msn", 1.0), {}), ("bla", "hyp", rnd("bla", "hyp", 1.0), {}),
             ("hyp", "mod", rnd("hyp", "mod", 2.0), {}), ("intero", "hyp", rnd("intero", "hyp", 3.0), {}),
             ("mod", "msn", rnd("mod", "msn", 0.5), {}), ("vthal", "vthal", -rnd("vthal", "vthal", 1.0), {}),
             ("bla", "bla", -rnd("bla", "bla", 0.5), {}), ("hyp", "bla", rnd("hyp", "bla", 1.0), {}),
             ("mod", "bla", rnd("mod", "bla", 0.5), {})]
    for a, b, w, extra in conns:
        net.add_connection(Connection(L[a], L[b], w=w, **extra), a, b)
    return net
