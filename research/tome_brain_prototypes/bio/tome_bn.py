"""The Tome's BindsNET partition: basal ganglia (two parallel loops) and amygdala, with dopamine three-factor
learning (MSTDPET; D2 synapses use the sign-inverted rule) and DA-dependent MSN excitability.

Channels: pacing loop = the Director's 8 beats; rhetoric loop = withhold, disclose, suggest, frame_necessity,
corrupt_npc, bend_player. Selection is read from GPi/SNr pauses (disinhibition of the matrix thalamus in JAX).
"""
from __future__ import annotations

import time

import torch
from bindsnet.learning import MSTDPET
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

PACING = ("calm", "escalate", "clue", "npc", "hazard", "respite", "twist", "advance")
RHETORIC = ("withhold", "disclose", "suggest", "frame_necessity", "corrupt_npc", "bend_player")
CH = len(PACING) + len(RHETORIC)     # 14
MSN_PER = 8
N_CTX = 96 + 24 + 14 + 64 + 32      # L5_IT x4, P.L5_ET, P.TC_mtx, CA1, Director-prior habit layer (SJ)


class MSTDPETInverted(MSTDPET):
    """Indirect-pathway (D2) plasticity: a dopamine dip potentiates, a burst depresses (Shen et al. 2008)."""

    def _connection_update(self, **kwargs) -> None:
        kwargs = dict(kwargs)
        kwargs["reward"] = -kwargs.get("reward", 0.0)
        super()._connection_update(**kwargs)


def lif(n, thresh, tc, rest=-70.0, reset=-75.0, refrac=2):
    return LIFNodes(n=n, thresh=thresh, rest=rest, reset=reset, refrac=refrac, tc_decay=tc, traces=True)


def build(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    net = Network(dt=1.0, batch_size=1, learning=True)
    L = {
        "ctx": Input(n=N_CTX, traces=True),
        "D1": lif(CH * MSN_PER, -50.0, 20.0, rest=-80.0), "D2": lif(CH * MSN_PER, -50.0, 20.0, rest=-80.0),
        "FSI": lif(16, -52.0, 8.0), "VS": lif(16, -50.0, 20.0, rest=-80.0),
        "GPe": lif(CH * 2, -55.0, 10.0, rest=-60.0), "STN": lif(CH * 2, -55.0, 10.0, rest=-60.0),
        "GPi": lif(CH * 2, -55.0, 10.0, rest=-60.0),
        "BLA": lif(48, -52.0, 20.0), "ITC": lif(8, -52.0, 10.0), "CeA": lif(16, -52.0, 15.0),
    }
    for k, v in L.items():
        net.add_layer(v, name=k)

    def chan_block(n_pre_per, n_post_per, w, sign=1.0):
        m = torch.zeros(CH * n_pre_per, CH * n_post_per)
        for c in range(CH):
            m[c * n_pre_per:(c + 1) * n_pre_per, c * n_post_per:(c + 1) * n_post_per] = w
        return sign * m

    kw = dict(update_rule=MSTDPET, nu=(1e-3, 1e-3), tc_plus=20.0, tc_minus=20.0, tc_e_trace=500.0)
    C = {}
    C[("ctx", "D1")] = Connection(L["ctx"], L["D1"], w=0.6 * torch.rand(N_CTX, CH * MSN_PER, generator=g),
                                 wmin=0.0, wmax=2.0, **kw)
    kw2 = dict(kw, update_rule=MSTDPETInverted)
    C[("ctx", "D2")] = Connection(L["ctx"], L["D2"], w=0.6 * torch.rand(N_CTX, CH * MSN_PER, generator=g),
                                 wmin=0.0, wmax=2.0, **kw2)
    C[("ctx", "VS")] = Connection(L["ctx"], L["VS"], w=0.4 * torch.rand(N_CTX, 16, generator=g), wmin=0.0,
                                 wmax=2.0, **kw)
    C[("ctx", "BLA")] = Connection(L["ctx"], L["BLA"], w=0.4 * torch.rand(N_CTX, 48, generator=g), wmin=0.0,
                                  wmax=2.0, **kw)
    C[("ctx", "FSI")] = Connection(L["ctx"], L["FSI"], w=0.5 * torch.rand(N_CTX, 16, generator=g))
    C[("ctx", "STN")] = Connection(L["ctx"], L["STN"], w=0.3 * torch.rand(N_CTX, CH * 2, generator=g))  # hyperdirect
    C[("FSI", "D1")] = Connection(L["FSI"], L["D1"], w=-2.0 * (torch.rand(16, CH * MSN_PER, generator=g) < 0.4))
    C[("FSI", "D2")] = Connection(L["FSI"], L["D2"], w=-2.0 * (torch.rand(16, CH * MSN_PER, generator=g) < 0.4))
    C[("D1", "D1")] = Connection(L["D1"], L["D1"], w=-0.3 * (torch.rand(CH * MSN_PER, CH * MSN_PER, generator=g) < 0.2))
    C[("D1", "GPi")] = Connection(L["D1"], L["GPi"], w=chan_block(MSN_PER, 2, 3.0, -1.0))   # direct: GO
    C[("D2", "GPe")] = Connection(L["D2"], L["GPe"], w=chan_block(MSN_PER, 2, 3.0, -1.0))   # indirect: NOGO
    C[("GPe", "STN")] = Connection(L["GPe"], L["STN"], w=chan_block(2, 2, 4.0, -1.0))
    C[("GPe", "GPi")] = Connection(L["GPe"], L["GPi"], w=chan_block(2, 2, 2.0, -1.0))
    C[("STN", "GPi")] = Connection(L["STN"], L["GPi"], w=torch.full((CH * 2, CH * 2), 0.8))        # diffuse
    C[("STN", "GPe")] = Connection(L["STN"], L["GPe"], w=torch.full((CH * 2, CH * 2), 0.4))
    C[("BLA", "CeA")] = Connection(L["BLA"], L["CeA"], w=1.0 * (torch.rand(48, 16, generator=g) < 0.3))
    C[("BLA", "ITC")] = Connection(L["BLA"], L["ITC"], w=1.0 * (torch.rand(48, 8, generator=g) < 0.3))
    C[("ITC", "CeA")] = Connection(L["ITC"], L["CeA"], w=-2.0 * (torch.rand(8, 16, generator=g) < 0.5))
    C[("BLA", "D1")] = Connection(L["BLA"], L["D1"], w=0.3 * (torch.rand(48, CH * MSN_PER, generator=g) < 0.1))
    C[("BLA", "VS")] = Connection(L["BLA"], L["VS"], w=0.5 * (torch.rand(48, 16, generator=g) < 0.3))
    for (a, b), c in C.items():
        net.add_connection(c, a, b)
    mons = {k: Monitor(L[k], state_vars=("s",), time=4) for k in ("GPi", "CeA", "BLA", "VS", "D1")}
    for k, m in mons.items():
        net.add_monitor(m, k)
    tonic = {"GPe": torch.full((CH * 2,), 1.6), "STN": torch.full((CH * 2,), 1.2),
             "GPi": torch.full((CH * 2,), 1.8)}  # mV/ms pacemaking (GPe ~60 Hz, STN ~20 Hz, GPi ~70 Hz)
    return net, L, C, mons, tonic


def window(net, tonic, ctx_spikes, da_phasic: float, da_tonic: float):
    """Advance W ms: ctx_spikes [W, N_CTX] from the bus; DA scales MSN excitability and is the MSTDPET reward."""
    W = ctx_spikes.shape[0]
    inj = dict(tonic)
    inj["D1"] = torch.full((CH * MSN_PER,), 0.3 * da_tonic)     # D1: DA raises excitability
    inj["D2"] = torch.full((CH * MSN_PER,), -0.3 * da_tonic)    # D2: DA lowers it
    net.run(inputs={"ctx": ctx_spikes.unsqueeze(1)}, time=W, reward=da_phasic, injects_v=inj)


if __name__ == "__main__":
    torch.set_num_threads(1)
    net, L, C, mons, tonic = build(0)
    g = torch.Generator().manual_seed(1)
    W = 4
    for rep in range(3):
        t0 = time.perf_counter()
        counts = {k: 0.0 for k in mons}
        for k in range(250 // W):
            ctx = (torch.rand(W, N_CTX, generator=g) < 0.01).float()
            window(net, tonic, ctx, da_phasic=0.1, da_tonic=0.5)
            for name, m in mons.items():
                counts[name] += m.get("s").float().sum().item()
                m.reset_state_variables()
        t1 = time.perf_counter()
        rates = {k: v / 0.248 / L[k].n for k, v in counts.items()}
        print(f"BindsNET BG+amygdala, 250 ms in {W}-ms windows: {1e3*(t1-t0):.1f} ms wall | rates Hz " +
              " ".join(f"{k}={v:.0f}" for k, v in rates.items()))
