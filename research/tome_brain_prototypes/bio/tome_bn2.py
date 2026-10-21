"""The Tome's BindsNET partition, merged for speed: 4 layers (cortical/thalamic afferents from the bus, striatum,
pallidum/STN, amygdala) with per-neuron LIF parameter tensors. One MSTDPETSigned rule covers both striatal
pathways: the dopamine reward is multiplied by +1 at D1 columns and -1 at D2 columns (Shen et al. 2008).

Striatum   = D1 MSN (14 ch x 8) | D2 MSN (14 ch x 8) | FSI 16 | VS critic 16          = 256
Pallidum   = GPe (14 x 2) | STN (14 x 2) | GPi/SNr (14 x 2)                            = 84
Amygdala   = BLA 48 (4 valence assemblies of 12) | ITC 8 | CeA 16                       = 72
"""
from __future__ import annotations

import time

import torch
from bindsnet.learning import MSTDPET
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

from windowed_rule import WindowedMSTDPET

PACING = ("calm", "escalate", "clue", "npc", "hazard", "respite", "twist", "advance")
RHETORIC = ("withhold", "disclose", "suggest", "frame_necessity", "corrupt_npc", "bend_player")
CH = len(PACING) + len(RHETORIC)
M = 8
N_CTX = 96 + 24 + 14 + 64 + 32
STR = {"D1": (0, CH * M), "D2": (CH * M, 2 * CH * M), "FSI": (2 * CH * M, 2 * CH * M + 16),
       "VS": (2 * CH * M + 16, 2 * CH * M + 32)}
N_STR = 2 * CH * M + 32
PAL = {"GPe": (0, 2 * CH), "STN": (2 * CH, 4 * CH), "GPi": (4 * CH, 6 * CH)}
N_PAL = 6 * CH
AMY = {"BLA": (0, 48), "ITC": (48, 56), "CeA": (56, 72)}
N_AMY = 72


class MSTDPETSigned(MSTDPET):
    """Three-factor rule with a per-postsynaptic dopamine sign (+1 D1/VS, -1 D2)."""

    def __init__(self, *args, post_sign: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.post_sign = post_sign

    def _connection_update(self, **kwargs) -> None:
        r = kwargs.get("reward", 0.0)
        if self.post_sign is not None:
            # eligibility e_ij is [pre, post]; folding the sign into the trace gives dw = nu * r * sign_j * e_ij
            kwargs = dict(kwargs)
            kwargs["reward"] = 1.0
            sign = self.post_sign
            saved = self.eligibility_trace.clone() if hasattr(self, "eligibility_trace") else None
            super()._connection_update(**dict(kwargs, reward=0.0))   # advance traces, no weight change
            if hasattr(self, "eligibility_trace"):
                with torch.no_grad():
                    self.connection.w += (self.nu[0] * r * self.eligibility_trace * sign).sum(0)
                    self.connection.w.clamp_(self.connection.wmin, self.connection.wmax)
            return
        super()._connection_update(**kwargs)


def lif_tensor(spec: list[tuple[int, float, float, float]]):
    n = sum(s[0] for s in spec)
    th = torch.cat([torch.full((s[0],), s[1]) for s in spec])
    tc = torch.cat([torch.full((s[0],), s[2]) for s in spec])
    rest = torch.cat([torch.full((s[0],), s[3]) for s in spec])
    return LIFNodes(n=n, thresh=th, rest=rest, reset=-75.0, refrac=2, tc_decay=tc, traces=True)


def build(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    net = Network(dt=1.0, batch_size=1, learning=True)
    ctx = Input(n=N_CTX, traces=True)
    stri = lif_tensor([(CH * M, -50.0, 20.0, -80.0), (CH * M, -50.0, 20.0, -80.0), (16, -52.0, 8.0, -70.0),
                       (16, -50.0, 20.0, -80.0)])
    pal = lif_tensor([(2 * CH, -55.0, 10.0, -60.0)] * 3)
    amy = lif_tensor([(48, -52.0, 20.0, -70.0), (8, -52.0, 10.0, -70.0), (16, -52.0, 15.0, -70.0)])
    for name, layer in (("ctx", ctx), ("str", stri), ("pal", pal), ("amy", amy)):
        net.add_layer(layer, name=name)

    def blk(rows, cols, R, Cc, val):
        rows[R[0]:R[1], Cc[0]:Cc[1]] = val

    def chan(n_pre, n_post, w):
        m = torch.zeros(CH * n_pre, CH * n_post)
        for c in range(CH):
            m[c * n_pre:(c + 1) * n_pre, c * n_post:(c + 1) * n_post] = w
        return m

    sign = torch.zeros(N_STR)
    sign[STR["D1"][0]:STR["D1"][1]] = 1.0
    sign[STR["D2"][0]:STR["D2"][1]] = -1.0
    sign[STR["VS"][0]:STR["VS"][1]] = 1.0
    w_cs = 0.6 * torch.rand(N_CTX, N_STR, generator=g)
    c_cs = Connection(ctx, stri, w=w_cs, wmin=0.0, wmax=2.0, update_rule=WindowedMSTDPET, nu=(1e-3, 1e-3),
                      tc_plus=20.0, tc_minus=20.0, tc_e_trace=500.0, post_sign=sign, window=4)
    c_ca = Connection(ctx, amy, w=0.4 * torch.rand(N_CTX, N_AMY, generator=g), wmin=0.0, wmax=2.0,
                      update_rule=WindowedMSTDPET, nu=(1e-3, 1e-3), tc_plus=20.0, tc_minus=20.0, tc_e_trace=500.0,
                      window=4)
    w_ss = torch.zeros(N_STR, N_STR)
    blk(w_ss, None, STR["FSI"], (0, 2 * CH * M), -2.0 * (torch.rand(16, 2 * CH * M, generator=g) < 0.4))
    blk(w_ss, None, STR["D1"], STR["D1"], -0.3 * (torch.rand(CH * M, CH * M, generator=g) < 0.2))
    blk(w_ss, None, STR["D2"], STR["D2"], -0.3 * (torch.rand(CH * M, CH * M, generator=g) < 0.2))
    c_ss = Connection(stri, stri, w=w_ss)
    w_sp = torch.zeros(N_STR, N_PAL)
    blk(w_sp, None, STR["D1"], PAL["GPi"], -chan(M, 2, 3.0))       # direct pathway (GO)
    blk(w_sp, None, STR["D2"], PAL["GPe"], -chan(M, 2, 3.0))       # indirect pathway (NOGO)
    c_sp = Connection(stri, pal, w=w_sp)
    w_pp = torch.zeros(N_PAL, N_PAL)
    blk(w_pp, None, PAL["GPe"], PAL["STN"], -chan(2, 2, 4.0))
    blk(w_pp, None, PAL["GPe"], PAL["GPi"], -chan(2, 2, 2.0))
    blk(w_pp, None, PAL["STN"], PAL["GPi"], torch.full((2 * CH, 2 * CH), 0.8))
    blk(w_pp, None, PAL["STN"], PAL["GPe"], torch.full((2 * CH, 2 * CH), 0.4))
    c_pp = Connection(pal, pal, w=w_pp)
    w_cp = torch.zeros(N_CTX, N_PAL)
    w_cp[:, PAL["STN"][0]:PAL["STN"][1]] = 0.3 * torch.rand(N_CTX, 2 * CH, generator=g)   # hyperdirect
    c_cp = Connection(ctx, pal, w=w_cp)
    w_aa = torch.zeros(N_AMY, N_AMY)
    blk(w_aa, None, AMY["BLA"], AMY["CeA"], 1.0 * (torch.rand(48, 16, generator=g) < 0.3))
    blk(w_aa, None, AMY["BLA"], AMY["ITC"], 1.0 * (torch.rand(48, 8, generator=g) < 0.3))
    blk(w_aa, None, AMY["ITC"], AMY["CeA"], -2.0 * (torch.rand(8, 16, generator=g) < 0.5))
    c_aa = Connection(amy, amy, w=w_aa)
    w_as = torch.zeros(N_AMY, N_STR)
    blk(w_as, None, AMY["BLA"], STR["D1"], 0.3 * (torch.rand(48, CH * M, generator=g) < 0.1))
    blk(w_as, None, AMY["BLA"], STR["VS"], 0.5 * (torch.rand(48, 16, generator=g) < 0.3))
    c_as = Connection(amy, stri, w=w_as)
    for c, a, b in ((c_cs, "ctx", "str"), (c_ca, "ctx", "amy"), (c_ss, "str", "str"), (c_sp, "str", "pal"),
                    (c_pp, "pal", "pal"), (c_cp, "ctx", "pal"), (c_aa, "amy", "amy"), (c_as, "amy", "str")):
        net.add_connection(c, a, b)
    mons = {k: Monitor(net.layers[k], state_vars=("s",), time=4) for k in ("str", "pal", "amy")}
    for k, m in mons.items():
        net.add_monitor(m, k)
    tonic = torch.zeros(N_PAL)
    tonic[PAL["GPe"][0]:PAL["GPe"][1]] = 1.6
    tonic[PAL["STN"][0]:PAL["STN"][1]] = 1.2
    tonic[PAL["GPi"][0]:PAL["GPi"][1]] = 1.8
    return net, mons, tonic


def window(net, tonic, ctx_spikes, da_phasic, da_tonic):
    W = ctx_spikes.shape[0]
    inj_s = torch.zeros(N_STR)
    inj_s[STR["D1"][0]:STR["D1"][1]] = 0.3 * da_tonic
    inj_s[STR["D2"][0]:STR["D2"][1]] = -0.3 * da_tonic
    net.run(inputs={"ctx": ctx_spikes.unsqueeze(1)}, time=W, reward=da_phasic,
            injects_v={"pal": tonic, "str": inj_s})


if __name__ == "__main__":
    torch.set_num_threads(1)
    net, mons, tonic = build(0)
    g = torch.Generator().manual_seed(1)
    W = 4
    for rep in range(3):
        t0 = time.perf_counter()
        cnt = {k: torch.zeros(net.layers[k].n) for k in mons}
        for k in range(250 // W):
            ctx = (torch.rand(W, N_CTX, generator=g) < 0.02).float()
            window(net, tonic, ctx, da_phasic=0.1, da_tonic=0.5)
            for name, m in mons.items():
                cnt[name] += m.get("s").float().sum(dim=(0, 1))
                m.reset_state_variables()
        t1 = time.perf_counter()
        gpi = cnt["pal"][PAL["GPi"][0]:PAL["GPi"][1]].mean() / 0.248
        d1 = cnt["str"][:CH * M].mean() / 0.248
        print(f"BindsNET merged + WindowedMSTDPET, 250 ms in {W}-ms windows: {1e3*(t1-t0):.1f} ms wall | "
              f"GPi {gpi:.0f} Hz, D1 {d1:.1f} Hz")
