"""One clock, one connectome: the Tome's hybrid network co-simulated across JAX, SpikingJelly and BindsNET.

* Master clock: 1 ms bus ticks. Every cross-partition projection has an integer delay >= D_MIN ms.
* Min-delay synchronisation (as in NEST/MUSIC): all partitions advance one window of D_MIN ticks independently,
  reading only bus data written in earlier windows, then exchange. Exact (no extrapolation) because no message
  can be needed sooner than D_MIN ticks after it is sent. The bus asserts this on every read.
* JAX dispatch is asynchronous: the JAX window computes on its own thread (GIL released) while the torch
  partitions run on the Python thread; the exchange is the only join point.
"""
from __future__ import annotations

import io
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/user/The-Voidlit-Tome/brain")
import jaxpath  # noqa: F401
import jax
import jax.numpy as jnp
from spikingjelly.activation_based import functional, neuron, surrogate
from bindsnet.learning import MSTDPET
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

import tome_jax as TJ

torch.set_num_threads(1)
D_MIN = 4

# ------------------------------------------------------------------------------------------------ the bus
class Bus:
    """Ring buffer of spike rasters per source, indexed by master tick. Reads are checked for causality."""

    def __init__(self, sizes: dict[str, int], horizon: int = 64):
        self.h = horizon
        self.buf = {k: np.zeros((horizon, n), np.float32) for k, n in sizes.items()}
        self.written = {k: -1 for k in sizes}      # last tick written per source

    def write(self, src: str, t0: int, x: np.ndarray) -> None:
        D = x.shape[0]
        idx = (np.arange(t0, t0 + D)) % self.h
        self.buf[src][idx] = x
        self.written[src] = t0 + D - 1

    def read(self, src: str, t0: int, D: int, delay: int) -> np.ndarray:
        need = t0 + D - 1 - delay
        assert need <= self.written[src], f"causality: {src} tick {need} not yet written ({self.written[src]})"
        assert delay < self.h - D
        idx = (np.arange(t0, t0 + D) - delay) % self.h
        return self.buf[src][idx]

    def state_dict(self):
        return {"buf": {k: v.copy() for k, v in self.buf.items()}, "written": dict(self.written)}

    def load_state_dict(self, d):
        self.buf = {k: v.copy() for k, v in d["buf"].items()}
        self.written = dict(d["written"])


# ------------------------------------------------------------------------------------------------ SpikingJelly
GROUPS = {"pace": 8, "disclose": 2, "rhetoric": 4, "dominion": 4}   # four parallel cortico-BG-thalamic loops
N_CH = sum(GROUPS.values())                                          # 18 action channels
GROUP_OF = np.concatenate([np.full(n, g) for g, n in enumerate(GROUPS.values())])


def lif(tau, vth=1.0):
    return neuron.LIFNode(tau=tau, v_threshold=vth, v_reset=0.0, decay_input=False,
                          surrogate_function=surrogate.ATan(), detach_reset=True, step_mode="s")


class SJPart(torch.nn.Module):
    """Basal ganglia (4 loops: D1/D2 MSN, FSI, GPe, STN, GPi/SNr, VA/VL) + cerebellum (MF->GrC->PC->DCN, IO).

    dt = 1 ms. BG loops step tick by tick (recurrent); the cerebellar cascade is feed-forward with 1 ms
    synaptic delays, so it runs in multi-step mode over the whole window with a 1-tick carry.
    """
    n_ctx, n_lim, n_mf = 256, 64, 128

    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        M, F, E, S, I, V = 16, 32, 6, 3, 6, 4   # per-channel D1 and D2 MSNs, FSI total, GPe, STN, GPi, VA/VL
        self.sizes = dict(M=M, E=E, S=S, I=I, V=V)
        ch = lambda n: torch.arange(N_CH).repeat_interleave(n)
        def block(n_pre, n_post_per, w):          # channel-topographic matrix
            pre, post = ch(n_pre), ch(n_post_per)
            return (pre[:, None] == post[None, :]).float() * w
        self.register_buffer("w_ctx_d1", torch.rand(self.n_ctx, N_CH * M, generator=g) * 0.05)
        self.register_buffer("w_ctx_d2", torch.rand(self.n_ctx, N_CH * M, generator=g) * 0.05)
        self.register_buffer("w_ctx_d1_prior", self.w_ctx_d1.clone())   # anchor (the designer's instincts)
        self.register_buffer("w_ctx_d2_prior", self.w_ctx_d2.clone())
        self.register_buffer("w_lim_d1", torch.rand(self.n_lim, N_CH * M, generator=g) * 0.03)
        self.register_buffer("w_ctx_fsi", torch.rand(self.n_ctx, F, generator=g) * 0.04)
        self.register_buffer("w_fsi_msn", -torch.rand(F, N_CH * M, generator=g) * 0.08)
        self.register_buffer("w_d1_gpi", -block(M, I, 0.80))
        self.register_buffer("w_d2_gpe", -block(M, E, 0.40))
        self.register_buffer("w_gpe_stn", -block(E, S, 0.25))
        self.register_buffer("w_stn_gpe", block(S, E, 0.20))
        self.register_buffer("w_stn_gpi", (torch.ones(N_CH * S, N_CH * I) * 0.06))   # diffuse hyperdirect
        self.register_buffer("w_gpi_vavl", -block(I, V, 0.50))
        self.register_buffer("w_ctx_vavl", torch.rand(self.n_ctx, N_CH * V, generator=g) * 0.02)
        # cerebellum: 4 mossy fibres per granule cell (fixed), PF->PC plastic, PC -| DCN
        mf_gc = torch.zeros(self.n_mf, 2048)
        for j in range(2048):
            mf_gc[torch.randperm(self.n_mf, generator=g)[:4], j] = 0.45
        self.register_buffer("w_mf_gc", mf_gc)
        self.register_buffer("w_pf_pc", torch.rand(2048, 64, generator=g) * 0.012)
        self.register_buffer("w_pc_dcn", -block(8, 4, 0.5)[: 64, : 32] if False else -torch.kron(torch.eye(8), torch.ones(8, 4)) * 0.25)
        self.register_buffer("w_mf_dcn", torch.rand(self.n_mf, 32, generator=g) * 0.02)
        self.register_buffer("tonic", torch.cat([torch.full((N_CH * E,), 0.14), torch.full((N_CH * S,), 0.13),
                                                 torch.full((N_CH * I,), 0.15)]))
        self.d1, self.d2, self.fsi = lif(10.0), lif(10.0), lif(5.0)
        self.gpe, self.stn, self.gpi, self.vavl = lif(8.0), lif(6.0), lif(8.0), lif(10.0)
        self.gc, self.pc, self.dcn = lif(5.0), lif(10.0), lif(10.0)
        for m in (self.gc, self.pc, self.dcn):
            functional.set_step_mode(m, "m")
        self.carry = {k: torch.zeros(1, n) for k, n in dict(
            d1=N_CH * M, d2=N_CH * M, fsi=F, gpe=N_CH * E, stn=N_CH * S, gpi=N_CH * I, gc=2048, pc=64).items()}
        self.prior_bias = torch.zeros(N_CH * M)     # Director habit pathway + rules mask, set per turn
        self.nogo = torch.zeros(N_CH * I)            # rules-forbidden channels: hard tonic GPi drive
        self.elig = torch.zeros(self.n_ctx, N_CH * M)
        self.ctx_trace = torch.zeros(1, self.n_ctx)
        self.noise_gen = torch.Generator().manual_seed(seed + 99)
        self.vavl_bias = 0.15
        self.hyper = 1.5
        self.fuse()

    def fuse(self):
        """Pack the BG loops into one population (order D1, D2, FSI, GPe, STN, GPi, VA/VL) and one recurrent
        matrix, so a tick is one matmul + one SpikingJelly LIF call. Input currents for the whole window come
        from one multi-step matmul (bus inputs are known at the window start)."""
        M, E, S, I, V = (self.sizes[k] for k in "MESIV")
        F = self.w_ctx_fsi.shape[1]
        sz = [N_CH * M, N_CH * M, F, N_CH * E, N_CH * S, N_CH * I, N_CH * V]
        off = np.cumsum([0] + sz)
        self.off = {k: (int(off[i]), int(off[i + 1])) for i, k in enumerate(["d1", "d2", "fsi", "gpe", "stn", "gpi", "vavl"])}
        n = int(off[-1])
        R = torch.zeros(n, n)
        def put(src, dst, w):
            a, b = self.off[src]; c, d = self.off[dst]
            R[a:b, c:d] = w
        put("fsi", "d1", self.w_fsi_msn); put("fsi", "d2", self.w_fsi_msn)
        put("d1", "gpi", self.w_d1_gpi); put("d2", "gpe", self.w_d2_gpe)
        put("gpe", "stn", self.w_gpe_stn); put("stn", "gpe", self.w_stn_gpe); put("stn", "gpi", self.w_stn_gpi)
        put("gpi", "vavl", self.w_gpi_vavl)
        self.register_buffer("R", R)
        self.RT_csr = R.T.contiguous().to_sparse_csr()     # block-sparse recurrent BG matrix (3% dense)
        self.n_bg = n
        self.bg = lif(8.0)
        self.bg_prev = torch.zeros(1, n)

    def _rec(self, prev):
        return (self.RT_csr @ prev.T).T

    def step_window(self, ctx, lim, mf, da):
        """ctx [D,256], lim [D,64], mf [D,128], da [D] -> dict of [D, n] spike rasters."""
        D = ctx.shape[0]
        o = self.off
        # window input currents (multi-step): cortex/limbic -> D1, D2, FSI, VA/VL; tonic drives; DA gain
        I_in = torch.zeros(D, self.n_bg)
        g1 = (1 + 0.5 * da)[:, None]; g2 = (1 - 0.4 * da)[:, None]
        I_in[:, o["d1"][0]:o["d1"][1]] = (ctx @ self.w_ctx_d1 + lim @ self.w_lim_d1) * g1 + self.prior_bias
        I_in[:, o["d2"][0]:o["d2"][1]] = (ctx @ self.w_ctx_d2) * g2
        I_in[:, o["fsi"][0]:o["fsi"][1]] = ctx @ self.w_ctx_fsi
        I_in[:, o["gpe"][0]:o["stn"][1]] = self.tonic[: o["stn"][1] - o["gpe"][0]]
        # hyperdirect pathway (GPR 2001): cortex -> STN carries the pooled command salience of each loop,
        # so GPi is raised diffusely in that loop while D1 suppresses it focally in the strongest channel
        d1_in = I_in[:, o["d1"][0]:o["d1"][1]].reshape(D, N_CH, self.sizes["M"]).mean(2)        # [D, ch]
        pooled = torch.stack([d1_in[:, GROUP_OF == g].sum(1) for g in range(len(GROUPS))], 1)   # [D, groups]
        stn_ch = pooled[:, GROUP_OF].repeat_interleave(self.sizes["S"], 1)
        I_in[:, o["stn"][0]:o["stn"][1]] += self.hyper * stn_ch
        I_in[:, o["gpi"][0]:o["gpi"][1]] = self.tonic[o["stn"][1] - o["gpe"][0]:] + self.nogo
        I_in[:, o["vavl"][0]:o["vavl"][1]] = ctx @ self.w_ctx_vavl + self.vavl_bias
        noise = 0.04 * torch.randn(D, o["d2"][1], generator=self.noise_gen)
        I_in[:, : o["d2"][1]] += noise
        out = torch.empty(D, self.n_bg)
        prev = self.bg_prev
        for t in range(D):
            prev = self.bg(I_in[t:t + 1] + self._rec(prev))
            out[t] = prev[0]
        self.bg_prev = prev
        # three-factor eligibility (one matmul per window): cortical trace x D1 spikes
        tr = torch.empty(D, self.n_ctx)
        c = self.ctx_trace
        for t in range(D):
            c = 0.9 * c + ctx[t:t + 1]
            tr[t] = c[0]
        self.ctx_trace = c
        self.elig.add_(tr.T @ out[:, o["d1"][0]:o["d1"][1]])
        # cerebellum, multi-step with 1-tick carries (feed-forward cascade)
        cr = self.carry
        gc = self.gc((mf @ self.w_mf_gc).unsqueeze(1))                         # [D,1,2048]
        gc_in = torch.cat([cr["gc"].unsqueeze(0), gc[:-1]], 0)
        pc = self.pc(gc_in @ self.w_pf_pc)                                     # [D,1,64]
        pc_in = torch.cat([cr["pc"].unsqueeze(0), pc[:-1]], 0)
        dcn = self.dcn(pc_in @ self.w_pc_dcn + (mf @ self.w_mf_dcn).unsqueeze(1) + 0.3)
        cr["gc"], cr["pc"] = gc[-1], pc[-1]
        sl = lambda k: out[:, o[k][0]:o[k][1]]
        return {"vavl": sl("vavl"), "gpi": sl("gpi"), "d1": sl("d1"), "dcn": dcn[:, 0], "pc": pc[:, 0]}

    def consolidate(self, rpe: float, eta=2e-4, anchor=0.01):
        """Retroactive three-factor update (last turn's choice, paid by this turn's dopamine), pulled toward the
        Director prior so the designer's pacing instincts remain the starting point."""
        dw = eta * rpe * torch.tanh(0.05 * self.elig)
        self.w_ctx_d1.add_(dw).add_(anchor * (self.w_ctx_d1_prior - self.w_ctx_d1)).clamp_(0.0, 0.2)
        self.elig.zero_()

    def memories(self):
        return {n: m.v if isinstance(m.v, torch.Tensor) else torch.tensor(m.v)
                for n, m in self.named_modules() if isinstance(m, neuron.BaseNode)}

    def full_state(self):
        return {"buffers": {k: v.clone() for k, v in self.state_dict().items()},
                "v": {k: v.clone() for k, v in self.memories().items()},
                "carry": {k: v.clone() for k, v in self.carry.items()},
                "bg_prev": self.bg_prev.clone(), "prior_bias": self.prior_bias.clone(), "nogo": self.nogo.clone(), "elig": self.elig.clone(),
                "ctx_trace": self.ctx_trace.clone(), "noise_gen": self.noise_gen.get_state()}

    def load_full_state(self, d):
        self.load_state_dict(d["buffers"])
        mods = dict(self.named_modules())
        for k, v in d["v"].items():
            mods[k].v = v.clone()
        self.carry = {k: v.clone() for k, v in d["carry"].items()}
        for k in ("bg_prev", "prior_bias", "nogo", "elig", "ctx_trace"):
            setattr(self, k, d[k].clone())
        self.noise_gen.set_state(d["noise_gen"])


# ------------------------------------------------------------------------------------------------ BindsNET
class BNPart:
    """Limbic core: hypothalamic need nuclei, BLA valence (R-STDP via MSTDPET), CeA, VTA (dopamine), LC (NE)."""

    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        net = Network(dt=1.0, batch_size=1, learning=True)
        self.layers = dict(ctx=Input(n=64), HYP=LIFNodes(n=48, thresh=-52., rest=-65., reset=-65., tc_decay=20.),
                           BLA=LIFNodes(n=64, thresh=-52., rest=-65., reset=-65., tc_decay=20.),
                           CeA=LIFNodes(n=16, thresh=-52., rest=-65., reset=-65., tc_decay=20.),
                           VTA=LIFNodes(n=16, thresh=-52., rest=-65., reset=-65., tc_decay=20.),
                           LC=LIFNodes(n=8, thresh=-52., rest=-65., reset=-65., tc_decay=20.))
        for k, v in self.layers.items():
            net.add_layer(v, name=k)
        L = self.layers
        self.c_ctx_bla = Connection(L["ctx"], L["BLA"], w=torch.rand(64, 64, generator=g) * 4.0, wmin=0., wmax=8.,
                                    update_rule=MSTDPET, nu=1.0, tc_plus=20., tc_minus=20., tc_e_trace=200.)
        net.add_connection(self.c_ctx_bla, "ctx", "BLA")
        pos_neg = torch.zeros(64, 64); pos_neg[:32, 32:] = -3.0; pos_neg[32:, :32] = -3.0
        net.add_connection(Connection(L["BLA"], L["BLA"], w=pos_neg), "BLA", "BLA")
        net.add_connection(Connection(L["BLA"], L["CeA"], w=torch.rand(64, 16, generator=g) * 3.0), "BLA", "CeA")
        net.add_connection(Connection(L["HYP"], L["CeA"], w=torch.rand(48, 16, generator=g) * 2.0), "HYP", "CeA")
        net.add_connection(Connection(L["BLA"], L["VTA"], w=torch.cat([torch.rand(32, 16, generator=g) * 2.5,
                                                                      -torch.rand(32, 16, generator=g) * 2.5])), "BLA", "VTA")
        net.add_connection(Connection(L["CeA"], L["LC"], w=torch.rand(16, 8, generator=g) * 4.0), "CeA", "LC")
        self.mon = {k: Monitor(L[k], state_vars=("s",), time=None) for k in ("HYP", "BLA", "CeA", "VTA", "LC")}
        for k, m in self.mon.items():
            net.add_monitor(m, k)
        self.net = net
        self.needs = np.array([0.8, 0.5, 0.4, 0.6], np.float64)   # survival, affiliation, certainty, competence
        self.vta_drive = 0.0                                        # phasic RPE drive, set per turn

    def step_window(self, ctx, hyp_drive, D):
        for m in self.mon.values():
            m.reset_state_variables()
        inj = {"HYP": torch.as_tensor(hyp_drive, dtype=torch.float32),
               "VTA": torch.full((D, 16), 1.6 + self.vta_drive, dtype=torch.float32)}
        self.net.run(inputs={"ctx": torch.as_tensor(ctx).unsqueeze(1)}, time=D, one_step=True, reward=0.0,
                     injects_v=inj)
        return {k: m.get("s")[:, 0].float() for k, m in self.mon.items()}

    def state_bytes(self):
        bio = io.BytesIO()
        torch.save({"net": self.net, "needs": self.needs, "vta_drive": self.vta_drive}, bio)
        return bio.getvalue()

    def load_bytes(self, b):
        d = torch.load(io.BytesIO(b), weights_only=False)
        self.net, self.needs, self.vta_drive = d["net"], d["needs"], d["vta_drive"]
        self.layers = {k: self.net.layers[k] for k in self.layers}
        self.mon = {k: self.net.monitors[k] for k in self.mon}
        self.c_ctx_bla = self.net.connections[("ctx", "BLA")]


# ------------------------------------------------------------------------------------------------ the Tome
class Tome:
    """The single hybrid network. One master clock (self.t, ms), one bus, three partitions."""

    def __init__(self, seed=0, window=D_MIN, sub=2, turn_ms=200, s_max=64):
        assert window <= D_MIN, "a window longer than the minimum cross-partition delay breaks causality"
        self.window, self.turn_ms, self.sub, self.s_max = window, turn_ms, sub, s_max
        self.P, self.meta = TJ.build(seed=seed, dt_sub=1.0 / sub)
        self.L = self.meta["L"]
        self.jrun = TJ.make_window(self.P, self.meta, window, sub, s_max=s_max, flags=dict(bg_inside=True))
        self.jst = TJ.init_state(self.P, self.meta, jax.random.PRNGKey(seed))
        self.sj = SJPart(seed)
        self.bn = BNPart(seed)
        N = self.L.n
        self.bus = Bus({"jax": N, "vavl": N_CH * 4, "dcn": 32, "VTA": 16, "LC": 8, "BLA": 64, "HYP": 48, "CeA": 16})
        rng = np.random.default_rng(seed + 7)
        sel = lambda p, k: self.L.start[p] + np.arange(min(k, self.L.size[p]))
        self.idx_ctx = np.concatenate([np.concatenate([sel(f"{c}.L5ET", 32), sel(f"{c}.L5IT", 32)]) for c in TJ.COLS])
        self.idx_mf = np.concatenate([sel("TC", 64), sel("PLAN.L5ET", 32), sel("TOM.L5ET", 32)])
        self.idx_lim = np.concatenate([sel("CA1", 32), sel("SELF.L5ET", 32)])
        # projection matrices into the JAX partition (pA per spike), fixed and seeded
        def proj(n_src, target_pop, k, amp):
            M = np.zeros((n_src, N), np.float32)
            tgt = self.L.start[target_pop] + np.arange(self.L.size[target_pop])
            for i in range(n_src):
                M[i, rng.choice(tgt, size=k, replace=False)] = amp
            return M
        self.M_vavl = proj(N_CH * 4, "PLAN.L4", 6, 160.0)       # motor thalamus -> PLAN column (decision readback)
        self.M_dcn = proj(32, "TC", 4, 140.0)                    # cerebellar nuclei -> VL thalamus (TC)
        self.M_bla = proj(64, "SELF.L4", 4, 120.0)               # amygdala -> SELF column (interoception/affect)
        self.M_hyp = proj(48, "SELF.L23", 4, 100.0)              # needs -> SELF column
        self.delays = {"jax->sj": 5, "jax->bn": 8, "vavl->jax": 4, "dcn->jax": 5, "bla->jax": 8, "hyp->jax": 8,
                       "vta->all": 12, "lc->jax": 10, "bla->sj": 6}
        self.delays = {k: max(v, window) for k, v in self.delays.items()}   # floor at the window (min delay)
        assert min(self.delays.values()) >= window
        self.t = 0
        self.game_drive = np.zeros(N, np.float32)
        self.lumen = 1.0
        self.da_level = 0.0
        self.log = []

    # -- per-turn interface ------------------------------------------------------------------------
    def set_game_inputs(self, features: np.ndarray, lumen: float):
        """features in [0,1], 64 channels -> 2 TC cells each (thalamic core relay of the game state)."""
        tc = self.L.start["TC"]
        self.game_drive[:] = 0.0
        self.game_drive[tc:tc + 128] = 220.0 * np.repeat(np.clip(features, 0, 1), 2).astype(np.float32)
        self.lumen = float(lumen)

    def run_turn(self, ms=None, record=False, until=None, min_ms=120):
        """Advance the whole network. With `until`, commit adaptively: after min_ms, stop at the first window
        where until(vavl_counts_this_turn) is true (the GO gate), else at ms."""
        ms = ms or self.turn_ms
        W = self.window
        rec = {"jax": [], "vavl": [], "VTA": []}
        self.turn_counts = np.zeros(N_CH, np.float32)
        self.turn_elapsed = 0
        for _ in range(ms // W):
            if until is not None and self.turn_elapsed >= min_ms and until(self.turn_counts):
                break
            t0 = self.t
            # 1. read every partition's inputs from the bus (all written in earlier windows)
            vavl = self.bus.read("vavl", t0, W, self.delays["vavl->jax"])
            dcn = self.bus.read("dcn", t0, W, self.delays["dcn->jax"])
            bla_j = self.bus.read("BLA", t0, W, self.delays["bla->jax"])
            hyp_j = self.bus.read("HYP", t0, W, self.delays["hyp->jax"])
            vta = self.bus.read("VTA", t0, W, self.delays["vta->all"]).mean(1)
            lc = self.bus.read("LC", t0, W, self.delays["lc->jax"]).mean(1)
            jx_sj = self.bus.read("jax", t0, W, self.delays["jax->sj"])
            jx_bn = self.bus.read("jax", t0, W, self.delays["jax->bn"])
            bla_s = self.bus.read("BLA", t0, W, self.delays["bla->sj"])
            da = np.clip(4.0 * (vta - 0.02), -1, 1).astype(np.float32)
            bus_I = (self.game_drive[None, :] + vavl @ self.M_vavl + dcn @ self.M_dcn + bla_j @ self.M_bla
                     + hyp_j @ self.M_hyp)
            # 2. dispatch JAX (async; runs on XLA's thread while torch works below)
            self.jst, jspk = self.jrun(self.jst, jnp.asarray(bus_I), self.lumen, jnp.asarray(da), jnp.asarray(lc))
            # 3. torch partitions
            with torch.no_grad():
                so = self.sj.step_window(torch.from_numpy(jx_sj[:, self.idx_ctx]), torch.from_numpy(bla_s),
                                         torch.from_numpy(jx_sj[:, self.idx_mf]), torch.from_numpy(da))
            need_def = (1.0 - self.bn.needs)                                   # deficits drive the need nuclei
            hyp_drive = np.repeat(need_def, 12)[None, :].repeat(W, 0) * 9.0
            bo = self.bn.step_window(jx_bn[:, self.idx_lim], hyp_drive, W)
            # 4. join + exchange
            jsp = np.asarray(jspk)
            self.bus.write("jax", t0, jsp)
            self.bus.write("vavl", t0, so["vavl"].numpy()); self.bus.write("dcn", t0, so["dcn"].numpy())
            for k in ("VTA", "LC", "BLA", "HYP", "CeA"):
                self.bus.write(k, t0, bo[k].numpy())
            if record:
                rec["jax"].append(jsp); rec["vavl"].append(so["vavl"].numpy()); rec["VTA"].append(bo["VTA"].numpy())
            self.turn_counts += so["vavl"].numpy().reshape(W, N_CH, 4).sum((0, 2))
            self.turn_elapsed += W
            self.t += W
        return {k: np.concatenate(v) for k, v in rec.items()} if record else None

    def readout(self, ms):
        """Channel decisions per loop from VA/VL counts over the last `ms` ticks (read straight off the bus)."""
        h = self.bus.buf["vavl"][(np.arange(self.t - ms, self.t)) % self.bus.h]
        counts = h.reshape(ms, N_CH, 4).sum((0, 2))
        out, o = {}, 0
        for g, n in GROUPS.items():
            out[g] = int(np.argmax(counts[o:o + n])); o += n
        return out, counts

    # -- persistence -------------------------------------------------------------------------------
    def state(self):
        return {"t": self.t, "jax": jax.tree_util.tree_map(np.asarray, self.jst), "sj": self.sj.full_state(),
                "bn": self.bn.state_bytes(), "bus": self.bus.state_dict(), "game_drive": self.game_drive.copy(),
                "lumen": self.lumen, "P": self.P, "meta": {k: v for k, v in self.meta.items()},
                "sub": self.sub, "s_max": self.s_max,
                "bus_proj": {k: getattr(self, k).copy() for k in ("M_vavl", "M_dcn", "M_bla", "M_hyp", "idx_ctx", "idx_mf", "idx_lim")},
                "delays": dict(self.delays)}

    def load(self, d):
        self.t = d["t"]
        # the connectome is state too (microglia prune it, oligodendrocytes re-time it): restore, then re-jit
        self.P, self.meta = d["P"], d["meta"]
        self.L = self.meta["L"]
        self.jrun = TJ.make_window(self.P, self.meta, self.window, d["sub"], s_max=d["s_max"], flags=dict(bg_inside=True))
        self.jst = jax.tree_util.tree_map(jnp.asarray, d["jax"])
        for k, v in d["bus_proj"].items():
            setattr(self, k, v.copy())
        self.delays = dict(d["delays"])
        self.sj.load_full_state(d["sj"])
        self.bn.load_bytes(d["bn"])
        self.bus.load_state_dict(d["bus"])
        self.game_drive = d["game_drive"].copy()
        self.lumen = d["lumen"]
