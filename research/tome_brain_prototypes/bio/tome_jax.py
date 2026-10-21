"""The Tome's JAX partition: 4 cortical columns (6 layers), core/matrix thalamus + TRN, hippocampus, the
neuromodulatory and hypothalamic nuclei, astrocytes (De Pitta/Li-Rinzel), the ATP/K_ATP energy budget, Tsodyks-
Markram STP, Graupner-Brunel calcium plasticity and Vogels iSTDP. One call advances W_SYNC ms.

Timing-faithful prototype: real shapes, real update rules, plausible (untuned) connectivity.
dt = 0.25 ms for membranes (exponential Euler), 1 ms for synaptic delivery, plasticity, glia and energy.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import jaxpath  # noqa: F401
import jax
import jax.numpy as jnp

import cells
from cells import substep

DT = 0.25
SUB = 4             # substeps per ms
W_SYNC = 4          # ms per call = minimum cross-partition delay
D_RING = 32         # ms of spike history (axonal delays up to 31 ms)
K_E, K_I = 96, 48   # max spikes gathered per ms (overflow is counted, never silently dropped)
ABL: set = set()    # profiling ablations
GAIN = dict(e=1.0, i=1.0)   # global synaptic gain (calibrated against layer rate targets)

COLS = ("S", "P", "T", "I")  # reader, schemer, mind-reader (TPJ), interoception (insula)
COLUMN = [("L1_NGF", 8, "ngf"), ("L23_PC", 96, "pc23h"), ("L23_PV", 12, "pv"), ("L23_SST", 8, "sst"),
          ("L23_VIP", 8, "vip"), ("L4_SS", 40, "rs"), ("L4_PV", 6, "pv"), ("L5_ET", 24, "pc5"),
          ("L5_IT", 24, "rs"), ("L5_PV", 6, "pv"), ("L5_SST", 6, "sst"), ("L6_CT", 32, "rs"), ("L6_PV", 4, "pv"),
          ("L6_SST", 4, "sst")]
THAL = [("TC_core", 32, "tc"), ("TC_mtx", 14, "tc"), ("TRN", 10, "trn")]
HIPPO = [("DG", 128, "dg"), ("CA3", 96, "rs"), ("CA3_PV", 12, "pv"), ("CA1", 64, "pc5"), ("CA1_PV", 8, "pv"),
         ("CA1_OLM", 8, "sst")]
NUCLEI = [("VTA_DA", 24, "da"), ("VTA_GABA", 8, "pv"), ("LC", 16, "pace"), ("DRN", 16, "pace"),
          ("NBM", 16, "pace"), ("ARC", 12, "rs"), ("PVN", 12, "rs")]
INHIB = {"pv", "sst", "vip", "ngf", "trn"}


def layout():
    pops, kinds = {}, []
    def add(name, n, kind):
        pops[name] = (len(kinds), len(kinds) + n)
        kinds.extend([kind] * n)
    allp = [(f"{c}.{name}", n, kind) for c in COLS for name, n, kind in COLUMN + THAL] + HIPPO + NUCLEI
    order = sorted(range(len(allp)), key=lambda i: -cells.KINDS[allp[i][2]].get("ncomp", 1))  # stable
    for i in order:
        add(*allp[i])
    return pops, kinds


POPS, KINDS = layout()
N = len(KINDS)
IS_I = np.array([k in INHIB for k in KINDS])
IS_E = ~IS_I
NCOMP = np.array([cells.KINDS[k].get("ncomp", 1) for k in KINDS])
B_IDX = np.flatnonzero(NCOMP == 3)            # targets with a basal compartment
A_IDX = np.flatnonzero(NCOMP >= 2)            # targets with an apical tuft
E_IDX, I_IDX = np.flatnonzero(IS_E), np.flatnonzero(IS_I)
FAC_TARGET = np.array([k in ("sst", "vip", "ngf") for k in KINDS])  # E->SST/VIP/NGF facilitate; others depress


def sl(name):
    a, b = POPS[name]
    return np.arange(a, b)


def sli(name):
    a, b = POPS[name]
    return slice(a, b)


@dataclass
class Proj:
    pre: str
    post: str
    comp: str      # s (soma AMPA or GABA_A), b (basal AMPA+NMDA), a (apical AMPA+NMDA or slow GABA_A + GABA_B)
    p: float
    w: float       # nS per spike at full release
    long: bool = False
    length_mm: float = 0.0


def projections():
    P = []
    for c in COLS:
        L = lambda x: f"{c}.{x}"
        P += [Proj(L("TC_core"), L("L4_SS"), "s", .30, 1.6), Proj(L("TC_core"), L("L4_PV"), "s", .50, 2.0),
              Proj(L("TC_core"), L("L23_PC"), "b", .08, .5), Proj(L("TC_core"), L("L5_ET"), "s", .08, .6),
              Proj(L("L4_SS"), L("L23_PC"), "b", .15, .8), Proj(L("L4_SS"), L("L23_PV"), "s", .30, 1.4),
              Proj(L("L4_SS"), L("L4_SS"), "s", .10, .5), Proj(L("L4_SS"), L("L4_PV"), "s", .30, 1.4),
              Proj(L("L23_PC"), L("L23_PV"), "s", .30, 1.2), Proj(L("L23_PC"), L("L23_SST"), "s", .20, .8),
              Proj(L("L23_PC"), L("L23_VIP"), "s", .10, .6), Proj(L("L23_PC"), L("L5_ET"), "s", .10, .8),
              Proj(L("L23_PC"), L("L5_IT"), "s", .10, .8),
              Proj(L("L23_PV"), L("L23_PV"), "s", .40, 2.0), Proj(L("L23_SST"), L("L23_PV"), "s", .20, 1.5),
              Proj(L("L23_VIP"), L("L23_SST"), "s", .50, 2.5), Proj(L("L1_NGF"), L("L23_PC"), "a", .40, 1.0),
              Proj(L("L1_NGF"), L("L5_ET"), "a", .30, 1.0),
              Proj(L("L5_ET"), L("L5_PV"), "s", .30, 1.2), Proj(L("L5_ET"), L("L5_SST"), "s", .20, .8),
              Proj(L("L5_IT"), L("L5_ET"), "s", .10, .8), Proj(L("L5_IT"), L("L5_PV"), "s", .20, 1.0),
              Proj(L("L5_PV"), L("L5_ET"), "s", .50, 3.0), Proj(L("L5_PV"), L("L5_IT"), "s", .50, 3.0),
              Proj(L("L5_SST"), L("L5_ET"), "a", .40, 2.0),
              Proj(L("L6_CT"), L("TC_core"), "s", .20, .6), Proj(L("L6_CT"), L("TRN"), "s", .30, 1.0),
              Proj(L("L6_CT"), L("L6_PV"), "s", .30, 1.0), Proj(L("L6_PV"), L("L6_CT"), "s", .50, 3.0),
              Proj(L("L6_SST"), L("L6_CT"), "s", .40, 2.0), Proj(L("L5_IT"), L("L6_CT"), "s", .10, .6),
              Proj(L("TRN"), L("TC_core"), "s", .40, 3.0), Proj(L("TRN"), L("TC_mtx"), "s", .30, 2.5),
              Proj(L("TRN"), L("TRN"), "s", .20, 1.0), Proj(L("TC_core"), L("TRN"), "s", .30, 1.0),
              Proj(L("L5_ET"), L("TC_mtx"), "s", .25, 2.0, long=True, length_mm=25.0),
              Proj(L("TC_mtx"), L("L1_NGF"), "s", .30, 1.0), Proj(L("TC_mtx"), L("L23_PC"), "a", .15, .8),
              Proj(L("TC_mtx"), L("L5_ET"), "a", .25, 1.0), Proj(L("TC_mtx"), L("L23_VIP"), "s", .20, 1.0)]
    # cortico-cortical (myelinated, long): feedforward to L4 / basal, feedback to the apical tuft
    for a, b in (("S", "T"), ("S", "P"), ("T", "P"), ("I", "P"), ("S", "I")):
        P += [Proj(f"{a}.L23_PC", f"{b}.L4_SS", "s", .05, .6, True, 40.0),
              Proj(f"{a}.L23_PC", f"{b}.L23_PC", "b", .03, .5, True, 40.0)]
        P += [Proj(f"{b}.L5_IT", f"{a}.L23_PC", "a", .05, .6, True, 40.0),
              Proj(f"{b}.L5_IT", f"{a}.L1_NGF", "s", .10, .6, True, 40.0)]
    # hippocampal formation
    P += [Proj("S.L5_IT", "DG", "s", .10, 1.0, True, 60.0), Proj("T.L5_IT", "DG", "s", .10, 1.0, True, 60.0),
          Proj("S.L5_IT", "CA3", "s", .05, .6, True, 60.0), Proj("DG", "CA3", "s", .05, 4.0),
          Proj("CA3", "CA3_PV", "s", .30, 1.2), Proj("CA3_PV", "CA3", "s", .50, 3.0),
          Proj("CA3", "CA1", "a", .15, .8), Proj("CA3", "CA1_PV", "s", .20, 1.0), Proj("CA1_PV", "CA1", "s", .50, 3.0),
          Proj("CA1", "CA1_OLM", "s", .20, 1.0), Proj("CA1_OLM", "CA1", "a", .40, 2.0),
          Proj("CA1", "P.L5_IT", "s", .05, .6, True, 70.0), Proj("CA1", "T.L23_PC", "b", .03, .5, True, 70.0)]
    # neuromodulatory and hypothalamic nuclei
    P += [Proj("P.L5_IT", "VTA_DA", "s", .10, .8, True, 80.0), Proj("VTA_GABA", "VTA_DA", "s", .50, 2.0),
          Proj("P.L5_ET", "LC", "s", .10, .8, True, 90.0), Proj("P.L5_ET", "DRN", "s", .10, .8, True, 90.0),
          Proj("P.L5_IT", "NBM", "s", .10, .8, True, 80.0), Proj("I.L5_ET", "ARC", "s", .10, .8, True, 90.0),
          Proj("ARC", "PVN", "s", .20, 1.0), Proj("ARC", "VTA_DA", "s", .10, .6, True, 30.0)]
    return P


# plastic blocks kept outside the gathered matrices
GB_BLOCKS = [("CA3", "CA3", "s"), ("T.L4_SS", "T.L23_PC", "b"), ("T.L23_PC", "T.L23_PC", "b"),
             ("P.L23_PC", "P.L23_PC", "b")]
VG_BLOCKS = [(f"{c}.L23_PV", f"{c}.L23_PC", "s") for c in COLS] + [(f"{c}.L23_SST", f"{c}.L23_PC", "a") for c in COLS]

# ATP budget per neuron (normalised; Attwell & Laughlin 2001 split: resting ~ 1/3, synaptic + spiking ~ 2/3 at 3 Hz)
ENERGY = dict(k_sup=1.0, k_rest=0.06, k_syn=0.01, k_spike=0.004)   # 1/s, 1/s, 1/(s nS), per spike
HUNGER_PA = 260.0   # AgRP-like ARC cells are glucose-INHIBITED: drive = HUNGER_PA * (1 - glucose)


def glucose_of_lumen(dice: int, max_dice: int = 6) -> float:
    """The Lumen is the Tome's blood sugar: residual station power plus its own light."""
    return 0.15 + 0.85 * dice / max_dice


# Graupner & Brunel 2012, 'DP' fit to Sjostrom 2001 cortical slices
GB = dict(tau=150e3, rho_star=0.5, gamma_p=321.808, gamma_d=200.0, theta_p=1.3, theta_d=1.0, sigma=2.8284,
          C_pre=1.0, C_post=2.0, tau_ca=20.0, D=14, w_min=0.05, w_max=1.0)
VOGELS = dict(eta=2e-2, rho0_hz=3.0, tau=20.0, w_max=6.0)
# De Pitta et al. 2009 (Li-Rinzel + IP3 metabolism, AM-encoding set); time in seconds inside the RHS
AST = dict(r_C=6.0, r_L=0.11, C_T=2.0, c1=0.185, v_ER=0.9, K_ER=0.1, d1=0.13, d2=1.049, d3=0.9434, d5=0.08234,
           a2=0.2, v_beta=0.5, v_delta=0.02, kappa_delta=1.5, K_delta=0.1, v_3K=2.0, K_D=0.7, K_3=1.0, r_5P=0.04,
           C_theta=0.5, tau_G=1.0, U_A=0.6)


def build(seed: int = 0):
    rng = np.random.default_rng(seed)
    projs = projections()
    w_e = np.zeros((len(E_IDX) + 1, N + len(B_IDX) + len(A_IDX)), np.float32)   # [soma | basal | apical]
    w_i = np.zeros((len(I_IDX) + 1, N + len(A_IDX)), np.float32)                 # [soma | apical]
    w_long = np.zeros((len(E_IDX) + 1, N + len(B_IDX) + len(A_IDX)), np.float32)
    delay = np.ones(len(E_IDX), np.int32)
    length = np.zeros(len(E_IDX), np.float32)
    epos = {g: i for i, g in enumerate(E_IDX)}
    ipos = {g: i for i, g in enumerate(I_IDX)}
    bpos = {g: i for i, g in enumerate(B_IDX)}
    apos = {g: i for i, g in enumerate(A_IDX)}
    plastic = {(a, b) for a, b, _ in GB_BLOCKS + VG_BLOCKS}
    for pr in projs:
        if (pr.pre, pr.post) in plastic:
            continue
        pre, post = sl(pr.pre), sl(pr.post)
        mask = rng.random((len(pre), len(post))) < pr.p
        w = (pr.w * rng.lognormal(0.0, 0.5, mask.shape) / np.exp(0.125)) * mask
        exc = IS_E[pre[0]]
        for i, gi in enumerate(pre):
            if exc:
                row, M = epos[gi], (w_long if pr.long else w_e)
                if pr.long:
                    length[row] = pr.length_mm
            else:
                row, M = ipos[gi], w_i
            for j, gj in enumerate(post):
                if w[i, j] == 0:
                    continue
                if pr.comp == "s" or (pr.comp == "b" and gj not in bpos) or (pr.comp == "a" and gj not in apos):
                    col = gj
                elif pr.comp == "b":
                    col = N + bpos[gj]
                else:
                    col = (N + len(B_IDX) + apos[gj]) if exc else (N + apos[gj])
                M[row, col] += w[i, j]
    # myelinated conduction: c0 = 4 mm/ms (moderately myelinated), delay = L/c + 0.5 ms synaptic
    cvel = np.full(len(E_IDX), 4.0, np.float32)
    delay = np.where(length > 0, np.clip(np.round(length / cvel + 0.5), 2, D_RING - 1), 1).astype(np.int32)

    gb = []
    for a, b, comp in GB_BLOCKS:
        npre, npost = len(sl(a)), len(sl(b))
        m = (rng.random((npre, npost)) < 0.12).astype(np.float32)
        gb.append(dict(mask=m, rho=(rng.random((npre, npost)) < 0.3).astype(np.float32) * m))
    vg = []
    for a, b, comp in VG_BLOCKS:
        npre, npost = len(sl(a)), len(sl(b))
        m = (rng.random((npre, npost)) < 0.45).astype(np.float32)
        vg.append(dict(mask=m, w=m * (2.5 if comp == "s" else 1.8)))

    params = cells.build_params(KINDS, g_katp=12.0)
    sensors = np.zeros(N, bool)
    for name in ("ARC", "PVN"):
        sensors[sl(name)] = True        # glucose-sensing hypothalamic cells keep firing as ATP falls
    params["g_katp"] = jnp.where(jnp.asarray(sensors), 0.0, params["g_katp"])
    st = cells.init_state(params)
    n_ast = 40
    dom = np.full(N, -1, np.int32)                 # astrocyte domain per neuron (PCs only)
    k = 0
    for c in COLS:
        idx = sl(f"{c}.L23_PC")
        for j in range(8):
            dom[idx[j * 12:(j + 1) * 12]] = k; k += 1
    for name in ("CA3", "CA1"):
        idx = sl(name)
        for j in range(4):
            dom[idx[j * len(idx) // 4:(j + 1) * len(idx) // 4]] = k; k += 1
    st.update(
        ring=jnp.zeros((D_RING, N), jnp.float32), t=jnp.zeros((), jnp.int32),
        u_dep=jnp.full(N, 0.5, jnp.float32), x_dep=jnp.ones(N, jnp.float32), u_fac=jnp.full(N, 0.1, jnp.float32),
        x_fac=jnp.ones(N, jnp.float32), r_dep=jnp.zeros(N, jnp.float32), r_fac=jnp.zeros(N, jnp.float32),
        ca_pre=jnp.zeros(N), ca_post=jnp.zeros(N), xv=jnp.zeros(N),
        ast_C=jnp.full(n_ast, 0.05, jnp.float32), ast_h=jnp.full(n_ast, 0.8, jnp.float32), ast_I=jnp.full(n_ast, 0.05, jnp.float32),
        ast_G=jnp.zeros(n_ast), ast_glu=jnp.zeros(n_ast),
        mod=jnp.zeros(4, jnp.float32), noise=jnp.zeros(N, jnp.float32), count=jnp.zeros(N, jnp.float32),
        overflow=jnp.zeros((), jnp.int32), glucose=jnp.ones((), jnp.float32), key=jax.random.PRNGKey(seed),
        gb_rho=[jnp.asarray(g["rho"]) for g in gb], vg_w=[jnp.asarray(v["w"]) for v in vg])
    w_e, w_long, w_i = w_e * GAIN["e"], w_long * GAIN["e"], w_i * GAIN["i"]
    static = dict(params=params, w_e=jnp.asarray(w_e), w_i=jnp.asarray(w_i), w_long=jnp.asarray(w_long),
                  delay=jnp.asarray(delay), gb_mask=[jnp.asarray(g["mask"]) for g in gb],
                  vg_mask=[jnp.asarray(v["mask"]) for v in vg], dom=jnp.asarray(dom),
                  fac=jnp.asarray(FAC_TARGET.astype(np.float32)),
                  dep_cols=jnp.asarray(np.concatenate([1 - FAC_TARGET, np.ones(len(B_IDX) + len(A_IDX))]).astype(np.float32)),
                  fac_cols=jnp.asarray(np.concatenate([FAC_TARGET, np.zeros(len(B_IDX) + len(A_IDX))]).astype(np.float32)), cvel=jnp.asarray(cvel), length=jnp.asarray(length))
    return st, static


GB_IDX = [(sli(a), sli(b), comp) for a, b, comp in GB_BLOCKS]
VG_IDX = [(sli(a), sli(b), comp) for a, b, comp in VG_BLOCKS]
VTA, LC, DRN, NBM = sli("VTA_DA"), sli("LC"), sli("DRN"), sli("NBM")
ARC = sli("ARC")
assert (B_IDX == np.arange(len(B_IDX))).all() and (A_IDX == np.arange(len(A_IDX))).all()
NB, NA = len(B_IDX), len(A_IDX)
NOISE_SD = np.where(IS_I, 25.0, 35.0).astype(np.float32)


def _gather(w, spk_rows, k):
    """Sum the weight rows of the presynaptic cells that spiked (at most k; extra spikes are counted)."""
    n = spk_rows.shape[0]
    idx = jnp.nonzero(spk_rows > 0, size=k, fill_value=n)[0]
    over = jnp.maximum(0, jnp.sum(spk_rows > 0) - k)
    return idx, over


K_SMALL = 24


def deliver(w, spk_rows, coef_rows, k_big):
    """sum_i coef_i * w[i] over presynaptic rows that spiked. Two tiers: a cheap K_SMALL gather for ordinary
    activity, the k_big gather only during bursts (lax.cond picks at run time). Spikes beyond k_big are counted."""
    n = spk_rows.shape[0]
    cnt = jnp.sum(spk_rows > 0)
    coef = jnp.concatenate([coef_rows, jnp.zeros(1, coef_rows.dtype)])
    def tier(k):
        def f(_):
            idx = jnp.nonzero(spk_rows > 0, size=k, fill_value=n)[0]
            return coef[idx] @ w[idx]
        return f
    out = jax.lax.cond(cnt <= K_SMALL, tier(K_SMALL), tier(k_big), None)
    return out, jnp.maximum(0, cnt - k_big)


def ms_step(static, st, i_drive):
    """One millisecond: deliver spikes from the ring (local 1 ms, long-range per-axon delay), 4 membrane substeps,
    STP, plasticity, astrocytes, energy and neuromodulators."""
    p = static["params"]
    t = st["t"]
    last = st["ring"][(t - 1) % D_RING]
    # ---- presynaptic release (Tsodyks-Markram), computed for every cell that spiked last ms
    r_dep = st["r_dep"]          # u+ x- released by last ms's spikes (Tsodyks-Markram order)
    r_fac = st["r_fac"]
    e_last, i_last = last[E_IDX], last[I_IDX]
    fac = static["fac"]
    # depressing release onto E/PV targets, facilitating release onto SST/VIP/NGF targets (Markram/Reyes 1998)
    idx, over_e = _gather(static["w_e"], e_last, K_E)
    rows = static["w_e"][idx]
    rd = jnp.concatenate([r_dep[E_IDX], jnp.zeros(1)])[idx]
    rf = jnp.concatenate([r_fac[E_IDX], jnp.zeros(1)])[idx]
    exc = (rd @ rows) * static["dep_cols"] + (rf @ rows) * static["fac_cols"]
    # long-range: each axon has its own (myelin-set) delay
    tlong = (t - static["delay"]) % D_RING
    spk_long = st["ring"][tlong, E_IDX]
    if "long" not in ABL:
        idx2, over_l = _gather(static["w_long"], spk_long, K_E)
        exc = exc + jnp.concatenate([spk_long, jnp.zeros(1)])[idx2] @ static["w_long"][idx2]
    else:
        over_l = 0
    iidx, over_i = _gather(static["w_i"], i_last, K_I)
    inh = jnp.concatenate([i_last, jnp.zeros(1)])[iidx] @ static["w_i"][iidx]
    # plastic blocks (GB excitatory, Vogels inhibitory)
    add_s = jnp.zeros(N); add_b = jnp.zeros(NB); add_a = jnp.zeros(NA); add_gs = jnp.zeros(N); add_ga = jnp.zeros(NA)
    for (pre, post, comp), rho, m in zip(GB_IDX, st["gb_rho"], static["gb_mask"]):
        w = m * (GB["w_min"] + rho * (GB["w_max"] - GB["w_min"]))
        g = r_dep[pre] @ w
        if comp == "s":
            add_s = add_s.at[post].add(g)
        else:
            add_b = add_b.at[post].add(g)          # basal targets are a prefix: same index
    for (pre, post, comp), w in zip(VG_IDX, st["vg_w"]):
        g = last[pre] @ w
        if comp == "s":
            add_gs = add_gs.at[post].add(g)
        else:
            add_ga = add_ga.at[post].add(g)

    mod = st["mod"]                       # [DA, NE, 5-HT, ACh] extracellular, arbitrary units
    ach = mod[3]
    gs = dict(st)
    gs["gA_s"] = st["gA_s"] + exc[:N] + add_s
    nm_gain = 1.0 + 0.5 * jnp.take(jnp.concatenate([st["ast_G"], jnp.zeros(1)]), static["dom"])  # D-serine
    gb_ = exc[N:N + NB] + add_b
    ga_ = (exc[N + NB:] + add_a) * (1.0 - 0.5 * ach)                    # ACh suppresses feedback onto the tuft
    padb, pada = jnp.zeros(N - NB), jnp.zeros(N - NA)
    gs["gA_b"] = st["gA_b"] + jnp.concatenate([gb_, padb])
    gs["gN_b"] = st["gN_b"] + jnp.concatenate([0.5 * gb_ * nm_gain[:NB], padb])
    gs["gA_a"] = st["gA_a"] + jnp.concatenate([ga_, pada])
    gs["gN_a"] = st["gN_a"] + jnp.concatenate([0.6 * ga_ * nm_gain[:NA], pada])
    gs["gGA_s"] = st["gGA_s"] + inh[:N] + add_gs
    gs["gGA_a"] = st["gGA_a"] + jnp.concatenate([inh[N:] + add_ga, pada])
    # ---- noise (OU, tau 5 ms) + game drive
    key, k1 = jax.random.split(st["key"])
    eps = jax.random.normal(k1, (N,)) if "rng" not in ABL else jnp.sin(st["t"] * 1.7 + jnp.arange(N))
    noise = st["noise"] * np.exp(-1 / 5.0) + jnp.asarray(NOISE_SD) * np.sqrt(1 - np.exp(-2 / 5.0)) * eps
    ne = mod[1]
    p2 = dict(p, VT=p["VT"] - 2.0 * ne)  # NE raises gain
    hunger = jnp.zeros(N).at[ARC].set(HUNGER_PA * (1.0 - st["glucose"]))
    i_ext = jnp.zeros((3, N)).at[0].set(noise + i_drive + hunger)
    spk_ms = jnp.zeros(N)
    def body(_, carry):
        s_, acc = carry
        s_, spk = substep(p2, s_, i_ext, DT)
        return s_, acc + spk.astype(jnp.float32)
    gs, spk_ms = jax.lax.fori_loop(0, SUB, body, (gs, spk_ms))
    spk = jnp.minimum(spk_ms, 1.0)
    # ---- STP state update (exact relaxation over 1 ms, then the spike jump)
    def relax(u, x, U, tf, td):
        u = U + (u - U) * np.exp(-1.0 / tf)
        x = 1.0 - (1.0 - x) * np.exp(-1.0 / td)
        u_plus = u + U * (1 - u) * spk          # facilitation jump first
        rel = u_plus * x * spk                  # release uses the pre-depletion pool
        return u_plus, x - rel, rel
    u_dep, x_dep, rel_dep = relax(st["u_dep"], st["x_dep"], 0.5, 20.0, 500.0)
    u_fac, x_fac, rel_fac = relax(st["u_fac"], st["x_fac"], 0.1, 1000.0, 100.0)
    # ---- Graupner-Brunel (per synapse calcium = delayed pre trace + post trace)
    dpre = st["ring"][(t - GB["D"]) % D_RING]
    ca_pre = st["ca_pre"] * np.exp(-1 / GB["tau_ca"]) + GB["C_pre"] * dpre
    ca_post = st["ca_post"] * np.exp(-1 / GB["tau_ca"]) + GB["C_post"] * spk
    key, k2 = jax.random.split(key)
    new_rho = []
    for (pre, post, comp), rho, m in zip(GB_IDX, st["gb_rho"], static["gb_mask"]):
        c = ca_pre[pre][:, None] + ca_post[post][None, :]
        up, dn = (c >= GB["theta_p"]).astype(jnp.float32), (c >= GB["theta_d"]).astype(jnp.float32)
        drift = -rho * (1 - rho) * (GB["rho_star"] - rho) + GB["gamma_p"] * (1 - rho) * up - GB["gamma_d"] * rho * dn
        rho = jnp.clip(rho + drift / GB["tau"], 0, 1) * m   # GB noise term omitted in the prototype
        new_rho.append(rho)
    # ---- Vogels iSTDP: pre spike -> eta (x_post - alpha); post spike -> eta x_pre
    xv = st["xv"] * np.exp(-1 / VOGELS["tau"]) + spk
    alpha = 2 * VOGELS["rho0_hz"] * 1e-3 * VOGELS["tau"]
    new_vg = []
    for (pre, post, comp), w, m in zip(VG_IDX, st["vg_w"], static["vg_mask"]):
        dw = VOGELS["eta"] * (spk[pre][:, None] * (xv[post][None, :] - alpha) + xv[pre][:, None] * spk[post][None, :])
        new_vg.append(jnp.clip(w + dw * m, 0.0, VOGELS["w_max"]))
    # ---- astrocytes (Li-Rinzel + IP3, De Pitta 2009), dt = 1 ms = 1e-3 s
    a = AST
    dom = static["dom"]
    glu = jax.ops.segment_sum(spk, jnp.where(dom >= 0, dom, 40), num_segments=41, indices_are_sorted=False)[:40] / 12.0
    ast_glu = st["ast_glu"] * np.exp(-1 / 50.0) + glu
    C, h, I = st["ast_C"], st["ast_h"], st["ast_I"]
    m_inf = I / (I + a["d1"]); n_inf = C / (C + a["d5"])
    J_chan = a["r_C"] * m_inf ** 3 * n_inf ** 3 * h ** 3 * (a["C_T"] - (1 + a["c1"]) * C)
    J_leak = a["r_L"] * (a["C_T"] - (1 + a["c1"]) * C)
    J_pump = a["v_ER"] * C ** 2 / (C ** 2 + a["K_ER"] ** 2)
    Q2 = a["d2"] * (I + a["d1"]) / (I + a["d3"])
    h_inf = Q2 / (Q2 + C); tau_h = 1.0 / (a["a2"] * (Q2 + C))
    J_beta = a["v_beta"] * jnp.tanh(ast_glu)
    J_delta = a["v_delta"] / (1 + I / a["kappa_delta"]) * C ** 2 / (C ** 2 + a["K_delta"] ** 2)
    J_3K = a["v_3K"] * C ** 4 / (C ** 4 + a["K_D"] ** 4) * I / (I + a["K_3"])
    J_5P = a["r_5P"] * I
    dts = 1e-3
    C = jnp.clip(C + dts * (J_chan + J_leak - J_pump), 0, 5)
    h = h_inf + (h - h_inf) * jnp.exp(-dts / tau_h)
    I = jnp.clip(I + dts * (J_beta + J_delta - J_3K - J_5P), 0, 10)
    G = st["ast_G"] * np.exp(-dts / a["tau_G"]) + a["U_A"] * (1 - st["ast_G"]) * ((C >= a["C_theta"]) & (st["ast_C"] < a["C_theta"]))
    # ---- energy: ATP per neuron (normalised), supply = glucose (the Lumen) x local blood flow (astrocyte Ca)
    cbf = 1.0 + 0.5 * jnp.take(jnp.concatenate([C / (C + 0.5), jnp.zeros(1)]), dom)
    g_exc = gs["gA_s"] + gs["gA_b"] + gs["gA_a"]
    atp = gs["atp"] + 1e-3 * (ENERGY["k_sup"] * st["glucose"] * cbf * (1.05 - gs["atp"]) - ENERGY["k_rest"]
                              - ENERGY["k_syn"] * g_exc) - ENERGY["k_spike"] * spk
    gs["atp"] = jnp.clip(atp, 0.05, 1.0)
    # ---- neuromodulators: volume transmission from the nuclei, first-order clearance
    rel = jnp.stack([spk[VTA].sum(), spk[LC].sum(), spk[DRN].sum(), spk[NBM].sum()])
    mod = mod * jnp.exp(-1.0 / jnp.array([200.0, 500.0, 1000.0, 300.0])) + 0.05 * rel

    ring = st["ring"].at[t % D_RING].set(spk)
    gs.update(ring=ring, t=t + 1, u_dep=u_dep, x_dep=x_dep, u_fac=u_fac, x_fac=x_fac, r_dep=rel_dep, r_fac=rel_fac, ca_pre=ca_pre,
              ca_post=ca_post, xv=xv, gb_rho=new_rho, vg_w=new_vg, ast_C=C, ast_h=h, ast_I=I, ast_G=G,
              ast_glu=ast_glu, mod=mod, noise=noise, key=key, count=st["count"] + spk,
              overflow=(st["overflow"] + over_e + over_l + over_i).astype(jnp.int32))
    return gs, spk


def make_quiet():
    """Between turns: Delta seconds of quiet wakefulness. Fast state is carried over unchanged (working memory
    persists); slow variables integrate with spike rates frozen at the last window's mean (adiabatic elimination):
    ATP (exact exponential solution), astrocyte Li-Rinzel (20-ms steps), GB bistable consolidation (no Ca2+),
    tonic neuromodulator clearance."""
    @jax.jit
    def quiet(static, st, seconds, rate_hz):
        a = AST
        # ATP: linear ODE in atp with frozen demand -> exact solution
        C = st["ast_C"]
        cbf = 1.0 + 0.5 * jnp.take(jnp.concatenate([C / (C + 0.5), jnp.zeros(1)]), static["dom"])
        k = ENERGY["k_sup"] * st["glucose"] * cbf
        demand = ENERGY["k_rest"] + ENERGY["k_syn"] * 1.0 + ENERGY["k_spike"] * rate_hz
        a_inf = 1.05 - demand / k
        atp = jnp.clip(a_inf + (st["atp"] - a_inf) * jnp.exp(-k * seconds), 0.05, 1.0)
        glu = jax.ops.segment_sum(rate_hz * 0.05, jnp.where(static["dom"] >= 0, static["dom"], 40), num_segments=41)[:40] / 12.0
        dts = 0.02
        def astro(c, _):
            C, h, I, G = c
            m_inf = I / (I + a["d1"]); n_inf = C / (C + a["d5"])
            J = (a["r_C"] * m_inf ** 3 * n_inf ** 3 * h ** 3 * (a["C_T"] - (1 + a["c1"]) * C)
                 + a["r_L"] * (a["C_T"] - (1 + a["c1"]) * C) - a["v_ER"] * C ** 2 / (C ** 2 + a["K_ER"] ** 2))
            Q2 = a["d2"] * (I + a["d1"]) / (I + a["d3"])
            h = Q2 / (Q2 + C) + (h - Q2 / (Q2 + C)) * jnp.exp(-dts * a["a2"] * (Q2 + C))
            dI = (a["v_beta"] * jnp.tanh(glu) + a["v_delta"] / (1 + I / a["kappa_delta"]) * C ** 2 / (C ** 2 + a["K_delta"] ** 2)
                  - a["v_3K"] * C ** 4 / (C ** 4 + a["K_D"] ** 4) * I / (I + a["K_3"]) - a["r_5P"] * I)
            Cn = jnp.clip(C + dts * J, 0, 5)
            G = G * jnp.exp(-dts / a["tau_G"]) + a["U_A"] * (1 - G) * ((Cn >= a["C_theta"]) & (C < a["C_theta"]))
            return (Cn, h, jnp.clip(I + dts * dI, 0, 10), G), None
        n_ast = jnp.int32(seconds / dts)
        (C, h, I, G), _ = jax.lax.scan(astro, (st["ast_C"], st["ast_h"], st["ast_I"], st["ast_G"]), None, length=1000)
        rho = []
        for r in st["gb_rho"]:
            def cons(r, _):
                return jnp.clip(r + 1.0 * (-r * (1 - r) * (GB["rho_star"] - r)) / (GB["tau"] / 1000.0), 0, 1), None
            r, _ = jax.lax.scan(cons, r, None, length=20)       # 20 x 1 s
            rho.append(r)
        mod = st["mod"] * jnp.exp(-seconds * 1000.0 / jnp.array([200.0, 500.0, 1000.0, 300.0]))
        return dict(st, atp=atp, ast_C=C, ast_h=h, ast_I=I, ast_G=G, gb_rho=rho, mod=mod)
    return quiet


def make_window(static_keys=None):
    @jax.jit
    def window(static, st, i_drive):
        def body(st, _):
            st, spk = ms_step(static, st, i_drive)
            return st, spk
        st, spikes = jax.lax.scan(body, st, None, length=W_SYNC)
        return st, spikes
    return window


if __name__ == "__main__":
    t0 = time.perf_counter()
    st, static = build(0)
    t1 = time.perf_counter()
    win = make_window()
    drive = jnp.zeros(N)
    st, sp = win(static, st, drive); jax.block_until_ready(sp)
    t2 = time.perf_counter()
    print(f"N={N}  build {t1-t0:.2f}s  compile+first window {t2-t1:.2f}s")
    print("w_e", static["w_e"].shape, "nnz", int((static["w_e"] != 0).sum()), "w_i", static["w_i"].shape,
          "nnz", int((static["w_i"] != 0).sum()), "w_long nnz", int((static["w_long"] != 0).sum()))
    for rep in range(3):
        t3 = time.perf_counter()
        for _ in range(250 // W_SYNC):
            st, sp = win(static, st, drive)
        jax.block_until_ready(sp)
        t4 = time.perf_counter()
        cnt = np.asarray(st["count"])
        print(f"250 ms bio-time: {1e3*(t4-t3):.1f} ms wall | mean rate {cnt.sum()/N/((rep+1)*0.25+0.004):.2f} Hz | "
              f"overflow {int(st['overflow'])} | nan {bool(np.isnan(np.asarray(st['vs'])).any())}")
