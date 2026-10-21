"""Prototype of the Tome's JAX partition (cortex x4 columns, thalamus, hippocampus, glia, energy).

Runtime prototype: realistic sizes and op counts, stable integrators (exponential Euler), event-driven synapses.
One call = one min-delay window of D bus ticks (1 ms each), each split into `sub` membrane substeps.

Populations (index into one global neuron vector; MC = 3-compartment human L2/3-style cells, rest point AdEx):
  per column c in SENSE, TOM, SELF, PLAN:
    L23 PC (MC) 128, L5 ET (MC) 32, L4 IT 64, L5 IT 32, L6 CT 48, PV 24, SST 16, VIP 12, LAMP5 8
  thalamus: TC core 128, matrix 64, TRN 48
  hippocampus: CA3 thorny 192, CA3 a-thorny 64, CA1 128
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import jaxpath  # noqa: F401
import jax
import jax.numpy as jnp
from jax import lax

COLS = ("SENSE", "TOM", "SELF", "PLAN")
COL_POPS = (("L23", 128, "mc", "E"), ("L5ET", 32, "mc", "E"), ("L4", 64, "pt", "E"), ("L5IT", 32, "pt", "E"),
            ("L6CT", 48, "pt", "E"), ("PV", 24, "pt", "I"), ("SST", 16, "pt", "I"), ("VIP", 12, "pt", "I"),
            ("LAMP5", 8, "pt", "I"))
SUB_POPS = (("TC", 128, "pt", "E"), ("MTX", 64, "pt", "E"), ("TRN", 48, "pt", "I"),
            ("CA3T", 192, "pt", "E"), ("CA3A", 64, "pt", "E"), ("CA1", 128, "pt", "E"))
# receptor index: 0 AMPA, 1 NMDA, 2 GABA_A, 3 GABA_B ; compartment: 0 soma, 1 basal, 2 apical
TAU_REC = np.array([2.0, 80.0, 8.0, 150.0])
E_REC = np.array([0.0, 0.0, -75.0, -95.0])
HIST = 16          # spike-history ring (ms); covers GB presynaptic calcium delay D = 13.7 ms
QDEPTH = 8         # future-conductance ring (ms); max within-partition delay 7 ms


# Graupner & Brunel 2012 (PNAS), "DP" cortical parameter set; theta_p = 1.3 (not the spec's 1.6)
GB = dict(Cpre=1.0, Cpost=2.0, tauCa=20.0, D=13.7, thd=1.0, thp=1.3, gd=200.0, gp=321.808, tau=150e3, rs=0.5)


def gb_drho(rho, c, gain=1.0, P=GB):
    """d rho / d t (per ms) for calcium c; `gain` is the dopamine gate on consolidation rate."""
    return gain * (-rho * (1 - rho) * (P["rs"] - rho) + P["gp"] * (1 - rho) * (c >= P["thp"])
                   - P["gd"] * rho * (c >= P["thd"])) / P["tau"]


@dataclass
class Layout:
    names: list
    start: dict
    size: dict
    kind: dict      # mc | pt
    sign: dict      # E | I
    n: int
    n_mc: int       # MC cells occupy global indices [0, n_mc)


def layout() -> Layout:
    pops = []
    # MC populations first so MC cells are a contiguous prefix
    for c in COLS:
        for nm, n, k, s in COL_POPS:
            if k == "mc":
                pops.append((f"{c}.{nm}", n, k, s))
    for c in COLS:
        for nm, n, k, s in COL_POPS:
            if k == "pt":
                pops.append((f"{c}.{nm}", n, k, s))
    pops += [(nm, n, k, s) for nm, n, k, s in SUB_POPS]
    start, size, kind, sign, i = {}, {}, {}, {}, 0
    for nm, n, k, s in pops:
        start[nm], size[nm], kind[nm], sign[nm] = i, n, k, s
        i += n
    n_mc = sum(size[p] for p in size if kind[p] == "mc")
    return Layout([p[0] for p in pops], start, size, kind, sign, i, n_mc)


def projections(L: Layout):
    """(src, dst, in-degree K, receptor list, compartment, weight nS, delay range ms, stp 0 dep/1 fac, plastic)."""
    P = []
    def add(src, dst, K, recs, comp, w, d=(1, 2), stp=0, plastic=False):
        P.append((src, dst, K, recs, comp, w, d, stp, plastic))
    for c in COLS:
        C = lambda p: f"{c}.{p}"
        add("TC", C("L4"), 12, (0, 1), 0, 1.2, (2, 4))
        add("TC", C("L23"), 6, (0, 1), 1, 0.9, (2, 4))
        add(C("L4"), C("L23"), 16, (0, 1), 1, 1.0)
        add(C("L23"), C("L23"), 24, (0, 1), 1, 0.7, (1, 3), plastic=c in ("TOM", "PLAN"))
        add(C("L23"), C("L5ET"), 16, (0, 1), 1, 1.0)
        add(C("L23"), C("L5IT"), 16, (0, 1), 0, 1.0)
        add(C("L5IT"), C("L5ET"), 8, (0, 1), 1, 0.8)
        add(C("L5ET"), C("L6CT"), 8, (0, 1), 0, 0.8)
        add(C("L23"), C("PV"), 16, (0,), 0, 1.6, stp=0)
        add(C("L23"), C("SST"), 16, (0,), 0, 0.9, stp=1)
        add(C("L4"), C("PV"), 8, (0,), 0, 1.6)
        add(C("PV"), C("L23"), 10, (2,), 0, 3.0)
        add(C("PV"), C("L5ET"), 8, (2,), 0, 3.0)
        add(C("PV"), C("L4"), 8, (2,), 0, 3.0)
        add(C("PV"), C("PV"), 6, (2,), 0, 2.0)
        add(C("SST"), C("L23"), 8, (2,), 2, 2.5)       # apical dendrite-targeting
        add(C("SST"), C("L5ET"), 6, (2,), 2, 2.5)
        add(C("VIP"), C("SST"), 6, (2,), 0, 3.0)       # disinhibition
        add(C("LAMP5"), C("L23"), 4, (2, 3), 2, 1.5)   # L1 NGF: slow apical inhibition
        add("MTX", C("L23"), 8, (0, 1), 2, 1.2, (3, 5))  # matrix thalamus -> L1 -> apical tuft
        add("MTX", C("LAMP5"), 6, (0,), 0, 1.5, (3, 5))
        add("MTX", C("VIP"), 4, (0,), 0, 1.2, (3, 5))
        add(C("L6CT"), "TC", 4, (0,), 0, 0.6, (3, 6))
        add(C("L6CT"), "TRN", 4, (0,), 0, 0.8, (3, 6))
        add(C("L5ET"), "MTX", 4, (0, 1), 0, 1.0, (3, 6))
        add(C("L23"), "CA3T", 4, (0, 1), 0, 0.8, (4, 7))
    # inter-column feedback onto apical tufts (the consensus pathway)
    for a in COLS:
        for b in COLS:
            if a != b:
                add(f"{a}.L23", f"{b}.L23", 6, (0, 1), 2, 0.6, (3, 6))
    add("TRN", "TC", 8, (2, 3), 0, 2.0)
    add("TRN", "TRN", 4, (2,), 0, 1.5)
    add("TC", "TRN", 6, (0,), 0, 1.0)
    add("CA3T", "CA3T", 24, (0, 1), 0, 0.6, (1, 3), plastic=True)
    add("CA3A", "CA3T", 8, (0, 1), 0, 0.8)
    add("CA3T", "CA3A", 8, (0,), 0, 0.8)
    add("CA3T", "CA1", 20, (0, 1), 0, 0.7, (2, 4), plastic=True)
    add("CA1", "PLAN.L4", 8, (0, 1), 0, 0.8, (4, 7))
    add("CA1", "TOM.L23", 6, (0, 1), 2, 0.6, (4, 7))
    return P


def build(seed: int = 0, dt_sub: float = 0.25):
    L = layout()
    rng = np.random.default_rng(seed)
    N, n_mc = L.n, L.n_mc
    # slot = conductance slot (cell, comp, receptor). MC cells: 3 comps x 4 rec; point cells: 1 comp x 4 rec
    def slot(cell, comp, rec):
        if cell < n_mc:
            return (cell * 3 + comp) * 4 + rec
        return n_mc * 12 + (cell - n_mc) * 4 + rec
    n_slot = n_mc * 12 + (N - n_mc) * 4
    edges = []  # (pre, slot, w, delay, stp, plastic_id)
    plastic_post, plastic_pre = [], []
    for (src, dst, K, recs, comp, w, d, stp, plastic) in projections(L):
        s0, ns, t0, nt = L.start[src], L.size[src], L.start[dst], L.size[dst]
        for j in range(nt):
            post = t0 + j
            pres = rng.choice(ns, size=min(K, ns), replace=False) + s0
            for pre in pres:
                if pre == post:
                    continue
                delay = int(rng.integers(d[0], d[1] + 1))
                c = comp if post < n_mc else 0
                pid = -1
                if plastic:
                    pid = len(plastic_post)
                    plastic_post.append(post)
                    plastic_pre.append(pre)
                for r in recs:
                    ww = w * (0.35 if r == 1 else 1.0) * (0.3 if r == 3 else 1.0) * rng.uniform(0.7, 1.3)
                    edges.append((pre, slot(post, c, r), ww, delay, stp, pid))
    # out-edge table, padded
    by_pre = [[] for _ in range(N)]
    for e in edges:
        by_pre[e[0]].append(e)
    kmax = max(len(b) for b in by_pre)
    T_slot = np.full((N, kmax), n_slot, np.int32)        # sink slot = n_slot
    T_w = np.zeros((N, kmax), np.float32)
    T_d = np.ones((N, kmax), np.int32)
    T_stp = np.zeros((N, kmax), np.int32)
    T_pid = np.full((N, kmax), -1, np.int32)
    for i, b in enumerate(by_pre):
        for k, (pre, sl, w, d, stp, pid) in enumerate(b):
            T_slot[i, k], T_w[i, k], T_d[i, k], T_stp[i, k], T_pid[i, k] = sl, w, d, stp, pid
    # cell parameters (point AdEx), per neuron
    C = np.full(N, 150.0); gL = np.full(N, 8.0); EL = np.full(N, -72.0); VT = np.full(N, -50.0)
    DT = np.full(N, 2.0); a = np.full(N, 2.0); b = np.full(N, 40.0); tw = np.full(N, 150.0)
    Vr = np.full(N, -58.0); ref = np.full(N, 2.0); inh = np.zeros(N, bool)
    for p in L.names:
        sl = slice(L.start[p], L.start[p] + L.size[p])
        kind = p.split(".")[-1]
        if L.sign[p] == "I":
            inh[sl] = True
        if kind == "PV":
            C[sl], gL[sl], a[sl], b[sl], tw[sl], DT[sl], Vr[sl], ref[sl] = 60, 6.0, 0.0, 0.0, 50, 0.8, -60, 1.0
        elif kind == "SST":
            C[sl], gL[sl], a[sl], b[sl], tw[sl] = 80, 5.0, 4.0, 30.0, 300
        elif kind in ("VIP", "LAMP5"):
            C[sl], gL[sl], a[sl], b[sl], tw[sl] = 50, 3.0, 1.0, 20.0, 100
        elif kind in ("TC", "MTX"):
            a[sl], b[sl], tw[sl] = 10.0, 15.0, 100     # subthreshold adaptation -> rebound bursts
        elif kind == "TRN":
            C[sl], gL[sl], a[sl], b[sl], tw[sl] = 80, 5.0, 8.0, 20.0, 100
        elif kind == "CA3A":
            a[sl], b[sl], tw[sl], Vr[sl] = 4.0, 80.0, 60, -48   # a-thorny bursting
    # MC cell compartments: human L2/3, Cm 0.45 uF/cm2, Rm 38907 ohm cm2 (tau 17.5 ms); total 360 pF
    mc = dict(C=np.array([72.0, 162.0, 126.0]), gL=np.array([4.1, 9.3, 7.2]), g_sb=30.0, g_sa=12.0,
              gd=20.0, ECa=120.0)
    pid_n = len(plastic_post)
    P = dict(T_slot=T_slot, T_w=T_w, T_d=T_d, T_stp=T_stp, T_pid=T_pid,
             C=C.astype(np.float32), gL=gL.astype(np.float32), EL=EL.astype(np.float32),
             VT=VT.astype(np.float32), DT=DT.astype(np.float32), a=a.astype(np.float32),
             b=b.astype(np.float32), tw=tw.astype(np.float32), Vr=Vr.astype(np.float32),
             ref=ref.astype(np.float32), inh=inh,
             pl_pre=np.array(plastic_pre, np.int32), pl_post=np.array(plastic_post, np.int32))
    col_of = np.full(N, 4, np.int32)  # 0..3 column, 4 = subcortical
    for ci, c in enumerate(COLS):
        for nm, *_ in COL_POPS:
            p = f"{c}.{nm}"
            col_of[L.start[p]:L.start[p] + L.size[p]] = ci
    P["col_of"] = col_of
    # astrocyte domains: 16 per column over L23 PCs (8 PCs each)
    astro_of = np.full(N, -1, np.int32)
    for ci, c in enumerate(COLS):
        s0 = L.start[f"{c}.L23"]
        for k in range(128):
            astro_of[s0 + k] = ci * 16 + k // 8
    P["astro_of"] = astro_of
    BASE = {"L23": 330, "L5ET": 340, "L4": 175, "L5IT": 175, "L6CT": 170, "PV": 120, "SST": 90, "VIP": 70,
            "LAMP5": 60, "TC": 420, "MTX": 400, "TRN": 140, "CA3T": 170, "CA3A": 170, "CA1": 170}
    base = np.zeros(N, np.float32)
    for p in L.names:
        base[L.start[p]:L.start[p] + L.size[p]] = BASE[p.split(".")[-1]]
    P["bg_mu"] = base
    P["bg_sigma"] = np.full(N, 150.0, np.float32)
    meta = dict(L=L, n_slot=n_slot, kmax=kmax, n_edges=len(edges), n_plastic=pid_n, mc=mc, dt=dt_sub)
    return P, meta


def init_state(P, meta, key):
    N, L = meta["L"].n, meta["L"]
    n_mc = L.n_mc
    f = jnp.float32
    return dict(
        V=jnp.asarray(P["EL"], f) + jax.random.uniform(key, (N,), f, 0, 4),
        Vd=jnp.full((n_mc, 2), -72.0, f),            # basal, apical
        w=jnp.zeros(N, f), refr=jnp.zeros(N, f),
        dm=jnp.zeros(n_mc, f), dh=jnp.ones(n_mc, f), dj=jnp.ones(n_mc, f), du=jnp.zeros(n_mc, f),
        G=jnp.zeros(meta["n_slot"] + 1, f),
        Q=jnp.zeros((QDEPTH, meta["n_slot"] + 1), f),
        hist=jnp.zeros((HIST, N), f),
        x=jnp.ones((2, N), f), u=jnp.stack([jnp.full(N, 0.5, f), jnp.full(N, 0.1, f)]),
        cpre=jnp.zeros(N, f), cpost=jnp.zeros(N, f),
        rho=jnp.full(meta["n_plastic"], 0.5, f) + 0.02 * jax.random.normal(key, (meta["n_plastic"],), f),
        astro=jnp.stack([jnp.full(64, 0.1, f), jnp.full(64, 0.07, f), jnp.full(64, 0.8, f),
                         jnp.zeros(64, f)]),       # IP3, Ca, h, G
        atp=jnp.ones(5, f), cbf=jnp.ones(5, f), kext=jnp.zeros(5, f),
        t=jnp.zeros((), jnp.int32), key=key,
    )


def make_window(P, meta, D: int, sub: int, s_max: int = 192, flags=None, dca=None):
    FL = dict(gb=True, astro=True, stp=True, energy=True, events=True, membrane=True, bg_inside=False)
    FL.update(flags or {})
    DCA = dict(th=-36.0, u_half=21.0, u_k=1.5, gd=None, g_sa=None, tau_j=20.0, tau_u=0.5, tau_h=0.5, tau_h_rec=25.0)
    DCA.update(dca or {})
    """Return jitted run_window(state, ext[D, N] (pA drive), lumen, da, ne) -> (state, spikes[D, N])."""
    L, n_mc, N = meta["L"], meta["L"].n_mc, meta["L"].n
    dt = meta["dt"]
    mcp = dict(meta["mc"])
    if DCA['g_sa'] is not None:
        mcp['g_sa'] = DCA['g_sa']
    f = jnp.float32
    T_slot, T_w, T_d = jnp.asarray(P["T_slot"]), jnp.asarray(P["T_w"]), jnp.asarray(P["T_d"])
    T_stp, T_pid = jnp.asarray(P["T_stp"]), jnp.asarray(P["T_pid"])
    C, gL, EL, VT, DT_ = (jnp.asarray(P[k]) for k in ("C", "gL", "EL", "VT", "DT"))
    a, b, tw, Vr, ref = (jnp.asarray(P[k]) for k in ("a", "b", "tw", "Vr", "ref"))
    col_of = jnp.asarray(P["col_of"]); astro_of = jnp.asarray(P["astro_of"])
    pl_pre, pl_post = jnp.asarray(P["pl_pre"]), jnp.asarray(P["pl_post"])
    decay_sub = np.exp(-dt / TAU_REC)
    decay_vec = jnp.asarray(np.concatenate([np.tile(decay_sub, meta["n_slot"] // 4), [0.0]]), f)
    E_rec = jnp.asarray(E_REC, f)
    n_slot = meta["n_slot"]
    kca = math.exp(-1.0 / GB["tauCa"])
    w_lo, w_hi = 0.2, 1.4   # efficacy of DOWN / UP plastic synapses (scales base weight)

    def membrane_substep(st, Gs, I_ext, katp):
        V, Vd = st["V"], st["Vd"]
        # receptor conductances per slot -> per (cell, comp) currents
        Gm = Gs[: n_mc * 12].reshape(n_mc, 3, 4)
        Gp = Gs[n_mc * 12: n_slot].reshape(N - n_mc, 4)
        def mg(v):  # Jahr-Stevens Mg2+ block, [Mg] = 1 mM
            return 1.0 / (1.0 + (1.0 / 3.57) * jnp.exp(-0.062 * v))
        # ---- MC cells: exponential Euler per compartment, neighbours frozen within the substep
        Vs, Vb, Va = V[:n_mc], Vd[:, 0], Vd[:, 1]
        def g_e(Gc, v):   # total conductance and conductance-weighted reversal for a compartment
            gN = Gc[:, 1] * mg(v)
            g = Gc[:, 0] + gN + Gc[:, 2] + Gc[:, 3]
            ge = Gc[:, 0] * E_rec[0] + gN * E_rec[1] + Gc[:, 2] * E_rec[2] + Gc[:, 3] * E_rec[3]
            return g, ge
        gs_, ges = g_e(Gm[:, 0], Vs); gb_, geb = g_e(Gm[:, 1], Vb); ga_, gea = g_e(Gm[:, 2], Va)
        # dCaAP (Gidon 2020) apical window: activation m(V_a), drive-dependent inactivation h(u), slow termination j(V_a)
        dm = st["dm"] + (1 - jnp.exp(-dt / 2.0)) * (jax.nn.sigmoid((Va - DCA['th']) / 2.5) - st["dm"])
        du = st["du"] + (1 - jnp.exp(-dt / DCA['tau_u'])) * (Gm[:, 2, 0] + Gm[:, 2, 1] - st["du"])
        h_inf = jax.nn.sigmoid(-(du - DCA['u_half']) / DCA['u_k'])
        tau_h = jnp.where(h_inf < st["dh"], DCA['tau_h'], DCA['tau_h_rec'])   # fast inactivation, slow recovery
        dh = st["dh"] + (1 - jnp.exp(-dt / tau_h)) * (h_inf - st["dh"])
        dj = st["dj"] + (1 - jnp.exp(-dt / DCA['tau_j'])) * (jax.nn.sigmoid(-(Va + 5.0) / 4.0) - st["dj"])
        g_dca = (mcp["gd"] if DCA['gd'] is None else DCA['gd']) * dm * dh * dj
        g_k = katp[col_of[:n_mc]]
        # soma: AdEx exponential term as a current, clamped
        expo = gL[:n_mc] * DT_[:n_mc] * jnp.exp(jnp.minimum((Vs - VT[:n_mc]) / DT_[:n_mc], 8.0))
        gtot_s = mcp["gL"][0] + gs_ + mcp["g_sb"] + mcp["g_sa"] + g_k
        einf_s = (mcp["gL"][0] * EL[:n_mc] + ges + mcp["g_sb"] * Vb + mcp["g_sa"] * Va - 90.0 * g_k
                  + expo - st["w"][:n_mc] + I_ext[:n_mc]) / gtot_s
        Vs_n = einf_s + (Vs - einf_s) * jnp.exp(-dt * gtot_s / mcp["C"][0])
        gtot_b = mcp["gL"][1] + gb_ + mcp["g_sb"] + g_k
        einf_b = (mcp["gL"][1] * EL[:n_mc] + geb + mcp["g_sb"] * Vs - 90.0 * g_k) / gtot_b
        Vb_n = einf_b + (Vb - einf_b) * jnp.exp(-dt * gtot_b / mcp["C"][1])
        gtot_a = mcp["gL"][2] + ga_ + mcp["g_sa"] + g_dca + g_k
        einf_a = (mcp["gL"][2] * EL[:n_mc] + gea + mcp["g_sa"] * Vs + g_dca * mcp["ECa"] - 90.0 * g_k) / gtot_a
        Va_n = einf_a + (Va - einf_a) * jnp.exp(-dt * gtot_a / mcp["C"][2])
        # ---- point cells
        Vp = V[n_mc:]
        gNp = Gp[:, 1] * mg(Vp)
        gk_p = katp[col_of[n_mc:]]
        g_syn = Gp[:, 0] + gNp + Gp[:, 2] + Gp[:, 3]
        ge_syn = Gp[:, 0] * E_rec[0] + gNp * E_rec[1] + Gp[:, 2] * E_rec[2] + Gp[:, 3] * E_rec[3]
        expo_p = gL[n_mc:] * DT_[n_mc:] * jnp.exp(jnp.minimum((Vp - VT[n_mc:]) / DT_[n_mc:], 8.0))
        gtot = gL[n_mc:] + g_syn + gk_p
        einf = (gL[n_mc:] * EL[n_mc:] + ge_syn - 90.0 * gk_p + expo_p - st["w"][n_mc:] + I_ext[n_mc:]) / gtot
        Vp_n = einf + (Vp - einf) * jnp.exp(-dt * gtot / C[n_mc:])
        Vn = jnp.concatenate([Vs_n, Vp_n])
        # refractory clamp, spike, reset, adaptation (exact exponential for w)
        inref = st["refr"] > 0
        Vn = jnp.where(inref, Vr, Vn)
        spk = (Vn >= 0.0) & ~inref
        Vn = jnp.where(spk, Vr, Vn)
        winf = a * (Vn - EL)
        w = winf + (st["w"] - winf) * jnp.exp(-dt / tw) + b * spk
        refr = jnp.where(spk, ref, jnp.maximum(st["refr"] - dt, 0.0))
        st = dict(st, V=Vn, Vd=jnp.stack([Vb_n, Va_n], 1), w=w, refr=refr, dm=dm, dh=dh, dj=dj, du=du)
        return st, spk

    def ms_step(st, inp):
        I_ext, lumen, da, ne = inp
        t = st["t"]
        # deliver this ms's queued conductance increments
        q = t % QDEPTH
        G = st["G"] + st["Q"][q]
        Q = st["Q"].at[q].set(0.0)
        # energy: K_ATP-like conductance from the column ATP deficit
        katp = 6.0 * (1.0 - st["atp"]) ** 2
        # NE (locus coeruleus) raises gain by reducing adaptation current scale (cheap proxy)
        I_ext = I_ext * (1.0 + 0.3 * ne)
        spk_ms = jnp.zeros(N, bool)
        for _ in range(sub if FL["membrane"] else 0):
            G = G * decay_vec
            st, spk = membrane_substep(st, G, I_ext, katp)
            spk_ms = spk_ms | spk
        if not FL["membrane"]:
            spk_ms = I_ext > 400.0
        s = spk_ms.astype(f)
        # Tsodyks-Markram STP per presynaptic cell, two synapse classes (depressing, facilitating)
        U = jnp.array([[0.5], [0.1]], f)
        tau_d = jnp.array([[800.0], [100.0]], f)
        tau_f = jnp.array([[20.0], [1000.0]], f)
        x, u = st["x"], st["u"]
        x = 1.0 - (1.0 - x) * jnp.exp(-1.0 / tau_d)
        u = U + (u - U) * jnp.exp(-1.0 / tau_f)
        u_plus = u + U * (1.0 - u) * s[None, :]
        rel = u_plus * x                     # release fraction for a spike now
        x = x - rel * s[None, :]
        u = u_plus
        # astrocyte gliotransmitter scales release of L23 synapses in its domain (De Pitta 2011)
        Gm_astro = st["astro"][3]
        glia = jnp.where(astro_of >= 0, 1.0 + 0.5 * Gm_astro[jnp.maximum(astro_of, 0)], 1.0)
        # event-driven delivery: compact spiking presynaptic cells
        idx = jnp.nonzero(spk_ms, size=s_max, fill_value=N)[0]
        if not FL["events"]:
            idx = jnp.full(s_max, N)
        valid = idx < N
        ii = jnp.minimum(idx, N - 1)
        sl = T_slot[ii]                       # [s_max, kmax]
        ww = T_w[ii]
        # plastic synapses: efficacy from GB rho
        pid = T_pid[ii]
        eff_pl = jnp.where(pid >= 0, w_lo + (w_hi - w_lo) * st["rho"][jnp.maximum(pid, 0)], 1.0)
        stp = jnp.where(T_stp[ii] == 0, rel[0, ii][:, None], rel[1, ii][:, None]) / jnp.where(
            T_stp[ii] == 0, 0.5, 0.1)
        inc = ww * eff_pl * stp * glia[ii][:, None] * valid[:, None]
        slot_q = (t + T_d[ii]) % QDEPTH
        if FL["events"]:
            Q = Q.at[slot_q, sl].add(inc)
        # GB calcium-based plasticity (pre calcium delayed by D=14 ms via the history ring)
        hist = st["hist"].at[t % HIST].set(s)
        pre_d = hist[(t - 14) % HIST]
        cpre = st["cpre"] * kca + GB["Cpre"] * pre_d
        cpost = st["cpost"] * kca + GB["Cpost"] * s
        c = cpre[pl_pre] + cpost[pl_post]
        rho = st["rho"]
        rho = jnp.clip(rho + gb_drho(rho, c, 1.0 + da), 0.0, 1.0) if FL["gb"] else st["rho"]
        # astrocytes (Li-Rinzel / De Pitta ChI model), Euler at 1 ms (tau ~ s)
        ip3, ca, h, gm = st["astro"]
        dom_rate = jax.ops.segment_sum(s * (astro_of >= 0), jnp.maximum(astro_of, 0), 64) / 8.0
        v_beta = 0.5e-3 * dom_rate * 1000.0
        dip3 = v_beta - (ip3 - 0.16) / 7000.0
        m_inf = ip3 / (ip3 + 0.13); n_inf = ca / (ca + 0.08)
        J_chan = 6e-3 * (m_inf ** 3) * (n_inf ** 3) * (h ** 3) * (2.0 - ca * 1.185)
        J_pump = 0.9e-3 * ca ** 2 / (ca ** 2 + 0.05 ** 2)
        J_leak = 0.11e-3 * (2.0 - ca * 1.185)
        dca = J_chan + J_leak - J_pump
        Q2 = 0.13 * (ip3 + 1.049) / (ip3 + 0.9434)
        dh_ = 0.2e-3 * (Q2 - h * (Q2 + ca))
        dgm = -gm / 120.0 + 0.05 * (ca > 0.3)
        astro = jnp.stack([ip3 + dip3, jnp.clip(ca + dca, 0, 5), jnp.clip(h + dh_, 0, 1), jnp.clip(gm + dgm, 0, 1)]) if FL["astro"] else st["astro"]
        # neurovascular energy budget per column (+ subcortical): ATP supply ~ CBF * lumen
        rate_c = jax.ops.segment_sum(s, col_of, 5) / jnp.array([460., 460., 460., 460., 624.], f)
        kext = st["kext"] * math.exp(-1 / 200.0) + 0.5 * rate_c
        cbf = st["cbf"] + (1.0 + 2.0 * kext - st["cbf"]) / 1500.0
        supply = 2e-4 * cbf * (0.25 + 0.75 * lumen)
        cost = 1.5e-4 + 0.02 * rate_c
        atp = jnp.clip(st["atp"] + supply * (1.0 - st["atp"]) * 4 - cost * st["atp"] * 0.2, 0.0, 1.0)
        st = dict(st, G=G, Q=Q, hist=hist, x=x, u=u, cpre=cpre, cpost=cpost, rho=rho, astro=astro,
                  atp=atp, cbf=cbf, kext=kext, t=t + 1)
        return st, s

    bg_mu, bg_sigma = jnp.asarray(P["bg_mu"]), jnp.asarray(P["bg_sigma"])

    def run_window(st, ext, lumen, da, ne):
        Ds = ext.shape[0]
        da = jnp.broadcast_to(jnp.asarray(da, f), (Ds,))
        ne = jnp.broadcast_to(jnp.asarray(ne, f), (Ds,))
        inp = (ext, jnp.full(Ds, lumen, f), da, ne)
        return lax.scan(ms_step, st, inp)

    def run_window_bg(st, bus_I, lumen, da, ne):
        # background bombardment from the rest of the (unmodelled) brain: deterministic, keyed by state
        # keyed by absolute tick, so the draw is identical whatever the window size (lockstep == windowed)
        ticks = st["t"] + jnp.arange(bus_I.shape[0])
        keys = jax.vmap(jax.random.fold_in, (None, 0))(st["key"], ticks)
        noise = jax.vmap(lambda k: jax.random.normal(k, (bus_I.shape[1],), f))(keys)
        ext = bg_mu[None, :] + bg_sigma[None, :] * noise + bus_I
        return run_window(st, ext, lumen, da, ne)

    if FL.get("bg_inside", False):
        return jax.jit(run_window_bg)

    return jax.jit(run_window)
