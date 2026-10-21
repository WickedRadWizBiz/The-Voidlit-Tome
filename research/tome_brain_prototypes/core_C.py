"""Prototype of the Tome's JAX partition: 6 cortical columns (6 layers, human L2/3 3-compartment PCs with dCaAP,
L5 ET 2-compartment with BAC Ca spike, AdEx interneurons PV/SST/VIP/LAMP5), core + matrix thalamus with TRN and
T-type Ca, hippocampus (DG, CA3, CA1), brainstem modulatory nuclei (LC, DRN, NBM).

Conductance synapses (AMPA, NMDA+Mg block, GABA_A fast/slow, GABA_B), Tsodyks-Markram STP per presynaptic class,
per-pathway axonal delays through a 1-ms spike ring buffer, Graupner-Brunel calcium-threshold occupancy for plastic
projections, Vogels iSTDP on PV->PC, ATP accounting, K_ATP coupling.

Integration: exponential Euler for every voltage (exact for the linear part, conductances frozen over a step),
exact exponential updates for gates, traces and conductances. dt = 0.25 ms, delivery window 1 ms (= the minimum
intra-partition delay), sync window 5 ms (= the minimum cross-partition delay).

This is a timing/stability prototype: connectivity statistics are plausible, not tuned.
"""
import jaxenv  # noqa: F401  (adds the JAX site-packages)
import time
import numpy as np
import jax
import jax.numpy as jnp

DT = 0.25
GE, GESD, GI, GISD = 9.0, 3.0, 14.0, 5.0
SUB = 4            # substeps per delivery millisecond
WIN_MS = 5         # cross-partition sync window (ms)
DMAX = 16          # ring buffer depth (ms)
NCOL = 6

# ----------------------------------------------------------------------------------------- column layout
LAYOUT = [  # (name, n, kind)
    ("L1_LAMP5", 6, "ngf"), ("L23_IT", 80, "pc23"), ("L23_PV", 10, "pv"), ("L23_SST", 8, "sst"),
    ("L23_VIP", 8, "vip"), ("L4_IT", 30, "rs"), ("L4_PV", 6, "pv"), ("L5_ET", 20, "pc5"), ("L5_IT", 20, "rs"),
    ("L5_PV", 6, "pv"), ("L5_SST", 6, "sst"), ("L6_CT", 30, "rs"), ("L6_PV", 4, "pv"), ("L6_SST", 6, "sst"),
]
NC = sum(n for _, n, _ in LAYOUT)            # 240 per column
EXC = {"pc23", "pc5", "rs"}
SUBC = [("TC", 12, "tc"), ("TRN", 8, "trn"), ("MTX", 8, "tc")]          # per-column thalamus
HIPPO = [("DG", 100, "dg"), ("CA3", 100, "rs"), ("CA3_PV", 12, "pv"), ("CA1", 60, "pc5"), ("CA1_OLM", 10, "sst")]
NUCLEI = [("LC", 15, "pace"), ("DRN", 15, "pace"), ("NBM", 15, "pace")]

KIND = {  # C pF, gL nS, EL, VT, DeltaT, Vr, a nS, b pA, tau_w ms, ref ms
    "pc23": (150., 7.5, -70., -50., 2.0, -58., 2.0, 60., 150., 2.0),
    "pc5":  (200., 10., -68., -50., 2.0, -55., 4.0, 80., 100., 2.0),
    "rs":   (120., 6.0, -70., -50., 2.0, -58., 2.0, 40., 150., 2.0),
    "pv":   (60.,  6.0, -68., -48., 0.8, -60., 0.0, 0.0, 30., 1.0),
    "sst":  (80.,  4.0, -64., -50., 2.0, -56., 4.0, 20., 200., 1.5),
    "vip":  (60.,  3.0, -66., -50., 2.0, -56., 2.0, 30., 150., 1.5),
    "ngf":  (80.,  4.0, -70., -48., 1.5, -60., 1.0, 10., 400., 1.5),
    "tc":   (100., 5.0, -65., -50., 2.0, -55., -2.0, 0.0, 200., 2.0),
    "trn":  (80.,  4.0, -65., -50., 2.0, -55., -1.0, 0.0, 200., 2.0),
    "dg":   (100., 8.0, -75., -48., 2.0, -60., 2.0, 60., 200., 2.0),
    "pace": (80.,  3.0, -60., -50., 2.0, -56., 1.0, 40., 300., 2.0),
}


def build(seed=0):
    rng = np.random.default_rng(seed)
    kinds, groups = [], {}
    def add(name, n, kind):
        groups[name] = (len(kinds), len(kinds) + n)
        kinds.extend([kind] * n)
    for c in range(NCOL):
        for name, n, kind in LAYOUT:
            add(f"c{c}.{name}", n, kind)
    for c in range(NCOL):
        for name, n, kind in SUBC:
            add(f"c{c}.{name}", n, kind)
    for name, n, kind in HIPPO + NUCLEI:
        add(name, n, kind)
    N = len(kinds)
    par = np.array([KIND[k] for k in kinds], np.float32)
    is_exc = np.array([k in EXC or k in ("tc", "dg") or (k == "pace") for k in kinds])
    is_pc23 = np.array([k == "pc23" for k in kinds])
    is_pc5 = np.array([k == "pc5" for k in kinds])
    has_t = np.array([k in ("tc", "trn") for k in kinds])

    def blk(a, b):
        return np.arange(*groups[a])
    # ------------------------------------------------ local block-diagonal (1 ms): pre x post inside a column
    Wl = np.zeros((5, NCOL, NC, NC), np.float32)   # 0 E->soma, 1 E->basal, 2 E->apical, 3 I->soma, 4 I->apical
    off = {name: (sum(n for nm, n, _ in LAYOUT[:i]), sum(n for nm, n, _ in LAYOUT[:i + 1]))
           for i, (name, _, _) in enumerate(LAYOUT)}
    def conn(c, pre, post, p, w, tgt):
        a0, a1 = off[pre]; b0, b1 = off[post]
        m = rng.random((a1 - a0, b1 - b0)) < p
        Wl[tgt, c, a0:a1, b0:b1] += m * w * rng.lognormal(0, 0.5, m.shape)
    for c in range(NCOL):
        conn(c, "L4_IT", "L23_IT", 0.15, 0.6, 1); conn(c, "L23_IT", "L23_IT", 0.10, 0.4, 1)
        conn(c, "L23_IT", "L5_ET", 0.15, 0.5, 1); conn(c, "L23_IT", "L5_IT", 0.15, 0.5, 1)
        conn(c, "L5_IT", "L5_ET", 0.10, 0.4, 1); conn(c, "L5_ET", "L6_CT", 0.05, 0.4, 0)
        conn(c, "L6_CT", "L4_IT", 0.05, 0.3, 0)
        for e in ("L23_IT", "L4_IT", "L5_ET", "L5_IT", "L6_CT"):
            for pv in ("L23_PV", "L4_PV", "L5_PV", "L6_PV"):
                conn(c, e, pv, 0.25 if e[:2] == pv[:2] else 0.05, 1.0, 0)
            for sst in ("L23_SST", "L5_SST", "L6_SST"):
                conn(c, e, sst, 0.20 if e[:2] == sst[:2] else 0.05, 0.6, 0)
        conn(c, "L23_IT", "L23_VIP", 0.10, 0.4, 0)
        for pv, tgts in (("L23_PV", ("L23_IT", "L23_PV")), ("L4_PV", ("L4_IT", "L4_PV")),
                         ("L5_PV", ("L5_ET", "L5_IT", "L5_PV")), ("L6_PV", ("L6_CT", "L6_PV"))):
            for t in tgts:
                conn(c, pv, t, 0.40, 2.0, 3)
        for sst in ("L23_SST", "L5_SST", "L6_SST"):
            for t in ("L23_IT", "L5_ET"):
                conn(c, sst, t, 0.35, 1.5, 4)
            conn(c, sst, "L23_PV", 0.2, 0.8, 3)
        conn(c, "L23_VIP", "L23_SST", 0.5, 1.2, 3); conn(c, "L23_VIP", "L5_SST", 0.2, 0.8, 3)
        conn(c, "L1_LAMP5", "L23_IT", 0.5, 0.8, 4); conn(c, "L1_LAMP5", "L5_ET", 0.4, 0.8, 4)
    # ------------------------------------------------ intercolumn (myelinated, per column-pair delay)
    pre_off = (off["L23_IT"], off["L5_IT"])
    NP = 100
    Wc = np.zeros((NCOL, NP, NCOL, NC), np.float32)
    for c1 in range(NCOL):
        for c2 in range(NCOL):
            if c1 == c2:
                continue
            m = rng.random((NP, NC)) < 0.04
            Wc[c1, :, c2, :] = m * 0.4 * rng.lognormal(0, 0.5, m.shape)
    ff = np.array([[1.0 if c1 < c2 else 0.0 for c2 in range(NCOL)] for c1 in range(NCOL)], np.float32)  # FF->basal
    L = rng.uniform(20.0, 80.0, (NCOL, NCOL)).astype(np.float32)           # path length mm
    # ------------------------------------------------ everything else: one sparse matrix per delay class
    rows, cols, vals, tgts, dly = [], [], [], [], []
    def proj(pre_idx, post_idx, p, w, tgt, d):
        m = rng.random((len(pre_idx), len(post_idx))) < p
        i, j = np.nonzero(m)
        rows.append(pre_idx[i]); cols.append(post_idx[j]); vals.append((w * rng.lognormal(0, 0.4, len(i))).astype(np.float32))
        tgts.append(np.full(len(i), tgt)); dly.append(np.full(len(i), d))
    for c in range(NCOL):
        g = lambda nm: blk(f"c{c}.{nm}", None)
        proj(g("TC"), np.concatenate([g("L4_IT"), g("L4_PV"), g("L5_ET"), g("L6_CT")]), 0.3, 1.0, 1, 2)
        proj(g("MTX"), np.concatenate([g("L1_LAMP5"), g("L23_IT"), g("L5_ET")]), 0.3, 0.6, 2, 2)
        proj(g("L6_CT"), np.concatenate([g("TC"), g("TRN")]), 0.3, 0.4, 0, 4)
        proj(g("L5_ET"), g("MTX"), 0.3, 0.8, 0, 4)
        proj(g("TRN"), g("TC"), 0.5, 2.0, 3, 1)
        proj(g("TC"), g("TRN"), 0.4, 1.0, 0, 1)
    ec = np.concatenate([blk("c5.L23_IT", None)])
    proj(ec, blk("DG", None), 0.2, 0.6, 1, 4); proj(ec, blk("CA3", None), 0.1, 0.4, 1, 4)
    proj(blk("DG", None), blk("CA3", None), 0.05, 3.0, 1, 2)
    proj(blk("CA3", None), blk("CA3", None), 0.1, 0.5, 1, 1)
    proj(blk("CA3", None), blk("CA3_PV", None), 0.3, 1.0, 0, 1); proj(blk("CA3_PV", None), blk("CA3", None), 0.4, 2.0, 3, 1)
    proj(blk("CA3", None), blk("CA1", None), 0.15, 0.5, 1, 2); proj(ec, blk("CA1", None), 0.15, 0.4, 2, 4)
    proj(blk("CA1", None), blk("CA1_OLM", None), 0.3, 0.8, 0, 1); proj(blk("CA1_OLM", None), blk("CA1", None), 0.4, 1.5, 4, 1)
    proj(blk("CA1", None), np.concatenate([blk("c5.L5_IT", None), blk("c4.L5_IT", None)]), 0.1, 0.5, 1, 8)
    for c in range(NCOL):
        proj(blk("LC", None), blk(f"c{c}.L23_PV", None), 0.0, 0.0, 0, 6)  # LC/DRN/NBM act by volume transmission
    rows, cols, vals, tgts, dly = map(np.concatenate, (rows, cols, vals, tgts, dly))
    sparse = []
    for d in sorted(set(dly.tolist())):
        for t in range(5):
            m = (dly == d) & (tgts == t)
            if m.any():
                W = np.zeros((N, N), np.float32)
                np.add.at(W, (rows[m], cols[m]), vals[m])
                pre_used = np.unique(rows[m]); post_used = np.unique(cols[m])
                sparse.append((d, t, pre_used, post_used, W[np.ix_(pre_used, post_used)]))
    col_base = np.array([groups[f"c{c}.L1_LAMP5"][0] for c in range(NCOL)])
    return dict(N=N, par=par, is_exc=is_exc, is_pc23=is_pc23, is_pc5=is_pc5, has_t=has_t, Wl=Wl, Wc=Wc, ff=ff,
                L=L, sparse=sparse, groups=groups, col_base=col_base, pre_off=pre_off, NP=NP)


def make_window(net):
    N = net["N"]
    par = jnp.asarray(net["par"])
    C, gL, EL, VT, DTh, Vr, a, b, tw, tref = [par[:, i] for i in range(10)]
    is_exc = jnp.asarray(net["is_exc"]); pc23 = jnp.asarray(net["is_pc23"]); pc5 = jnp.asarray(net["is_pc5"])
    has_t = jnp.asarray(net["has_t"])
    Wl = jnp.asarray(net["Wl"]); Wc = jnp.asarray(net["Wc"]); ff = jnp.asarray(net["ff"])
    col_idx = jnp.asarray((net["col_base"][:, None] + np.arange(NC)[None, :]).reshape(-1))     # [6*240]
    pre_idx = jnp.asarray(np.concatenate([
        net["col_base"][c] + np.r_[np.arange(*net["pre_off"][0]), np.arange(*net["pre_off"][1])]
        for c in range(NCOL)]))                                                                # [6*100]
    sp = [(d, t, jnp.asarray(pi), jnp.asarray(po), jnp.asarray(W)) for d, t, pi, po, W in net["sparse"]]
    dend = pc23 | pc5
    gsa = jnp.where(pc23, 8.0, jnp.where(pc5, 10.0, 0.0)); gsb = jnp.where(dend, 20.0, 0.0)
    E_AMPA, E_GA, E_GB = 0.0, -75.0, -95.0
    tauA, tauNr, tauNd, tauGA, tauGAs, tauGB = 2.0, 2.0, 80.0, 6.0, 15.0, 200.0
    dec = lambda tau: jnp.exp(-DT / tau)

    def B(V):
        return 1.0 / (1.0 + jnp.exp(-0.062 * V) / 3.57)

    def substep(s, x):
        g_in, noise = x
        (Vs, Vb, Va, w, ref, gAs, gAb, gAa, gNr, gNb, gNa, gGAs, gGAa, gGB, s1, s2, aref, Aamp, hT, atp) = s
        # synaptic input arriving this substep (delivered at the 1 ms boundary, split evenly over SUB substeps)
        gAs = gAs * dec(tauA) + g_in[0]; gAb = gAb * dec(tauA) + g_in[1]; gAa = gAa * dec(tauA) + g_in[2]
        gNr = gNr * dec(tauNr) + 0.3 * (g_in[1] + g_in[2])
        gNb = 0.3 * g_in[1] + gNb * dec(tauNd); gNa = 0.3 * g_in[2] + gNa * dec(tauNd)
        gGAs = gGAs * dec(tauGA) + g_in[3]; gGAa = gGAa * dec(tauGAs) + g_in[4]; gGB = gGB * dec(tauGB) + 0.2 * g_in[4]
        # point-conductance background (Destexhe 2001), OU-like per substep
        ge = jnp.maximum(0.0, GE + GESD * noise[0])
        gi = jnp.maximum(0.0, GI + GISD * noise[1])
        # K_ATP (opens as ATP falls): g = gmax / (1 + (ATP/K)^4)
        gK = 6.0 / (1.0 + (atp / 0.35) ** 4)
        # dCaAP (human L2/3 tuft): threshold-triggered, amplitude decays with drive at onset (Gidon et al. 2020)
        Iexc_a = (gAa + gNa * B(Va)) * (E_AMPA - Va)
        onset = pc23 & (Va >= -42.0) & (aref <= 0.0)
        Aamp = jnp.where(onset, jnp.exp(-jnp.maximum(0.0, Iexc_a - 360.0) / 360.0), Aamp)
        s1 = (s1 + onset) * dec(15.0); s2 = (s2 + onset) * dec(1.5)
        aref = jnp.where(onset, 60.0, jnp.maximum(aref - DT, 0.0))
        gdca = 45.0 * Aamp * (s1 - s2) / 0.6
        # L5 ET BAC Ca spike: sigmoidal high-threshold Ca conductance on the tuft (Larkum 1999, Naud 2014 style)
        gca5 = jnp.where(pc5, 20.0 / (1.0 + jnp.exp(-(Va + 30.0) / 3.0)), 0.0)
        # tuft
        ga = gL * 0.5 + gsa + gAa + gNa * B(Va) + gGAa + gGB + gdca + gca5 + 0.5 * ge + 0.3 * gi
        Vai = (gL * 0.5 * EL + gsa * Vs + (gAa + gNa * B(Va)) * E_AMPA + (gGAa + 0.3 * gi) * E_GA + gGB * E_GB
               + (gdca + gca5) * 60.0) / ga
        Va_n = jnp.where(dend, Vai + (Va - Vai) * jnp.exp(-DT * ga / (C * 0.5)), EL)
        # basal
        gb = gL * 0.7 + gsb + gAb + gNb * B(Vb) + ge + 0.5 * gi
        Vbi = (gL * 0.7 * EL + gsb * Vs + (gAb + gNb * B(Vb)) * E_AMPA + 0.5 * gi * E_GA) / gb
        Vb_n = jnp.where(dend, Vbi + (Vb - Vbi) * jnp.exp(-DT * gb / (C * 0.7)), EL)
        # T-type Ca (thalamus): m_inf^2 h, h with tau ~ 30 ms (Destexhe 1998 simplified)
        minf = 1.0 / (1.0 + jnp.exp(-(Vs + 57.0) / 6.2)); hinf = 1.0 / (1.0 + jnp.exp((Vs + 81.0) / 4.0))
        hT = hinf + (hT - hinf) * dec(30.0)
        gT = jnp.where(has_t, 2.0 * minf * minf * hT, 0.0)
        # soma (AdEx, exponential Euler)
        gs = gL + gsb + gsa + gAs + ge + gi + gGAs + gK + gT
        Vsi = (gL * EL + gsb * Vb + gsa * Va + (gAs + ge) * E_AMPA + (gi + gGAs) * E_GA + gK * -90.0 + gT * 120.0
               + gL * DTh * jnp.exp(jnp.minimum((Vs - VT) / DTh, 20.0)) - w) / gs
        Vs_n = Vsi + (Vs - Vsi) * jnp.exp(-DT * gs / C)
        Vs_n = jnp.where(ref > 0.0, Vr, Vs_n)
        winf = a * (Vs - EL)
        w = winf + (w - winf) * jnp.exp(-DT / tw)
        spk = Vs_n >= -30.0
        Vs_n = jnp.where(spk, Vr, Vs_n); w = jnp.where(spk, w + b, w)
        ref = jnp.where(spk, tref, jnp.maximum(ref - DT, 0.0))
        # ATP: spikes and synaptic load cost, supply relaxes toward the substrate-limited ceiling
        atp = atp + DT * (2e-5 * (1.0 - atp) - 2e-4 * spk - 1e-7 * (gAs + gAb + gAa))
        return (Vs_n, Vb_n, Va_n, w, ref, gAs, gAb, gAa, gNr, gNb, gNa, gGAs, gGAa, gGB, s1, s2, aref, Aamp, hT, atp), spk

    # split local weights by presynaptic sign: E rows -> targets 0..2, I rows -> targets 3..4
    exc_rows = np.array([k in EXC for _, n, k in LAYOUT for _ in range(n)])
    WlE = jnp.asarray(net["Wl"][:3][:, :, exc_rows, :]); WlI = jnp.asarray(net["Wl"][3:][:, :, ~exc_rows, :])
    ex_r = jnp.asarray(np.nonzero(exc_rows)[0]); in_r = jnp.asarray(np.nonzero(~exc_rows)[0])
    NCORT = NCOL * NC
    assert np.array_equal(np.asarray(col_idx), np.arange(NCORT))
    pre_local = jnp.asarray(np.r_[np.arange(*net["pre_off"][0]), np.arange(*net["pre_off"][1])])
    c1_idx = jnp.arange(NCOL)[:, None] * jnp.ones((1, NCOL), jnp.int32)

    def deliver(ring, m, delays_c, rel):
        cur = (ring[(m - 1) % DMAX] * rel)[:NCORT].reshape(NCOL, NC)
        gE = jnp.einsum("cp,tcpq->tcq", cur[:, ex_r], WlE).reshape(3, NCORT)
        gI = jnp.einsum("cp,tcpq->tcq", cur[:, in_r], WlI).reshape(2, NCORT)
        ring_pre = ring[:, :NCORT].reshape(DMAX, NCOL, NC)[:, :, pre_local]          # [D, c1, NP]
        pre = ring_pre[(m - delays_c) % DMAX, c1_idx]                                 # [c1, c2, NP]
        gc = jnp.einsum("abp,apbq->abq", pre, Wc)                                     # [c1, c2, NC]
        to_b = jnp.einsum("abq,ab->bq", gc, ff).reshape(-1); to_a = jnp.einsum("abq,ab->bq", gc, 1 - ff).reshape(-1)
        g_cort = jnp.concatenate([gE[0:1], gE[1:2] + to_b[None], gE[2:3] + to_a[None], gI], 0)
        g = jnp.zeros((5, N), jnp.float32).at[:, :NCORT].set(g_cort)
        for d, t, pi, po, W in sp:
            g = g.at[t, po].add(ring[(m - d) % DMAX][pi] @ W)
        return g

    def window(state, ring, stp, m0, delays_c, key, ext):
        """Advance WIN_MS milliseconds. ext: [WIN_MS, 5, N] extra conductance from other partitions (the bus)."""
        def ms(carry, i):
            s, ring, stp = carry
            m = m0 + i
            u, x = stp
            rel = u * x * 4.0                                                    # TM release scaling
            g = deliver(ring, m, delays_c, rel) + ext[i]
            noise = jax.random.normal(jax.random.fold_in(key, m), (2, N))
            s, spk = jax.lax.scan(substep, s, (jnp.broadcast_to(g / SUB, (SUB, 5, N)), jnp.broadcast_to(noise, (SUB, 2, N))))
            cnt = spk.sum(0).astype(jnp.float32)
            # Tsodyks-Markram, 1 ms resolution (event-driven at ms granularity)
            x = 1.0 - (1.0 - x) * jnp.exp(-1.0 / 800.0); u = 0.2 + (u - 0.2) * jnp.exp(-1.0 / 50.0)
            u = u + 0.2 * (1.0 - u) * jnp.minimum(cnt, 1.0); x = x - u * x * jnp.minimum(cnt, 1.0)
            ring = ring.at[m % DMAX].set(cnt)
            return (s, ring, (u, x)), cnt
        (state, ring, stp), cnts = jax.lax.scan(ms, (state, ring, stp), jnp.arange(WIN_MS))
        return state, ring, stp, cnts
    return jax.jit(window), N


def init_state(N):
    z = jnp.zeros(N, jnp.float32)
    return (z - 70.0, z - 70.0, z - 70.0, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z + 0.5, z + 1.0)
