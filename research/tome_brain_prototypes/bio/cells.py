"""Neuron models for the Tome's JAX partition. Units: mV, ms, nS, pA, pF.

Every compartment is advanced with exponential Euler: over one step the conductances (and the neighbouring
compartment voltages) are frozen, so each compartment relaxes exactly toward its instantaneous steady state

    V(t+dt) = V_inf + (V - V_inf) * exp(-dt * g_tot / C),   V_inf = (sum g_k E_k + I) / g_tot.

The compartments are coupled Jacobi-style (each sees its neighbour's previous voltage). Because every compartment's
g_tot contains its own leak plus all of its coupling conductances, the coupling matrix is strictly diagonally
dominant and the scheme cannot blow up at any dt (it only gets less accurate). This replaces the forward-Euler
integrator of the spec, which NaNs at dt = 0.05 ms.

Cell kinds (one flat vector of neurons; a per-neuron 'ncomp' says which compartments are real):
  ncomp 3: human L2/3 pyramidal cell: soma (AdEx), basal (NMDA, Jahr-Stevens Mg block), apical tuft with the
           Gidon-2020 dCaAP (amplitude falls as the drive grows past threshold: the XOR nonlinearity).
  ncomp 2: L5 ET / CA1 pyramidal cell: soma (AdEx) + apical tuft with a monotonic BAC Ca2+ spike (Larkum 1999).
  ncomp 1: AdEx point neurons (PV, SST, VIP, LAMP5/NGF, L4, L6, TC/TRN with T-type Ca2+, pacemakers).
"""
from __future__ import annotations

import jaxpath  # noqa: F401
import jax
import jax.numpy as jnp

E_AMPA, E_NMDA, E_GABAA, E_GABAB, E_K, E_CA = 0.0, 0.0, -70.0, -90.0, -90.0, 120.0
MG_MM = 1.0
V_DETECT = -30.0   # AdEx spike detection (the exponential runaway is certain past here)

TAU = dict(ampa=2.0, nmda=100.0, gabaa=6.0, gabaa_slow=20.0, gabab=150.0)


def mg_block(v):
    """Jahr & Stevens (1990) NMDA Mg2+ block."""
    return 1.0 / (1.0 + (MG_MM / 3.57) * jnp.exp(-0.062 * v))


def expeuler(v, g_tot, i_num, c, dt):
    v_inf = i_num / g_tot
    return v_inf + (v - v_inf) * jnp.exp(-dt * g_tot / c)


def t_type(v):
    """Reduced T-type Ca2+ (Destexhe et al. 1996, 36 C): m instantaneous, h slow."""
    m_inf = 1.0 / (1.0 + jnp.exp(-(v + 57.0) / 6.2))
    h_inf = 1.0 / (1.0 + jnp.exp((v + 81.0) / 4.0))
    tau_h = 8.0 + 60.0 / (1.0 + jnp.exp((v + 78.0) / 3.0))
    return m_inf, h_inf, tau_h


def dcaap_gain(drive, d_th, tau_k):
    """Gidon et al. 2020: dCaAP amplitude is maximal at threshold and decays as the stimulus grows past it."""
    return jnp.exp(-jnp.maximum(0.0, drive / d_th - 1.0) / tau_k)


def substep(p, s, i_ext, dt):
    """One exponential-Euler substep for every neuron. p: parameter dict, s: state dict, i_ext: [3, N] pA."""
    vs, vb, va, w = s["vs"], s["vb"], s["va"], s["w"]
    three, two, one = p["ncomp"] == 3, p["ncomp"] == 2, p["ncomp"] == 1
    has_b = three
    has_a = three | two
    # conductances seen by each compartment; for point neurons everything lands on the soma
    gA_b, gA_a = s["gA_b"], s["gA_a"]
    gN_b, gN_a = s["gN_b"], s["gN_a"]
    gGa, gGb = s["gGA_a"], s["gGB_a"]
    route_b = jnp.where(has_b, 0.0, 1.0)
    route_a = jnp.where(has_a, 0.0, 1.0)
    gA_s = s["gA_s"] + route_b * gA_b + route_a * gA_a
    gN_s = route_b * gN_b + route_a * gN_a
    gG_s = s["gGA_s"] + route_a * gGa
    gGB_s = route_a * gGb

    katp = p["g_katp"] / (1.0 + (s["atp"] / 0.55) ** 4)           # K_ATP opens as ATP falls
    mT, hT_inf, tau_hT = t_type(vs)
    gT = p["g_T"] * mT * mT * s["hT"]

    g_sb = jnp.where(has_b, p["g_sb"], 0.0)
    g_sa = jnp.where(has_a, p["g_sa"], 0.0)
    ex = p["gL_s"] * p["DT"] * jnp.exp(jnp.minimum((vs - p["VT"]) / p["DT"], 20.0))
    g_tot_s = p["gL_s"] + g_sb + g_sa + gA_s + gN_s * mg_block(vs) + gG_s + gGB_s + katp + gT
    i_s = (p["gL_s"] * p["EL"] + g_sb * vb + g_sa * va + gG_s * E_GABAA + gGB_s * E_GABAB + katp * E_K
           + gT * E_CA + ex - w + p["I_bias"] + i_ext[0])
    vs_new = expeuler(vs, g_tot_s, i_s, p["C_s"], dt)

    # basal dendrite (L2/3 human PC only)
    g_tot_b = p["gL_b"] + g_sb + gA_b + gN_b * mg_block(vb) + katp
    i_b = p["gL_b"] * p["EL"] + g_sb * vs + katp * E_K + i_ext[1]
    vb_new = jnp.where(has_b, expeuler(vb, g_tot_b, i_b, p["C_b"], dt), vs_new)

    # apical tuft: dCaAP (3-comp) or BAC Ca2+ spike (2-comp), as a bi-exponential current with its own refractory
    i_dca = p["A_dca"] * s["dca_gain"] * (s["dca_d"] - s["dca_r"])
    gKa = p["g_Ka"] / (1.0 + jnp.exp(-(va + 35.0) / 5.0))         # tuft outward rectifier (Kv), clamps plateaus
    g_tot_a = p["gL_a"] + g_sa + gA_a + gN_a * mg_block(va) + gGa + gGb + katp + gKa
    i_a = (p["gL_a"] * p["EL"] + g_sa * vs + gGa * E_GABAA + gGb * E_GABAB + (katp + gKa) * E_K + i_dca
           + s["bap"] * p["I_bap"] + i_ext[2])
    va_new = jnp.where(has_a, expeuler(va, g_tot_a, i_a, p["C_a"], dt), vs_new)

    # dCaAP / BAC trigger on an upward threshold crossing of the tuft
    drive = (gA_a + gN_a * mg_block(va)) * (E_AMPA - va) + i_ext[2]
    onset = has_a & (va < p["th_dca"]) & (va_new >= p["th_dca"]) & (s["dca_ref"] <= 0.0)
    gain = jnp.where(three, dcaap_gain(drive, p["D_th"], p["tau_k"]), 1.0)
    dca_gain = jnp.where(onset, gain, s["dca_gain"])
    norm = p["dca_norm"]
    dca_r = jnp.where(onset, norm, s["dca_r"] * jnp.exp(-dt / p["tau_dr"]))
    dca_d = jnp.where(onset, norm, s["dca_d"] * jnp.exp(-dt / p["tau_dd"]))
    dca_ref = jnp.where(onset, p["ref_dca"], s["dca_ref"] - dt)

    # adaptation (exact for frozen V), refractoriness, spikes
    w_inf = p["a"] * (vs - p["EL"])
    w_new = w_inf + (w - w_inf) * jnp.exp(-dt / p["tau_w"])
    refr = s["ref"] > 0.0
    vs_new = jnp.where(refr, p["Vr"], vs_new)
    spk = (vs_new >= V_DETECT) & ~refr
    vs_new = jnp.where(spk, p["Vr"], vs_new)
    w_new = jnp.where(spk, w_new + p["b"], w_new)
    ref = jnp.where(spk, p["t_ref"], s["ref"] - dt)
    bap = jnp.where(spk, 1.0, s["bap"] * jnp.exp(-dt / 1.0))       # ~1 ms back-propagating AP into the tuft
    hT = hT_inf + (s["hT"] - hT_inf) * jnp.exp(-dt / tau_hT)

    # synaptic conductances decay exactly
    d = {k: jnp.exp(-dt / TAU[v]) for k, v in (("a", "ampa"), ("n", "nmda"), ("ga", "gabaa"),
                                                ("gas", "gabaa_slow"), ("gb", "gabab"))}
    new = dict(s)
    new.update(vs=vs_new, vb=vb_new, va=va_new, w=w_new, ref=ref, bap=bap, hT=hT,
               dca_r=dca_r, dca_d=dca_d, dca_gain=dca_gain, dca_ref=dca_ref,
               gA_s=s["gA_s"] * d["a"], gA_b=gA_b * d["a"], gA_a=gA_a * d["a"],
               gN_b=gN_b * d["n"], gN_a=gN_a * d["n"], gGA_s=s["gGA_s"] * d["ga"],
               gGA_a=gGa * d["gas"], gGB_a=gGb * d["gb"])
    return new, spk


# ------------------------------------------------------------------------------------------- parameter sets
def _bi_norm(tr, td):
    tp = tr * td / (td - tr) * jnp.log(td / tr)
    return 1.0 / (jnp.exp(-tp / td) - jnp.exp(-tp / tr))


KINDS = {
    # human L2/3 PC: Cm 0.45 uF/cm2, Rm ~ 39 kOhm cm2 (Eyal et al. 2016) -> tau_m ~ 17.5 ms in every compartment
    "pc23h": dict(ncomp=3, C_s=30.0, gL_s=1.71, C_b=60.0, gL_b=3.43, C_a=45.0, gL_a=2.57, g_sb=12.0, g_sa=5.0,
                  EL=-72.0, VT=-50.0, DT=2.0, a=2.0, b=60.0, tau_w=150.0, Vr=-60.0, t_ref=2.0,
                  th_dca=-40.0, D_th=270.0, tau_k=0.30, A_dca=1200.0, tau_dr=3.0, tau_dd=18.0, ref_dca=40.0,
                  I_bap=60.0, g_Ka=8.0),
    # L5 ET thick-tufted / CA1 PC: BAC firing (monotonic Ca2+ spike)
    "pc5": dict(ncomp=2, C_s=150.0, gL_s=7.5, C_a=60.0, gL_a=3.0, g_sa=6.0, EL=-70.0, VT=-50.0, DT=2.0, a=2.0,
                b=80.0, tau_w=100.0, Vr=-58.0, t_ref=2.0, th_dca=-35.0, A_dca=700.0, tau_dr=2.0, tau_dd=25.0,
                ref_dca=40.0, I_bap=250.0, g_Ka=10.0),
    "rs": dict(C_s=150.0, gL_s=8.0, EL=-70.0, VT=-50.0, DT=2.0, a=2.0, b=40.0, tau_w=120.0, Vr=-58.0, t_ref=2.0),
    "pv": dict(C_s=90.0, gL_s=9.0, EL=-68.0, VT=-48.0, DT=1.5, a=0.0, b=0.0, tau_w=10.0, Vr=-60.0, t_ref=1.0),
    "sst": dict(C_s=100.0, gL_s=5.0, EL=-65.0, VT=-52.0, DT=2.0, a=4.0, b=40.0, tau_w=200.0, Vr=-55.0, t_ref=2.0),
    "vip": dict(C_s=60.0, gL_s=4.0, EL=-65.0, VT=-50.0, DT=2.0, a=2.0, b=20.0, tau_w=100.0, Vr=-56.0, t_ref=2.0),
    "ngf": dict(C_s=80.0, gL_s=5.0, EL=-70.0, VT=-45.0, DT=2.0, a=1.0, b=10.0, tau_w=300.0, Vr=-60.0, t_ref=2.0),
    "tc": dict(C_s=200.0, gL_s=10.0, EL=-65.0, VT=-50.0, DT=2.0, a=0.0, b=10.0, tau_w=50.0, Vr=-55.0, t_ref=2.0,
               g_T=60.0),
    "trn": dict(C_s=150.0, gL_s=8.0, EL=-70.0, VT=-50.0, DT=2.0, a=0.0, b=20.0, tau_w=60.0, Vr=-58.0, t_ref=2.0,
                g_T=90.0),
    "dg": dict(C_s=60.0, gL_s=5.0, EL=-78.0, VT=-48.0, DT=2.0, a=2.0, b=40.0, tau_w=100.0, Vr=-60.0, t_ref=3.0),
    # pacemakers: intrinsic I_bias sets the tonic rate (VTA DA ~4 Hz, LC ~2 Hz, DRN ~1.5 Hz, NBM ~3 Hz)
    "da": dict(C_s=150.0, gL_s=6.0, EL=-60.0, VT=-48.0, DT=3.0, a=4.0, b=50.0, tau_w=300.0, Vr=-55.0, t_ref=5.0,
               I_bias=140.0),
    "pace": dict(C_s=150.0, gL_s=6.0, EL=-60.0, VT=-48.0, DT=3.0, a=4.0, b=50.0, tau_w=400.0, Vr=-55.0, t_ref=5.0,
                 I_bias=120.0),
}

DEFAULTS = dict(ncomp=1, C_b=60.0, gL_b=3.0, C_a=45.0, gL_a=2.5, g_sb=0.0, g_sa=0.0, I_bias=0.0, g_T=0.0,
                g_katp=0.0, th_dca=1e9, D_th=1.0, tau_k=0.3, A_dca=0.0, tau_dr=2.0, tau_dd=20.0, ref_dca=40.0,
                I_bap=0.0, g_Ka=0.0)


def build_params(kinds: list[str], g_katp: float = 25.0) -> dict:
    rows = []
    for k in kinds:
        d = dict(DEFAULTS)
        d.update(KINDS[k])
        d["g_katp"] = g_katp
        rows.append(d)
    p = {key: jnp.array([r[key] for r in rows], dtype=jnp.float32) for key in rows[0]}
    p["ncomp"] = p["ncomp"].astype(jnp.int32)
    p["dca_norm"] = _bi_norm(p["tau_dr"], p["tau_dd"])
    return p


def init_state(p) -> dict:
    n = p["EL"].shape[0]
    z = jnp.zeros(n, jnp.float32)
    return dict(vs=p["EL"], vb=p["EL"], va=p["EL"], w=z, ref=z, bap=z, hT=jnp.ones(n) * 0.5,
                dca_r=z, dca_d=z, dca_gain=z, dca_ref=z, gA_s=z, gA_b=z, gA_a=z, gN_b=z, gN_a=z,
                gGA_s=z, gGA_a=z, gGB_a=z, atp=jnp.ones(n))
