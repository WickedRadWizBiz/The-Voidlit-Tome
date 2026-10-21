"""Human L2/3 pyramidal cell, 4 compartments (0 soma, 1 basal, 2 apical trunk, 3 tuft), stable at dt up to 0.1 ms.

Units: mV, ms, pF, nS, pA.  Voltage: linearly implicit (backward Euler on the conductance-linearised system, solved per
cell); gates: Rush-Larsen exponential update.  Soma channels: Pospischil 2008 regular-spiking HH (Na, Kd, M).
Human Cm = 0.45 uF/cm^2 (Eyal 2016).  Dendritic NMDA with Jahr-Stevens Mg block.  dCaAP (Gidon 2020 phenomenology):
a trunk Ca event triggered when V_trunk crosses theta from below; its amplitude falls with the apical excitatory drive
at onset, so near-threshold input gives the biggest event and strong input almost none.
"""
import jaxpath  # noqa: F401
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", False)

CM = 0.45                                     # uF/cm^2
AREA = jnp.array([3.0e-5, 6.0e-5, 2.0e-5, 5.0e-5])   # cm^2 (basal includes spine factor 1.9)
C = CM * AREA * 1e6                           # pF   (uF*1e6)
GL = 0.049 * AREA * 1e6                       # nS   (mS/cm^2 * cm^2 = mS -> *1e6 nS)
EL, ENA, EK, ECA, EE, EI = -70.0, 50.0, -90.0, 120.0, 0.0, -75.0
GNA, GKD, GM = 60.0 * AREA[0] * 1e6, 30.0 * AREA[0] * 1e6, 0.30 * AREA[0] * 1e6
VT, TAU_MAX = -56.2, 608.0
G_AX = {"sb": 25.0, "st": 12.0, "tt": 6.0}    # nS axial couplings soma-basal, soma-trunk, trunk-tuft
TAU_AMPA, TAU_NMDA_R, TAU_NMDA_D, TAU_GABA = 2.0, 2.0, 80.0, 8.0
DCA_THETA, DCA_GBAR, DCA_REF = -55.0, 3.0, 40.0      # mV, nS, ms
DCA_G0, DCA_SIG = 6.0, 1.5                    # apical drive (nS) at threshold, decay width (nS)
DCA_RISE, DCA_DECAY = 2.0, 10.0               # ms
LAP = jnp.array([[G_AX["sb"] + G_AX["st"], -G_AX["sb"], -G_AX["st"], 0.0],
                 [-G_AX["sb"], G_AX["sb"], 0.0, 0.0],
                 [-G_AX["st"], 0.0, G_AX["st"] + G_AX["tt"], -G_AX["tt"]],
                 [0.0, 0.0, -G_AX["tt"], G_AX["tt"]]])


def _rates(v):
    x = v - VT
    am = 0.32 * (13.0 - x) / (jnp.exp((13.0 - x) / 4.0) - 1.0 + 1e-9)
    bm = 0.28 * (x - 40.0) / (jnp.exp((x - 40.0) / 5.0) - 1.0 + 1e-9)
    ah = 0.128 * jnp.exp((17.0 - x) / 18.0)
    bh = 4.0 / (1.0 + jnp.exp((40.0 - x) / 5.0))
    an = 0.032 * (15.0 - x) / (jnp.exp((15.0 - x) / 5.0) - 1.0 + 1e-9)
    bn = 0.5 * jnp.exp((10.0 - x) / 40.0)
    pinf = 1.0 / (1.0 + jnp.exp(-(v + 35.0) / 10.0))
    taup = TAU_MAX / (3.3 * jnp.exp((v + 35.0) / 20.0) + jnp.exp(-(v + 35.0) / 20.0))
    return am, bm, ah, bh, an, bn, pinf, taup


def _rl(x, a, b, dt):
    """Rush-Larsen: exact exponential update of a gate with frozen alpha/beta."""
    tau = 1.0 / (a + b)
    return a * tau + (x - a * tau) * jnp.exp(-dt / tau)


def hines(d, r):
    """Exact elimination for the tree basal-soma-trunk-tuft (no matrix solve)."""
    gsb, gst, gtt = G_AX["sb"], G_AX["st"], G_AX["tt"]
    d2 = d[:, 2] - gtt * gtt / d[:, 3]
    r2 = r[:, 2] + gtt * r[:, 3] / d[:, 3]
    d0 = d[:, 0] - gsb * gsb / d[:, 1] - gst * gst / d2
    r0 = r[:, 0] + gsb * r[:, 1] / d[:, 1] + gst * r2 / d2
    v0 = r0 / d0
    v2 = (r2 + gst * v0) / d2
    v1 = (r[:, 1] + gsb * v0) / d[:, 1]
    v3 = (r[:, 3] + gtt * v2) / d[:, 3]
    return jnp.stack([v0, v1, v2, v3], axis=1)


def init(n):
    v = jnp.full((n, 4), -70.0)
    am, bm, ah, bh, an, bn, pinf, _ = _rates(v[:, 0])
    return dict(v=v, m=am / (am + bm), h=ah / (ah + bh), n=an / (an + bn), p=pinf,
                ga=jnp.zeros((n, 4)), gnr=jnp.zeros((n, 4)), gnd=jnp.zeros((n, 4)), gi=jnp.zeros((n, 4)),
                ca_amp=jnp.zeros(n), ca_r=jnp.zeros(n), ca_d=jnp.zeros(n), ca_ref=jnp.zeros(n),
                vt_prev=jnp.full(n, -70.0), spike_ref=jnp.zeros(n))


def step(s, w_ampa, w_nmda, w_gaba, i_soma, dt):
    """One substep. w_* [n,4]: conductance increments (nS) arriving this substep (spikes x weights)."""
    # synaptic conductances: exponential decay (exact) + increments
    ga = s["ga"] * jnp.exp(-dt / TAU_AMPA) + w_ampa
    gnr = s["gnr"] * jnp.exp(-dt / TAU_NMDA_R) + w_nmda
    gnd = s["gnd"] * jnp.exp(-dt / TAU_NMDA_D) + w_nmda
    gi = s["gi"] * jnp.exp(-dt / TAU_GABA) + w_gaba
    v = s["v"]
    mg = 1.0 / (1.0 + jnp.exp(-0.062 * v) / 3.57)
    gn = jnp.maximum(gnd - gnr, 0.0) * mg
    # dCaAP event: trigger on upward crossing in the trunk, amplitude set by apical drive at onset
    vt = v[:, 2]
    drive = ga[:, 2] + ga[:, 3] + gn[:, 2] + gn[:, 3]
    onset = (s["vt_prev"] < DCA_THETA) & (vt >= DCA_THETA) & (s["ca_ref"] <= 0.0)
    amp = jnp.exp(-jnp.maximum(drive - DCA_G0, 0.0) / DCA_SIG)
    ca_amp = jnp.where(onset, amp, s["ca_amp"])
    ca_r = jnp.where(onset, 0.0, s["ca_r"] * jnp.exp(-dt / DCA_RISE)) + jnp.where(onset, 1.0, 0.0)
    ca_d = jnp.where(onset, 0.0, s["ca_d"] * jnp.exp(-dt / DCA_DECAY)) + jnp.where(onset, 1.0, 0.0)
    kern = jnp.maximum(ca_d - ca_r, 0.0) * 2.6                           # normalised biexponential (peak ~1)
    g_dca = DCA_GBAR * ca_amp * kern
    ca_ref = jnp.where(onset, DCA_REF, jnp.maximum(s["ca_ref"] - dt, 0.0))
    # soma ionic conductances
    vs = v[:, 0]
    g_na = GNA * s["m"] ** 3 * s["h"]
    g_k = GKD * s["n"] ** 4 + GM * s["p"]
    n = v.shape[0]
    g_tot = (GL[None, :] + ga + gn + gi).at[:, 0].add(g_na + g_k).at[:, 2].add(g_dca)
    i_rev = (GL[None, :] * EL + (ga + gn) * EE + gi * EI).at[:, 0].add(g_na * ENA + g_k * EK + i_soma) \
        .at[:, 2].add(g_dca * ECA)
    # backward Euler: (C/dt + G + LAP) v' = C/dt v + I_rev
    d = C[None, :] / dt + g_tot + jnp.diag(LAP)[None, :]
    r = C[None, :] / dt * v + i_rev
    v_new = hines(d, r)
    # gates (Rush-Larsen at the new voltage)
    am, bm, ah, bh, an, bn, pinf, taup = _rates(v_new[:, 0])
    m = _rl(s["m"], am, bm, dt)
    h = _rl(s["h"], ah, bh, dt)
    nn = _rl(s["n"], an, bn, dt)
    p = pinf + (s["p"] - pinf) * jnp.exp(-dt / taup)
    spike = (v_new[:, 0] > -10.0) & (s["spike_ref"] <= 0.0)
    spike_ref = jnp.where(spike, 2.0, jnp.maximum(s["spike_ref"] - dt, 0.0))
    ns = dict(v=v_new, m=m, h=h, n=nn, p=p, ga=ga, gnr=gnr, gnd=gnd, gi=gi, ca_amp=ca_amp, ca_r=ca_r, ca_d=ca_d,
              ca_ref=ca_ref, vt_prev=vt, spike_ref=spike_ref)
    return ns, spike
