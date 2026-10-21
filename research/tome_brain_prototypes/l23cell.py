"""Prototype: human L2/3 pyramidal cell (9 compartments), staggered Rush-Larsen gates + linearly implicit
(backward-Euler) voltage solved by Hines elimination on the real tree (6 basal + trunk -> tuft).
Somatic kinetics: Pospischil et al. 2008 regular-spiking (Traub-Miles Na/K + slow M current).
dCaAP: Gidon et al. 2020 phenomenological event on the apical tuft, amplitude decaying with input drive."""
import jaxpath  # noqa
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", False)

NC = 9
area = jnp.array([0.12] + [0.11] * 8)
spine = jnp.array([1.0] + [1.9] * 8)
Cm = 0.45
C = Cm * area * spine                  # uF / cm2 of total cell area
gL, EL = 0.026, -70.0                  # Rm ~ 38 kOhm cm2 -> tau_m ~ 17 ms
G_basal, G_trunk, G_tuft = 0.8, 0.6, 0.35   # axial, mS/cm2 (total-area units)
gNa, gK, gM, ENa, EK, VT, tau_max = 50.0, 5.0, 0.07, 50.0, -90.0, -56.2, 608.0
E_AMPA, E_GABA, E_Ca = 0.0, -75.0, 120.0

def rates(V):
    x = V - VT
    def vtrap(a, b):  # a / (exp(a/b) - 1) with removable singularity
        return jnp.where(jnp.abs(a) < 1e-6, b, a / jnp.expm1(a / b))
    am = 0.32 * vtrap(-(x - 13.0), 4.0)
    bm = 0.28 * vtrap((x - 40.0), 5.0)
    ah = 0.128 * jnp.exp(-(x - 17.0) / 18.0)
    bh = 4.0 / (1.0 + jnp.exp(-(x - 40.0) / 5.0))
    an = 0.032 * vtrap(-(x - 15.0), 5.0)
    bn = 0.5 * jnp.exp(-(x - 10.0) / 40.0)
    p_inf = 1.0 / (1.0 + jnp.exp(-(V + 35.0) / 10.0))
    tau_p = tau_max / (3.3 * jnp.exp((V + 35.0) / 20.0) + jnp.exp(-(V + 35.0) / 20.0))
    return am, bm, ah, bh, an, bn, p_inf, tau_p

def rl(x, a, b, dt):  # Rush-Larsen exponential Euler for a gate
    tau = 1.0 / (a + b); xinf = a * tau
    return xinf + (x - xinf) * jnp.exp(-dt / tau)

def init(n):
    V = jnp.full((n, NC), EL)
    am, bm, ah, bh, an, bn, pinf, _ = rates(V[:, 0])
    return dict(V=V, m=am / (am + bm), h=ah / (ah + bh), n=an / (an + bn), p=pinf,
                s=jnp.zeros(n), a=jnp.zeros(n), r=jnp.zeros(n), armed=jnp.ones(n))

def solve(Vn, gtot, Isrc, dt):
    """Backward Euler on the tree: (C/dt + gtot + L) V = C/dt Vn + Isrc. Hines elimination, O(NC)."""
    b = C / dt * Vn + Isrc
    d = C / dt + gtot
    d_b = d[:, 1:7] + G_basal
    d8 = d[:, 8] + G_tuft
    d7 = d[:, 7] + G_trunk + G_tuft - G_tuft ** 2 / d8
    b7 = b[:, 7] + G_tuft * b[:, 8] / d8
    d0 = d[:, 0] + 6 * G_basal + G_trunk - jnp.sum(G_basal ** 2 / d_b, 1) - G_trunk ** 2 / d7
    b0 = b[:, 0] + jnp.sum(G_basal * b[:, 1:7] / d_b, 1) + G_trunk * b7 / d7
    V0 = b0 / d0
    Vb = (b[:, 1:7] + G_basal * V0[:, None]) / d_b
    V7 = (b7 + G_trunk * V0) / d7
    V8 = (b[:, 8] + G_tuft * V7) / d8
    return jnp.concatenate([V0[:, None], Vb, V7[:, None], V8[:, None]], 1)

def step(st, g_exc, g_inh, I_ap, dt, dcaap=True, g_dcaap=1.2, I_th=1.0, tau_A=0.35, tau_s=4.0, refr=15.0,
         monotone=False):
    """g_exc, g_inh: [n, NC] synaptic conductances (mS/cm2 total area). I_ap: [n] apical synaptic drive for the
    dCaAP trigger (the input amplitude, as in Gidon's current-step protocol)."""
    V = st["V"]; Vs = V[:, 0]
    am, bm, ah, bh, an, bn, pinf, taup = rates(Vs)
    m = rl(st["m"], am, bm, dt); h = rl(st["h"], ah, bh, dt); n = rl(st["n"], an, bn, dt)
    p = pinf + (st["p"] - pinf) * jnp.exp(-dt / taup)
    gna = gNa * m ** 3 * h; gk = gK * n ** 4; gm = gM * p
    # dCaAP event state (tuft, compartment 8)
    trig = (I_ap >= I_th) & (st["armed"] > 0) & (st["r"] <= 0) if dcaap else jnp.zeros_like(I_ap, bool)
    amp = jnp.where(monotone, 1.0, jnp.exp(-jnp.maximum(I_ap - I_th, 0.0) / tau_A))
    a = jnp.where(trig, amp, st["a"])
    s = jnp.where(trig, 1.0, st["s"] * jnp.exp(-dt / tau_s))
    r = jnp.where(trig, refr, jnp.maximum(st["r"] - dt, 0.0))
    armed = jnp.where(trig, 0.0, jnp.where(I_ap < 0.8 * I_th, 1.0, st["armed"]))  # re-arm after input falls
    g_ca = g_dcaap * a * s
    memb = area * spine
    gtot = memb * gL + g_exc + g_inh
    gtot = gtot.at[:, 0].add(area[0] * (gna + gk + gm))
    gtot = gtot.at[:, 8].add(g_ca)
    Isrc = memb * gL * EL + g_exc * E_AMPA + g_inh * E_GABA
    Isrc = Isrc.at[:, 0].add(area[0] * (gna * ENa + (gk + gm) * EK))
    Isrc = Isrc.at[:, 8].add(g_ca * E_Ca)
    Vn = solve(V, gtot, Isrc, dt)
    return dict(V=Vn, m=m, h=h, n=n, p=p, s=s, a=a, r=r, armed=armed)
