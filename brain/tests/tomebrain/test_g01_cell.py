"""Gate G1: the human L2/3 cell is stable at any dt, converges, and computes XOR in one dendrite (Gidon 2020)."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from voidlit.tomebrain.jaxpart import network as NW  # noqa: E402
from voidlit.tomebrain.jaxpart import params as PR  # noqa: E402

QUIET = dict(noise=False, plasticity=False, glia=False, energy=False)
PRODUCTION_DT, REFERENCE_DT = 0.25, 0.025


def one_cell(dt, kind="L23", **dcaap):
    pop = PR.Pop("S.L23" if kind == "L23" else "X", 1, kind, "E", region="S", mu=0.0, cortical=False)
    net = NW.build(pops=[pop], projs=[])
    return net, NW.make_window(net, dt=dt, flags=QUIET, dcaap=dcaap or None, record=True)


def run(net, win, *, tuft_nS=0.0, soma_pA=0.0, ms=120, on=(20, 70)):
    st = NW.init_state(net, jax.random.PRNGKey(0))
    st["V"] = jnp.asarray(net.point["EL"])
    st["mu"] = jnp.zeros(net.n)
    g = np.zeros((ms, max(net.n3, 0)), np.float32)
    I = np.zeros((ms, net.n), np.float32)
    if net.n3:
        g[on[0]:on[1], 0] = tuft_nS
    I[on[0]:on[1], 0] = soma_pA
    st, (s, _, rec) = win(st, jnp.asarray(I), 1.0, jnp.asarray(g) if net.n3 else None)
    return np.asarray(s)[:, 0], np.asarray(rec), st


@pytest.mark.parametrize("dt", [0.025, 0.05, 0.1, 0.25, 0.5, 1.0])
def test_g1a_integrator_is_stable_and_physiological(dt):
    net, win = one_cell(dt)
    spikes, rec, st = run(net, win, tuft_nS=15.0, soma_pA=200.0)
    assert np.isfinite(rec).all() and bool(jnp.isfinite(st["V"]).all())
    assert spikes.sum() > 0
    assert rec[:, 0].max() <= PR.SPIKE_V          # the soma is reset on detection
    assert -30.0 <= rec[:, 1].max() <= 10.0       # the tuft's dCaAP peak is physiological


def spec_forward_euler_peak(dt, steps=400):
    """The submitted spec's L2/3 cell (mS/cm2 conductances on fractional areas), integrated with its forward Euler.
    It is the mutant: explicit Euler on these stiff compartments diverges well below dt = 0.05 ms."""
    E_L, E_Na, E_K, E_Ca = -70.0, 50.0, -82.0, 120.0
    g_L, gNa, gK, gM, g_dca, Cm, F, p = 0.049, 98.0, 40.0, 0.042, 2.5, 0.45, 1.9, 0.12
    s_d = np.full(8, 0.11)
    G_ax = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6, 0.35])
    V = np.full(9, -70.0)
    h, n, mM = 0.8, 0.1, 0.05
    I_ext = np.zeros(9)
    I_ext[8] = 0.75
    peak = -70.0
    with np.errstate(all="ignore"):
        for _ in range(steps):
            Vs, Vd = V[0], V[1:]
            m_inf = 1 / (1 + np.exp(-(Vs + 38) / 6))
            I_ion = g_L * (Vs - E_L) + gNa * m_inf ** 3 * h * (Vs - E_Na) + gK * n ** 4 * (Vs - E_K) + gM * mM * (Vs - E_K)
            K = np.zeros(8)
            K[6:] = np.where(I_ext[7:] >= 0.75, 1.0, 0.0) * np.exp(-((I_ext[7:] - 0.75) ** 2) / (2 * 0.18 ** 2))
            I_dca = g_dca * K * (E_Ca - Vd)
            I_nmda = 0.45 / (1 + np.exp(-0.08 * (Vd + 35))) * np.clip(I_ext[1:], 0, 2) * (0 - Vd)
            dVs = (-p * I_ion + np.sum(G_ax * (Vd - Vs)) + I_ext[0]) / (p * Cm)
            dVd = (-s_d * F * g_L * (Vd - E_L) + I_dca + I_nmda + G_ax * (Vs - Vd) + I_ext[1:]) / (s_d * F * Cm)
            V = np.concatenate([[Vs + dt * dVs], Vd + dt * dVd])
            h += dt * (1 / (1 + np.exp((Vs + 58) / 7)) - h) / 2.0
            n += dt * (1 / (1 + np.exp(-(Vs + 28) / 10)) - n) / 2.5
            mM += dt * (1 / (1 + np.exp(-(Vs + 35) / 8)) - mM) / 40.0
            peak = max(peak, float(np.nanmax(V)) if np.isfinite(V).all() else np.inf)
    return peak


def test_g1a_mutant_forward_euler_spec_cell_blows_up():
    assert not np.isfinite(spec_forward_euler_peak(0.05)) or spec_forward_euler_peak(0.05) > 1e3


@pytest.mark.slow
@pytest.mark.parametrize("current", [400.0, 550.0, 800.0])
def test_g1b_production_dt_converges_to_reference(current):
    """Spike count within 15% and first-spike latency within max(1 ms, 5%) of the dt = 0.025 ms reference."""
    rates, first = [], []
    for dt in (PRODUCTION_DT, REFERENCE_DT):
        net, win = one_cell(dt)
        spikes, _, _ = run(net, win, soma_pA=current, ms=520, on=(20, 520))
        t = np.flatnonzero(spikes)
        rates.append(len(t))
        first.append(t[0] if len(t) else None)
    assert rates[1] > 0
    assert abs(rates[0] - rates[1]) <= max(1, 0.15 * rates[1])
    latency = first[1] - 20
    assert first[0] is not None and abs(first[0] - first[1]) <= max(1, 0.05 * latency)


def xor(dt, amp, **dcaap):
    net, win = one_cell(dt, **dcaap)
    return [int(run(net, win, tuft_nS=x + y)[0].sum()) for x, y in ((0, 0), (amp, 0), (0, amp), (amp, amp))]


@pytest.mark.parametrize("dt", [PRODUCTION_DT, pytest.param(REFERENCE_DT, marks=pytest.mark.slow)])
@pytest.mark.parametrize("scale", [0.8, 1.0, 1.2])
def test_g1c_one_dendrite_computes_xor(dt, scale):
    """One pathway (about u_th) evokes a dCaAP and somatic spikes; both together inactivate it."""
    s00, s10, s01, s11 = xor(dt, scale * PR.DCAAP["u_th"])
    assert s00 == 0 and s10 > 0 and s01 > 0 and s11 == 0


def test_g1c_mutants_fail():
    monotone = xor(PRODUCTION_DT, PR.DCAAP["u_th"], tau_K=1e9)   # h stays 1: no drive-dependent inactivation
    assert monotone[3] > 0
    no_dcaap = xor(PRODUCTION_DT, PR.DCAAP["u_th"], g_d=0.0)
    assert no_dcaap[1] == 0
