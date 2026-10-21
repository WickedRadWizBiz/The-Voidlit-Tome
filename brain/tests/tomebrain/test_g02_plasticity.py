"""Gate G2: the network's own plasticity rules behave like the papers they come from.

G2a Graupner & Brunel 2012 (DP set): frequency-dependent, bistable, calcium-based plasticity.
G2b Vogels et al. 2011: inhibitory STDP balances a spiking L2/3 cell to its target rate.
G2c Tsodyks-Markram: the release sequence of a 20-Hz train matches the analytic recursion.
"""

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from voidlit.tomebrain.jaxpart import network as NW  # noqa: E402
from voidlit.tomebrain.jaxpart import params as PR  # noqa: E402


# ---------------------------------------------------------------- G2a Graupner-Brunel
def pairing(freq_hz, dt_pair, n_pairs=60, rho0=0.5):
    """Pairs at ``freq_hz``; dt_pair = t_post - t_pre (ms); the network's calcium model at 1 ms resolution
    (presynaptic calcium arrives GB_DELAY ms late, C_pre 1, C_post 2, tau_Ca 20 ms). Returns the final rho."""
    period = 1000.0 / freq_hz
    T = int(n_pairs * period) + 200
    pre = np.zeros(T)
    post = np.zeros(T)
    for i in range(n_pairs):
        t = int(round(i * period)) + 50
        pre[t] = 1
        post[t + dt_pair] = 1
    k = math.exp(-1.0 / NW.GB["tauCa"])
    rho, cpre, cpost = rho0, 0.0, 0.0
    for t in range(T):
        cpre = cpre * k + NW.GB["Cpre"] * (pre[t - NW.GB_DELAY] if t >= NW.GB_DELAY else 0.0)
        cpost = cpost * k + NW.GB["Cpost"] * post[t]
        rho = min(1.0, max(0.0, rho + float(NW.gb_drho(rho, cpre + cpost))))
    return rho


def test_g2a_frequency_dependent_plasticity():
    assert pairing(1, -10) < 0.5        # post-before-pre at 1 Hz: depression
    assert pairing(40, -10) > 0.5       # the same pairs at 40 Hz: potentiation (Sjostrom 2001)
    assert pairing(1, +10) > 0.5        # pre-before-post: potentiation


def test_g2a_bistable_drift_away_from_the_unstable_point():
    lo, hi = 0.45, 0.55
    for _ in range(600):                # 10 minutes without calcium, 1-s steps
        lo = min(1.0, max(0.0, lo + 1000.0 * float(NW.gb_drho(lo, 0.0))))
        hi = min(1.0, max(0.0, hi + 1000.0 * float(NW.gb_drho(hi, 0.0))))
    assert lo < 0.45 and hi > 0.55


def test_g2a_mutant_spec_threshold_fails(monkeypatch):
    """The submitted spec's theta_p = 1.6 turns pre-before-post pairing into depression."""
    monkeypatch.setitem(NW.GB, "thp", 1.6)
    assert not (pairing(1, -10) < 0.5 and pairing(40, -10) > 0.5 and pairing(1, +10) > 0.5)


# ---------------------------------------------------------------- G2b Vogels
def vogels_rate(efficacy0, seconds=180, eta=None, monkeypatch=None):
    """One spiking human L2/3 cell under 40 PV synapses (3 nS base) whose efficacy starts at ``efficacy0``."""
    if eta is not None:
        monkeypatch.setitem(NW.VOGELS, "eta", eta)
    pops = [PR.Pop("S.L23", 1, "L23", "E", region="S", mu=520.0, cortical=False),
            PR.Pop("S.PV23", 40, "PV", "I", region="S", mu=170.0, cortical=False)]
    projs = [PR.Projection("S.PV23", "S.L23", 40, (PR.GABA_A,), PR.SOMA, 3.0, plastic="vogels")]
    net = NW.build(seed=1, pops=pops, projs=projs)
    win = NW.make_window(net, flags=dict(glia=False, energy=False))
    st = NW.init_state(net, jax.random.PRNGKey(1))
    st["wv"] = jnp.full_like(st["wv"], efficacy0)
    ext = jnp.zeros((1000, net.n), jnp.float32)
    counts = []
    for _ in range(seconds):
        st, (s, _) = win(st, ext, 1.0)
        counts.append(float(np.asarray(s)[:, 0].sum()))
    return float(np.mean(counts[-20:])), float(np.mean(counts[:3]))


@pytest.mark.parametrize("efficacy0", [0.05, 1.5])
def test_g2b_istdp_balances_to_the_target_rate(efficacy0):
    """From weak and from strong initial inhibition the cell settles at rho0 (2 Hz) within 30%."""
    target = NW.VOGELS["rho0"] * 1000.0
    final, start = vogels_rate(efficacy0)
    assert abs(start - target) > 0.3 * target          # it started out of balance
    assert abs(final - target) <= 0.3 * target


def test_g2b_mutant_frozen_rule_stays_far(monkeypatch):
    target = NW.VOGELS["rho0"] * 1000.0
    final, _ = vogels_rate(0.05, seconds=30, eta=0.0, monkeypatch=monkeypatch)
    assert abs(final - target) > 0.3 * target


# ---------------------------------------------------------------- G2c Tsodyks-Markram
def test_g2c_release_sequence_matches_the_analytic_recursion():
    x = jnp.ones((2, 1), jnp.float32)
    u = jnp.zeros((2, 1), jnp.float32)
    released = []
    for t in range(500):
        spike = jnp.asarray([1.0 if t % 50 == 0 else 0.0])
        x, u, rel = NW.tm_step(x, u, spike)
        if t % 50 == 0:
            released.append(np.asarray(rel[:, 0]) * np.asarray(NW.TM["U"]))
    for c in range(2):
        U, td, tf = NW.TM["U"][c], NW.TM["tau_d"][c], NW.TM["tau_f"][c]
        xa, ua = 1.0, 0.0
        for n, got in enumerate(released):
            if n:
                xa = 1 - (1 - xa) * math.exp(-50 / td)
                ua = ua * math.exp(-50 / tf)
            ua = ua + U * (1 - ua)
            r = ua * xa
            xa -= r
            assert abs(got[c] - r) < 1e-5
    dep = [r[0] for r in released]
    fac = [r[1] for r in released]
    assert dep[-1] < dep[0] and fac[-1] > fac[0]       # depressing vs facilitating classes
