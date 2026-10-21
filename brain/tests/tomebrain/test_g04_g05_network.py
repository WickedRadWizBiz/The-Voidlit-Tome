"""Milestone 0 gates on the full calibrated JAX partition (slow: seconds of brain time).

G5 the circuit is alive and balanced: every population in its band at rest and under load, irregular L2/3 firing,
   E:I 1:2-1:4 on the PCs, no dropped spikes.
G4 macroscopic stability with microscopic chaos; the no-inhibition and frozen-quiescent mutants must fail.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from voidlit.tomebrain.calib import calib_rates as CR  # noqa: E402
from voidlit.tomebrain.jaxpart import network as NW  # noqa: E402
from voidlit.tomebrain.jaxpart import params as PR  # noqa: E402



def calibrated_probe(**flags):
    cal = NW.load_calibration()
    assert cal["mu"], "run python -m voidlit.tomebrain.calib.calib_rates first"
    probe = CR.Probe(gains={k: cal["inh_gain"] for k in NW.inhibitory_onto_l23()}, flags=flags or None)
    mu = CR._mu_vector(probe.net, {**{p.name: p.mu for p in probe.net.pops}, **cal["mu"]})
    return probe, mu, cal["sd_scale"]


@pytest.fixture(scope="module")
def conditions():
    probe, mu, sd = calibrated_probe()
    return probe, mu, sd, probe.run(mu, False, sd_scale=sd), probe.run(mu, True, sd_scale=sd)


@pytest.mark.slow
def test_g5_every_population_in_band(conditions):
    *_, rest, busy = conditions
    for name, ok in CR.in_band(rest["rates"]).items():
        assert ok, (name, rest["rates"][name], CR.band(name))
    for name, ok in CR.in_band(busy["rates"]).items():
        assert ok, (name, busy["rates"][name], CR.band(name))


@pytest.mark.slow
def test_g5_balanced_and_irregular(conditions):
    *_, rest, _ = conditions
    assert CR.EI_BAND[0] <= rest["ei"] <= CR.EI_BAND[1]
    assert CR.CV_BAND[0] <= rest["cv"] <= CR.CV_BAND[1]


@pytest.mark.slow
def test_g5_no_dropped_spikes_over_fifty_busy_turns(conditions):
    """Synchronous volleys when every thalamic channel switches at once reach about 200 spikes in one ms; chunked
    delivery must still deliver every one of them."""
    probe, mu, sd, *_ = conditions
    busy = probe.run(mu, True, ms=50 * 240, warm=200, sd_scale=sd)
    assert busy["overflow"] == 0


def l23_rate_trace(probe, mu, sd, pulse=None, ms=1500, bin_ms=50, kick=None, noise_key=0, kick_mv=0.1,
                   record_v=False):
    st = NW.init_state(probe.net, jax.random.PRNGKey(noise_key))
    st["mu"] = jnp.asarray(mu)
    st["sd"] = jnp.asarray(probe.net.point["sd"] * sd, jnp.float32)
    ext = CR.thalamic_drive(probe.net, ms, busy=False)
    l23 = np.concatenate([np.arange(probe.net.idx(f"{c}.L23").start, probe.net.idx(f"{c}.L23").stop)
                          for c in PR.COLUMNS])
    probe.l23 = l23
    exc = np.zeros(probe.net.n, bool)
    for p in probe.net.pops:
        if p.sign == "E" and p.region in PR.COLUMNS:
            exc[probe.net.idx(p.name)] = True
    raster, volts = [], []
    for t in range(0, ms, probe.W):
        e = ext[t:t + probe.W].copy()
        if pulse and pulse[0] <= t < pulse[1]:
            e[:, exc] += 0.6 * mu[exc]
        if kick is not None and t == kick:
            v = np.asarray(st["V"])[l23]
            # a PC at the median potential: not about to fire, so a big kick inserts exactly one extra spike
            probe.kicked = int(l23[int(np.argsort(v)[len(v) // 2])])
            st["V"] = st["V"].at[probe.kicked].add(kick_mv)
        st, (s, _) = probe.window(st, jnp.asarray(e), 1.0)
        raster.append(np.asarray(s))
        if record_v:
            volts.append(np.asarray(st["V"]))
    R = np.concatenate(raster)
    assert np.isfinite(np.asarray(st["V"])).all()
    rate = R[:, l23].reshape(ms // bin_ms, bin_ms, -1).mean((1, 2)) * 1000.0
    if record_v:
        return rate, R, np.array(volts)
    return rate, R


@pytest.mark.slow
def test_g4a_a_drive_pulse_settles():
    probe, mu, sd = calibrated_probe()
    base, _ = l23_rate_trace(probe, mu, sd)
    hit, _ = l23_rate_trace(probe, mu, sd, pulse=(500, 700))
    baseline = base[4:].mean()
    after = hit[(700 + 500) // 50:].mean()     # from 500 ms after the pulse ends
    assert hit[10:14].mean() > 1.5 * baseline   # the pulse really drove the cortex
    assert abs(after - baseline) <= 0.25 * baseline + 0.2


@pytest.mark.slow
def test_g4a_mutant_without_inhibition_runs_away():
    probe, mu, sd = calibrated_probe()
    base, _ = l23_rate_trace(probe, mu, sd, ms=600)
    probe_x, _, _ = calibrated_probe(inhibition=False)
    wild, _ = l23_rate_trace(probe_x, mu, sd, ms=600)
    assert wild[4:].mean() > 3 * base[4:].mean()


def _diverging_cells(A, B, kicked, t0, t1):
    diff = (A[t0:t1] != B[t0:t1]).any(0)
    diff[kicked] = False
    return int(diff.sum())


@pytest.mark.slow
def test_g4b_stable_chaos_with_stable_rates():
    """Inhibition-dominated spiking networks are "stably chaotic" (Zillmer, Livi, Politi & Torcini 2006): an
    infinitesimal (0.1 mV) kick decays, but one extra spike spreads through the raster. Population rates stay
    within 10% either way. Kicks land on window boundaries (multiples of W)."""
    probe, mu, sd = calibrated_probe()
    _, B, VB = l23_rate_trace(probe, mu, sd, ms=900, record_v=True)
    spread = 0
    for kick in (200, 300, 400, 500, 600):
        assert kick % probe.W == 0
        _, A = l23_rate_trace(probe, mu, sd, ms=900, kick=kick, kick_mv=40.0)
        assert A[kick:kick + probe.W, probe.kicked].sum() > B[kick:kick + probe.W, probe.kicked].sum()
        kicked = probe.kicked
        early, late = _diverging_cells(A, B, kicked, kick, kick + 20), _diverging_cells(A, B, kicked, kick, kick + 200)
        spread += late > max(early, 0) and late > 0
        l23 = probe.l23
        assert abs(A[:, l23].sum() - B[:, l23].sum()) <= 0.10 * B[:, l23].sum()
    assert spread >= 4
    _, _, VA = l23_rate_trace(probe, mu, sd, ms=700, kick=300, kick_mv=0.1, record_v=True)
    dv = np.abs(VA - VB[:VA.shape[0]]).max(1)
    assert dv[(300 + 190) // 4] < dv[(300 + 8) // 4]          # the infinitesimal kick decays


@pytest.mark.slow
def test_g4b_mutant_frozen_quiescent_network_does_not_spread():
    probe, mu, sd = calibrated_probe(noise=False)
    quiet = np.zeros_like(mu)
    _, A = l23_rate_trace(probe, quiet, sd, ms=400, kick=100, kick_mv=40.0)
    _, B = l23_rate_trace(probe, quiet, sd, ms=400)
    assert _diverging_cells(A, B, probe.kicked, 100, 300) == 0


@pytest.mark.parametrize("volley", [5, 300])
def test_g5_delivery_is_exact_for_any_volley(volley):
    """Every out-edge of every spiking cell lands, whether the volley fits one chunk or needs several."""
    pops = [PR.Pop("X", volley, "RS", "E", region="S", mu=0.0), PR.Pop("Y", 1, "RS", "E", region="S", mu=0.0)]
    projs = [PR.Projection("X", "Y", volley, (PR.AMPA,), PR.SOMA, 0.01, (1, 1))]
    net = NW.build(pops=pops, projs=projs, s_max=64)
    win = NW.make_window(net, flags=dict(noise=False, plasticity=False, glia=False, energy=False))
    st = NW.init_state(net, jax.random.PRNGKey(0))
    st["mu"] = jnp.zeros(net.n)
    ext = np.zeros((2, net.n), np.float32)
    ext[0, :volley] = 20000.0
    st, (s, _) = win(st, jnp.asarray(ext), 1.0)
    assert int(np.asarray(s)[0, :volley].sum()) == volley
    slot = net.tables["slot"][0, 0]
    expected = net.tables["w"][:volley, 0].sum() * np.exp(-1.0 / PR.TAU_REC[PR.AMPA])
    assert abs(float(st["G"][slot]) - expected) < 1e-4 and int(st["overflow"]) == 0
