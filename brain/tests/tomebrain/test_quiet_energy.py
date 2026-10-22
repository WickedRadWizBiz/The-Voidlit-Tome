"""The quiet interval solves the window's whole ATP equation (design §5.6, §6.7), synaptic cost included:

    da/dt = k_sup g CBF (1.05 - a) - k_rest - k_syn g_exc - k_spike r

Each cell's excitatory conductance starts at its end-of-turn value and relaxes with the activity profile, so a cell
still bombarded at the end of a turn pays for it, and one whose inputs fall silent stops paying. It relaxes by the
population-rate drive's fall from the turn to rest, never upward, and rest is what the network does at the current
Lumen.
"""

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from voidlit.tomebrain.jaxpart import network as NW  # noqa: E402
from voidlit.tomebrain.jaxpart import params as PR  # noqa: E402

G_EXC = 5.0      # nS of AMPA + NMDA on the target cell: about what a fed P/I cell carries at the end of a turn


@pytest.fixture(scope="module")
def tiny():
    pops = [PR.Pop("A", 2, "RS", "E", region="nuclei"), PR.Pop("B", 2, "RS", "E", region="nuclei")]
    net = NW.build(pops=pops, projs=[PR.Projection("A", "B", 1, PR.AN, PR.SOMA, 1.0)])
    return net, NW.make_quiet(net)


def quiet_atp(tiny, rate_turn: float, rate_rest: float):
    net, quiet = tiny
    st = NW.init_state(net, jax.random.PRNGKey(0))
    b = net.idx("B").start
    st["G"] = st["G"].at[net.slot(b, PR.SOMA, PR.AMPA)].set(1.0).at[net.slot(b, PR.SOMA, PR.NMDA)].set(G_EXC - 1.0)
    out = quiet(st, jnp.asarray(1.0, jnp.float32), jnp.full(net.n, rate_turn, jnp.float32),
                jnp.full(net.n, rate_rest, jnp.float32))
    return [float(x) for x in out["atp"][b:b + 2]]


def test_quiet_charges_the_synaptic_cost_of_a_turn_that_ends_busy(tiny):
    E = NW.ENERGY
    loaded, unloaded = quiet_atp(tiny, 0.0, 0.0)            # no activity change: the conductance stays as it ended
    assert unloaded == pytest.approx(1.05 - E["k_rest"] / E["k_sup"], abs=2e-3)
    assert loaded == pytest.approx(1.05 - (E["k_rest"] + E["k_syn"] * G_EXC) / E["k_sup"], abs=2e-3)


def test_quiet_conductance_relaxes_with_the_activity(tiny):
    """The turn ends busy (10 Hz) and the network falls silent at rest: the inputs, and their cost, fade with tau_post
    long before the 20 s are over."""
    loaded, unloaded = quiet_atp(tiny, 10.0, 0.0)
    assert loaded == pytest.approx(unloaded, abs=2e-3)


@pytest.fixture(scope="module")
def fan():
    """Four A cells, each B cell driven by one of them."""
    pops = [PR.Pop("A", 4, "RS", "E", region="nuclei"), PR.Pop("B", 2, "RS", "E", region="nuclei")]
    net = NW.build(pops=pops, projs=[PR.Projection("A", "B", 1, PR.AN, PR.SOMA, 1.0)])
    return net, NW.make_quiet(net)


def fan_atp(fan, a_turn, a_rest) -> list[float]:
    """ATP of the B cells after the quiet; B's partners fire ``a_turn`` (Hz per A cell) in the turn and ``a_rest`` at
    rest, B itself is silent (no spike cost)."""
    net, quiet = fan
    st = NW.init_state(net, jax.random.PRNGKey(0))
    b = net.idx("B").start
    for c in (b, b + 1):
        st["G"] = st["G"].at[net.slot(c, PR.SOMA, PR.AMPA)].set(1.0).at[net.slot(c, PR.SOMA, PR.NMDA)].set(G_EXC - 1.0)
    rate = lambda a: jnp.asarray(list(a) + [0.0, 0.0], jnp.float32)  # noqa: E731
    out = quiet(st, jnp.asarray(1.0, jnp.float32), rate(a_turn), rate(a_rest))
    return [float(x) for x in out["atp"][b:b + 2]]


def test_quiet_rest_is_never_busier_than_the_turn(fan):
    """Inputs quieter in the turn than at rest (a short turn's shot noise, or a starving cortex against fed resting
    rates) once scaled the end-of-turn conductance up to 35-fold for 20 s, and ATP collapsed. Rest is the turn
    without its input: the conductance can only fall."""
    E = NW.ENERGY
    for atp in fan_atp(fan, [0.5] * 4, [10.0] * 4):
        assert atp == pytest.approx(1.05 - (E["k_rest"] + E["k_syn"] * G_EXC) / E["k_sup"], abs=2e-3)


def test_quiet_drive_is_read_from_population_rates(fan):
    """One partner's lucky burst in a short turn (14 Hz while the other three fired 4 Hz, against a resting 6 Hz)
    raises the population's drive 8%: whichever A cell drives it, a B cell rests at 92% of its end-of-turn
    conductance (read from its own partner, the scale was 0.43 or 1.5)."""
    E = NW.ENERGY
    expected = 1.05 - (E["k_rest"] + E["k_syn"] * G_EXC * 24.0 / 26.0) / E["k_sup"]
    atp = fan_atp(fan, [14.0, 4.0, 4.0, 4.0], [6.0] * 4)
    assert atp == pytest.approx([expected] * 2, abs=2e-3)


def test_a_starving_brain_rests_starving():
    """The quiet relaxes toward what the network does at rest at the current Lumen (G3b's protocol, settled at each
    level), not toward fed rest: at Lumen 0 the orexin cells rest many times faster, the cortex at a third and the VTA
    nearly silent. Fed, it is the G5 calibration; in between, interpolated by glucose."""
    import tomllib
    from types import SimpleNamespace

    from voidlit.tomebrain.brain import TomeBrain

    net = NW.build()
    table = TomeBrain._rest_rates(net)
    rest = lambda lumen, pop: float(table[lumen, net.idx(pop)].mean())  # noqa: E731
    assert rest(0, "ORX") >= 10 * rest(6, "ORX") and rest(0, "VTA") <= 0.3 * rest(6, "VTA")
    assert rest(0, "P.L23") <= 0.5 * rest(6, "P.L23")
    with open(NW.CALIBRATION_FILE, "rb") as fh:
        assert rest(6, "P.L23") == pytest.approx(tomllib.load(fh)["rates_rest"]["P.L23"], rel=1e-6)
    at = lambda glucose: TomeBrain.rest_rates(SimpleNamespace(rest_table=table), glucose)  # noqa: E731
    assert at(1.0) == pytest.approx(table[6]) and at(0.15) == pytest.approx(table[0])
    assert at(0.15 + 0.85 * 2.5 / 6) == pytest.approx((table[2] + table[3]) / 2, rel=1e-5)


def test_the_brain_rests_at_its_current_lumen():
    """The brain's quiet relaxes toward its table's resting rates at the current Lumen, and the striatum's tonic
    dopamine follows the level the VTA's resting rate sustains. (A brain that passed the fed row at every Lumen passed
    every other test, G3b and G23 included, while it rested a starving network fed.)"""
    import numpy as np

    from voidlit.tomebrain.brain import TomeBrain

    brain = TomeBrain(0)
    seen, quiet = [], brain.quiet

    def spy(st, glucose, rate_turn, rate_rest):
        seen.append(np.asarray(rate_rest))
        return quiet(st, glucose, rate_turn, rate_rest)
    brain.quiet = spy
    for lumen in (6, 0, 3):
        brain.set_input(glucose=0.15 + 0.85 * lumen / 6)
        brain._end_turn()
        assert np.array_equal(seen[-1], brain.rest_table[lumen]), lumen
        assert brain.sj.da_tonic == pytest.approx(NW.nm_level(brain.net, brain.rest_table[lumen])[0], rel=0.15), lumen


def test_recalibration_keeps_the_resting_rates_by_lumen(tmp_path):
    """A new G5 calibration replaces the file's rates and keeps its resting rates by Lumen (it dropped them, and the
    brain fell back to resting fed at every Lumen without a word)."""
    import tomllib

    from voidlit.tomebrain.calib import calib_rates as CR

    path = tmp_path / "calibrated.toml"
    path.write_text(NW.CALIBRATION_FILE.read_text())
    with open(path, "rb") as fh:
        before = tomllib.load(fh)
    rates = {p: 2.0 for p in before["rates_rest"]}
    result = {"score": 0, "inh_gain": 1.0, "sd_scale": 1.0, "mu": before["mu"],
              "rest": {"rates": rates, "ei": 3.0, "cv": 1.0}, "busy": {"rates": rates, "peak": 10, "overflow": 0}}
    for _ in range(2):
        CR.write(result, path)
        with open(path, "rb") as fh:
            after = tomllib.load(fh)
        assert after["rates_rest"] == rates and after[CR.REST_BY_LUMEN] == before[CR.REST_BY_LUMEN]
        assert path.read_text().count(CR.NOTE_BY_LUMEN) == 1
