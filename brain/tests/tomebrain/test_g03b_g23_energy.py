"""G3b: the survival need comes from metabolism. G23: desperation is physiology, not a script.

As the Lumen dies (glucose 1.0 -> 0.15): ATP falls, glucose-inhibited orexin cells rise at least tenfold, the dopamine
cells slow, and the cortex weakens to 40-60% of its baseline rather than going silent. Hunger reaches the schemer's
corruption and bend cells only through orexin -> PVT -> their tufts, so their bursting rises as the Lumen falls,
and a lesion of that path removes most of the rise. The whole brain's own Mind reads the dying light as a falling
survival need and a rising desperation; with glucose held at 1 while the insula still sees the light die, neither
moves.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")

import jax.numpy as jnp  # noqa: E402

from voidlit.cortex.teacher import sample_states  # noqa: E402
from voidlit.tomebrain.brain import TomeBrain  # noqa: E402
from voidlit.tomebrain.calib import calib_rates as CR  # noqa: E402
from voidlit.tomebrain.jaxpart import network as NW  # noqa: E402
from voidlit.tomebrain.jaxpart import params as PR  # noqa: E402
from voidlit.tomebrain.readout import atp_pi  # noqa: E402

pytestmark = pytest.mark.slow


def lumen_glucose(lumen: int) -> float:
    return 0.15 + 0.85 * lumen / 6


def live(glucose: float, gains: dict | None = None, turns: int = 3, flags: dict | None = None, measure_ms: int = 1000):
    """Three turns (1 s each, then a quiet interval) to settle energy, then ``measure_ms`` measured."""
    net = NW.build(gains=gains) if gains else NW.build()
    win, quiet = NW.make_window(net, flags=flags), NW.make_quiet(net)
    st = NW.init_state(net, jax.random.PRNGKey(0))
    ext = CR.thalamic_drive(net, max(1000, measure_ms), busy=False)
    for k in range(turns + 1):
        raster = []
        for t in range(0, 1000 if k < turns else measure_ms, 4):
            st, (s, _) = win(st, jnp.asarray(ext[t:t + 4]), glucose)
            raster.append(np.asarray(s))
        R = np.concatenate(raster)
        if k < turns:
            rate = R.sum(0).astype(np.float32)
            st = quiet(st, jnp.asarray(glucose, jnp.float32), jnp.asarray(rate), jnp.asarray(rate))
    rates = {p.name: float(R[:, net.idx(p.name)].mean() * 1000) for p in net.pops}
    puppet = R[:, net.idx("P.L5ET")][:, 24:72]
    bursts = 0
    for col in puppet.T:          # a burst: two or more spikes within 15 ms
        t = np.flatnonzero(col)
        bursts += int(np.sum(np.diff(t) <= 15)) if len(t) > 1 else 0
    rates = {p.name: float(R[:, net.idx(p.name)].mean() * 1000) for p in net.pops}
    return dict(atp=atp_pi(net, np.asarray(st["atp"])), rates=rates,
                bursts=bursts / puppet.shape[1] * 1000.0 / R.shape[0],          # bursts per cell per second
                l23=np.mean([rates[f"{c}.L23"] for c in PR.COLUMNS]))


@pytest.fixture(scope="module")
def fed_and_starved():
    return live(lumen_glucose(6)), live(lumen_glucose(0))


def test_g3b_energy_drives_the_survival_need(fed_and_starved):
    fed, starved = fed_and_starved
    assert starved["atp"] < fed["atp"] - 0.2
    assert starved["rates"]["ORX"] >= 10 * fed["rates"]["ORX"]
    assert starved["rates"]["VTA"] < fed["rates"]["VTA"]
    assert 0.4 <= starved["l23"] / fed["l23"] <= 0.6


def test_g3b_mutant_without_metabolism_the_cortex_stays_out_of_the_band():
    """With the energy model off (no ATP, no K_ATP) a dying Lumen still dims the cortex a little: orexin arousal
    (ORX -> LC, and the puppeteers' bursts -> LC) lowers interneuron thresholds. But it never reaches the 40-60%
    band: the band is metabolic."""
    fed, starved = live(1.0, flags={"energy": False}), live(0.15, flags={"energy": False})
    assert starved["l23"] / fed["l23"] > 0.65


def mind_sweep(levels, glucose, warm: int = 12, turns: int = 8, settle: int = 4, seed: int = 0):
    """The whole brain through a dying light: ``warm`` fed turns, then ``turns`` per Lumen level with the insula told
    the Lumen and the cells fed ``glucose(L)``. Returns the Mind's own survival need and desperation per level,
    averaged over its last ``settle`` turns."""
    brain = TomeBrain(seed)
    schedule = [6] * warm + [L for L in levels for _ in range(turns)]
    X = sample_states(len(schedule), torch.Generator().manual_seed(3)).numpy()
    survival, desp = [], []
    for L, x in zip(schedule, X):
        brain.set_us({})
        brain.set_input(director_features=x, body=np.array([L / 6, 1.0, 0.0], np.float32), glucose=glucose(L))
        brain.phase_a()
        brain.phase_b(x)
        survival.append(brain.mind.needs["survival"])
        desp.append(brain.mind.desperation)
    ends = [warm + (j + 1) * turns for j in range(len(levels))]
    return [float(np.mean(survival[e - settle:e])) for e in ends], [float(np.mean(desp[e - settle:e])) for e in ends]


def test_g23_the_minds_own_desperation_rises_as_the_light_dies():
    survival, desp = mind_sweep((6, 4, 2, 1), lumen_glucose)
    assert all(b < a for a, b in zip(survival, survival[1:])), survival
    # monotone, up to the turn-to-turn wobble of the other needs on a fed, eventless brain (desperation SD ~0.02)
    assert all(b >= a - 0.03 for a, b in zip(desp, desp[1:])) and desp[-1] >= desp[0] + 0.3, desp


def test_g23_mutant_glucose_held_at_one_desperation_does_not_rise():
    """The insula still sees the Lumen die, but the cells are fed: no survival deficit, no desperation."""
    survival, desp = mind_sweep((6, 1), lambda L: 1.0)
    assert survival[1] >= survival[0] - 0.02 and desp[1] <= desp[0] + 0.05, (survival, desp)


def test_g23_scheming_rises_as_the_light_dies():
    sweep = [live(lumen_glucose(L), measure_ms=3000) for L in (6, 4, 2, 1)]
    bursts = [s["bursts"] for s in sweep]
    assert bursts[-1] > bursts[0] and all(b >= a * 0.9 for a, b in zip(bursts, bursts[1:]))
    # lesion orexin -> PVT: most of the rise in scheming goes away
    k = next(i for i, pr in enumerate(PR.projections()) if pr.src == "ORX" and pr.dst == "PVT")
    gains = {k: 0.0}
    lesioned = [live(lumen_glucose(L), gains=gains, measure_ms=3000)["bursts"] for L in (6, 1)]
    assert (lesioned[1] - lesioned[0]) <= 0.5 * (bursts[-1] - bursts[0])
