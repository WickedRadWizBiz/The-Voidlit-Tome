"""G3b: the survival need comes from metabolism. G23: desperation is physiology, not a script.

As the Lumen dies (glucose 1.0 -> 0.15): ATP falls, glucose-inhibited orexin cells rise at least tenfold, the dopamine
cells slow, and the cortex weakens to 40-60% of its baseline rather than going silent. Hunger reaches the schemer's
corruption and bend cells only through orexin -> PVT -> their tufts, so their bursting rises as the Lumen falls,
and a lesion of that path removes most of the rise.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from voidlit.tomebrain.calib import calib_rates as CR  # noqa: E402
from voidlit.tomebrain.jaxpart import network as NW  # noqa: E402
from voidlit.tomebrain.jaxpart import params as PR  # noqa: E402
from voidlit.tomebrain.readout import atp_pi, desperation, drive  # noqa: E402

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


def _desperation(atp: float) -> float:
    h = {"survival": min(1.0, max(0.0, (atp - 0.55) / 0.44)), "affiliation": 0.5, "certainty": 1.0,
         "competence": 0.5}
    return desperation(drive(h))


def test_g23_desperation_and_scheming_rise_as_the_light_dies():
    sweep = [live(lumen_glucose(L), measure_ms=3000) for L in (6, 4, 2, 1)]
    desp = [_desperation(s["atp"]) for s in sweep]
    bursts = [s["bursts"] for s in sweep]
    assert all(b >= a - 1e-9 for a, b in zip(desp, desp[1:])) and desp[-1] > desp[0]
    assert bursts[-1] > bursts[0] and all(b >= a * 0.9 for a, b in zip(bursts, bursts[1:]))
    # lesion orexin -> PVT: most of the rise in scheming goes away
    k = next(i for i, pr in enumerate(PR.projections()) if pr.src == "ORX" and pr.dst == "PVT")
    gains = {k: 0.0}
    lesioned = [live(lumen_glucose(L), gains=gains, measure_ms=3000)["bursts"] for L in (6, 1)]
    assert (lesioned[1] - lesioned[0]) <= 0.5 * (bursts[-1] - bursts[0])
