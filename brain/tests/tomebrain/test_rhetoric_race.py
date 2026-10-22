"""RHETORIC commits through its basal ganglia race, not the fallback rule (M6 step 1).

RHETORIC has no habit prior. With only the cortex's input and its Gumbel excitability, about a third of its races
never released GPi: Phase B ran to its cap and the fallback rule chose. Its D1 cells now get the drive a PACE beat
gets at the top of its prior (``FLAT_D1``), so at init, when the cortex reaches every channel alike, it samples a flat
prior: the argmax of its Gumbel.

    100 races at init: RHETORIC falls back in <= 5%, and its winner is the argmax of its Gumbel in >= 80%.
    Mutant FLAT_D1 = 0: it falls back in > 10%.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
torch = pytest.importorskip("torch")

from voidlit.cortex.teacher import sample_states  # noqa: E402
from voidlit.tomebrain.brain import TomeBrain  # noqa: E402
from voidlit.tomebrain.sjpart import basal_ganglia as BG  # noqa: E402

RHETORIC = [c for c, (loop, _) in enumerate(BG.CHANNELS) if loop == "RHETORIC"]


def rhetoric_races(n: int, seed: int = 0) -> list[tuple[str, str, bool]]:
    """n races of a new brain: RHETORIC's winner, the argmax of its Gumbel, and whether it fell back."""
    brain = TomeBrain(seed)
    X = sample_states(n, torch.Generator().manual_seed(100 + seed)).numpy()
    rows = []
    for x in X:
        brain.set_input(director_features=x)
        brain.phase_a()
        out = brain.phase_b(x)
        g = brain.sj.streams.gumbel("bg_gumbel", brain.turn - 1, BG.N_CH)[RHETORIC]
        rows.append((out.rhetoric.channel, BG.CHANNELS[RHETORIC[int(np.argmax(g))]][1], out.rhetoric.fallback))
    return rows


@pytest.mark.slow
def test_rhetoric_commits_to_the_argmax_of_its_gumbel():
    rows = rhetoric_races(100)
    fallback = np.mean([fell for *_, fell in rows])
    hits = np.mean([won == best for won, best, _ in rows])
    assert fallback <= 0.05, fallback
    assert hits >= 0.80, hits


@pytest.mark.slow
def test_rhetoric_mutant_without_the_flat_drive_falls_back(monkeypatch):
    monkeypatch.setattr(BG, "FLAT_D1", 0.0)
    rows = rhetoric_races(100)
    assert np.mean([fell for *_, fell in rows]) > 0.10


class _SilentBus:
    """No cortex, thalamus, amygdala or neuromodulator input: the D1 cells get only the race's own drive."""
    buf: dict = {}
    WIDTH = {"J.ctx": BG.N_CTX, "J.pvt": BG.N_PVT, "J.nm": 4}

    def read(self, src, t0, W, delay):
        return np.zeros((W, self.WIDTH[src]), np.float32)


def d1_input(da_mod: float) -> np.ndarray:
    """The D1 cells' input on one Phase B tick, long after the ramp, with the slow dopamine at ``da_mod``."""
    sj = BG.SJPartition(0)
    sj.in_phase_b, sj.phase_b_tick, sj.da_mod = True, 0, da_mod
    sj.gumbel = torch.linspace(-1.0, 2.0, BG.N_CH)
    seen, pop = [], sj.pop
    sj.pop = lambda I: (seen.append(I.clone()), pop(I))[1]
    sj.step(_SilentBus(), 1000, 1)
    lo, hi = sj.off["d1"]
    return seen[0][lo:hi].numpy().astype(np.float64)


def test_dopamine_is_a_gain_on_the_flat_drive_and_the_gumbel():
    """The review rounds' rule: dopamine scales the D1 cells' whole drive, so the race keeps its temperature at any
    dopamine level. With no other input, what dopamine adds is 0.5 x da_mod x (Gumbel + flat drive), on the RHETORIC
    and FRAME cells (flat drive) as on the PACE cells (Gumbel only). A flat drive added outside the gain fails."""
    a = 0.4
    gain = 0.5 * a * (1.0 - 1.0 / BG.DA_MOD_TAU_MS)           # da_mod decays one tick toward 0 (no dopamine yet)
    excite = (BG.BETA_G * torch.linspace(-1.0, 2.0, BG.N_CH).repeat_interleave(BG.M)).numpy() \
        + BG.FLAT_D1 * BG.SJPartition(0).flat.numpy()
    np.testing.assert_allclose(d1_input(a) - d1_input(0.0), gain * excite, rtol=1e-4, atol=1e-7)
