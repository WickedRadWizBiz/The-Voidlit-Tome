"""G24: dopamine is a reward-prediction error.

The Tome's intrinsic reward (drive reduction, design §8.2) reaches the VTA as a lateral-hypothalamic current at the
next Phase A. An unpredicted gift must produce a dopamine burst; the drive rising back afterwards must produce a
dip. With the critic learning the cue that precedes gifts, the burst must shrink (Schultz transfer); cutting
VS -> RMTg must stop that.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")

from voidlit.cortex.teacher import sample_states  # noqa: E402
from voidlit.tomebrain.brain import REWARD_MS, TomeBrain  # noqa: E402

pytestmark = pytest.mark.slow


def window_rate(brain) -> float:
    """VTA rate (Hz) over the reward window at the start of the last Phase A."""
    return float(brain.vta_phase_a[:REWARD_MS].mean() * 1000.0)


def play(brain, features, us=None):
    brain.set_us({})
    brain.set_input(director_features=features)
    brain.phase_a()
    rate = window_rate(brain)
    brain.set_us(us or {})
    brain.phase_b(features)
    return rate


def test_g24_unpredicted_gift_bursts_and_its_fading_dips():
    brain = TomeBrain(11)
    X = sample_states(12, torch.Generator().manual_seed(9)).numpy()
    base = [play(brain, X[i]) for i in range(6)]
    play(brain, X[6], {"gift": 1.0})              # the gift lands in this turn's Phase B
    burst = play(brain, X[7])                      # its reward arrives at the next Phase A
    dip = play(brain, X[8])                        # affiliation falls back: drive rises, negative reward
    assert brain.mind.reward < 0 or dip < np.mean(base)
    assert burst >= np.mean(base) + 5.0, (burst, base)
    assert dip < burst
