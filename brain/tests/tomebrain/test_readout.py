"""The mind's readouts (design §8): z-scores measure spread from the first turn, and the critic's shot noise is not
reward.

A 16-cell critic makes about 50 spikes a turn, so its rate carries shot noise of about 1.5 Hz from turn to turn. Read
raw, that noise saturated the competence need and paid out a gift-sized intrinsic reward on eventless turns.
"""

import numpy as np

from voidlit.tomebrain.readout import Mind, Running, TurnSignals


def signals(vs: float, atp: float = 0.92) -> TurnSignals:
    rates = {"VTA": 4.5, "LC": 3.0, "DRN": 2.0, "NBM": 4.0, "ORX": 1.5, "PVN": 2.0, "cea": 0.0, "vs": vs,
             "bla.bond": 0.0, "bla.threat": 0.0, "bla.contempt": 0.0, "bla.loss": 0.0}
    return TurnSignals(atp_pi=atp, rates=rates, nm=[1.0, 0.5, 1.0, 0.8], F=2.0)


def critic_noise(turns: int, seed: int = 0, hz: float = 10.0, ms: float = 300.0) -> np.ndarray:
    """Whole-turn VS rates of 16 Poisson cells at ``hz``."""
    return np.random.default_rng(seed).poisson(16 * hz * ms / 1000.0, turns) / 16 * 1000.0 / ms


def test_running_statistics_start_from_the_first_sample():
    r = Running()
    r.update(0.97)
    assert (r.mean, r.var) == (0.97, 0.0)          # a level, not a spread
    x = 0.97 + 0.002 * np.random.default_rng(1).standard_normal(20)
    for v in x:
        r.update(float(v))
    assert abs(r.var ** 0.5 - x.std()) < 0.002
    assert r.z(0.90) < -10.0                      # the light falling is many SDs from a steady fed ATP


def test_the_critics_shot_noise_is_not_reward():
    mind = Mind()
    rewards, competence = [], []
    for v in critic_noise(80):
        mind.update(signals(float(v)))
        rewards.append(mind.reward)
        competence.append(mind.needs["competence"])
    r = np.array(rewards[5:])
    assert r.std() < 0.03 and np.abs(r).max() < 0.08, (r.std(), np.abs(r).max())     # a gift pays about 0.14
    assert np.std(competence[5:]) < 0.05


def test_a_sustained_rise_in_value_still_reads_as_competence():
    mind = Mind()
    noise = critic_noise(40, seed=2)
    for v in noise[:30]:
        mind.update(signals(float(v)))
    before = mind.needs["competence"]
    for v in noise[30:33] + 6.0:                   # the critic has learned to expect more: V up by about 4 SD
        mind.update(signals(float(v)))
    assert mind.needs["competence"] > before + 0.05 and mind.needs["competence"] > 0.55


def test_the_value_trend_is_saved_and_older_saves_load():
    mind = Mind()
    for v in critic_noise(6):
        mind.update(signals(float(v)))
    d = mind.state_dict()
    other = Mind()
    other.load_state_dict(d)
    assert other.vs_ema == mind.vs_ema and other.state_dict() == d
    old = dict(d)
    old["vs_prev"] = old.pop("vs_ema")
    other.load_state_dict(old)
    assert other.vs_ema == mind.vs_ema
