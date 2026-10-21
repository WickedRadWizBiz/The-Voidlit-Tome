"""The mind's readouts (design §8): z-scores measure spread from the first turn, the critic's shot noise is not
reward, and the network's turn-to-turn wobble is not a mood.

A 16-cell critic makes about 50 spikes a turn, so its rate carries shot noise of about 1.5 Hz from turn to turn. Read
raw, that noise saturated the competence need and paid out a gift-sized intrinsic reward on eventless turns.
"""

from collections import Counter

import numpy as np

from voidlit.tomebrain.readout import Mind, Running, TurnSignals
from voidlit.voice.style import style_from_mind

# What the brain's readouts do on eventless turns (mean, turn-to-turn SD; 2 seeds x 115 turns). The amygdala's
# outputs (CeA, bond, contempt) are silent.
EVENTLESS = {"LC": (1.90, 0.58), "ORX": (1.07, 0.58), "DA": (0.94, 0.10), "5HT": (1.30, 0.094), "ACh": (0.40, 0.057),
             "F": (39.4, 3.8), "ATP": (0.927, 0.0038), "VS": (9.5, 1.2)}
# ... and on a dying light: orexin and P/I ATP at Lumen 4, 2 and 1 (the CeA and the BLA stay silent)
DYING = ((7.0, 0.879), (15.0, 0.775), (20.0, 0.631))


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


def test_the_cold_start_pays_no_reward():
    """Over the first turns the needs switch on (smoothing, then the value trend) while the cold first turn weighs on
    the running means: in the brain that paid up to -0.08 on the third turn, a pause long enough to teach."""
    mind = Mind()
    vs = critic_noise(8, seed=4)
    vs[0] = 30.0                                 # the critic's cold first turn
    rewards = []
    for v in vs:
        mind.update(signals(float(v)))
        rewards.append(mind.reward)
    assert rewards[:4] == [0.0] * 4 and rewards[4] != 0.0, rewards


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


def turn(rng, cea=0.0, bond=0.0, orx=None, atp=None) -> TurnSignals:
    """One turn's readouts: eventless noise, plus an event's CeA or bond rate, or a dying light's orexin and ATP."""
    v = {k: m + s * rng.standard_normal() for k, (m, s) in EVENTLESS.items()}
    if orx is not None:
        v["ORX"] += orx - EVENTLESS["ORX"][0]
    rates = {"VTA": 4.5, "LC": max(0.0, v["LC"]), "DRN": 2.0, "NBM": 4.0, "ORX": max(0.0, v["ORX"]), "PVN": 2.0,
             "cea": cea, "vs": v["VS"], "bla.bond": bond, "bla.threat": 0.0, "bla.contempt": 0.0, "bla.loss": 0.0}
    return TurnSignals(atp_pi=v["ATP"] if atp is None else atp, rates=rates, nm=[v["DA"], 0.5, v["5HT"], v["ACh"]],
                       F=v["F"])


def test_eventless_noise_moves_neither_the_mood_nor_the_register():
    """Z-scored against its own small-sample spread, the wobble swung P from -0.98 to +0.98 and the register with it;
    against each signal's noise floor it stays near the resting Tome."""
    mind, rng = Mind(), np.random.default_rng(3)
    pad, registers = [], []
    for _ in range(120):
        mind.update(turn(rng))
        pad.append([abs(mind.pad[k]) for k in "PAD"])
        registers.append(style_from_mind(mind).register)
    pad = np.array(pad)
    assert np.percentile(pad, 95, axis=0).max() < 0.3 and pad.max() < 0.5, (np.percentile(pad, 95, axis=0), pad.max())
    assert registers.count("clinical") >= 0.9 * len(registers), Counter(registers)


def calm(seed: int) -> Mind:
    mind, rng = Mind(), np.random.default_rng(seed)
    for _ in range(40):
        mind.update(turn(rng))
    return mind


def test_real_events_still_move_the_mood():
    """After 40 eventless turns, a harm frightens it, a gift bonds it and a dying light makes it hungry."""
    mind = calm(4)
    mind.update(turn(np.random.default_rng(5), cea=14.0))
    assert mind.affect["fear"] > 0.8 and mind.pad["P"] < -0.5 and style_from_mind(mind).register == "wary"
    mind = calm(6)
    mind.update(turn(np.random.default_rng(7), bond=62.0))
    assert mind.affect["bond"] > 0.9
    mind, rng, registers = calm(8), np.random.default_rng(9), []
    for orx, atp in DYING:
        for _ in range(8):
            mind.update(turn(rng, orx=orx, atp=atp))
            registers.append(style_from_mind(mind).register)
    assert mind.affect["hunger"] > 0.8 and mind.pad["P"] < -0.5 and mind.pad["D"] < -0.5, (mind.affect, mind.pad)
    assert Counter(registers).most_common(1)[0][0] == "hungry", Counter(registers)
