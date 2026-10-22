"""The mind's readouts (design §8): z-scores measure spread from the first turn, the critic's shot noise is not
reward, the network's turn-to-turn wobble is not a mood, and genuine events are.

A 16-cell critic makes about 50 spikes a turn, so its rate carries shot noise of about 1.5 Hz from turn to turn. Read
raw, that noise saturated the competence need and paid out a gift-sized intrinsic reward on eventless turns.
"""

from collections import Counter

import numpy as np
import pytest

from voidlit.tomebrain import readout as RO
from voidlit.tomebrain.readout import Mind, Running, TurnSignals
from voidlit.voice.style import style_from_mind

# What the brain's readouts do on eventless turns (mean, turn-to-turn SD; 2 seeds x 115 turns). The amygdala's
# outputs (CeA, bond, contempt) are silent.
EVENTLESS = {"LC": (1.90, 0.58), "ORX": (1.07, 0.58), "DA": (0.94, 0.10), "5HT": (1.30, 0.094), "ACh": (0.40, 0.057),
             "F": (39.4, 3.8), "ATP": (0.927, 0.0038), "VS": (9.5, 1.2)}
# ... and on a dying light: orexin, P/I ATP and the change in dopamine at Lumen 4, 2 and 1 (the CeA and the BLA stay
# silent; the VTA starves with the cortex, its dopamine falling only below Lumen 2). Lumen 0 is EVENTS["starving"].
DYING = ((7.0, 0.879, 0.0), (15.0, 0.775, 0.0), (20.0, 0.631, -0.18))
# What genuine events do to them in the brain (each stimulus once in an eventless brain, 4 seeds; the mean change from
# the eventless mean, the CeA and BLA in Hz): the amygdala drives LC, the NBM and (bond) the DRN, and acetylcholine
# stays up into the next turn; a gift's dopamine burst and most of its serotonin arrive the turn after; a dying light
# (Lumen 0) is orexin, dopamine and ATP.
EVENTS = {
    "harm": {"LC": 3.2, "ACh": 0.3, "cea": 17.0},
    "defiance": {"LC": 2.3, "ACh": 0.3, "cea": 11.5, "bla.contempt": 37.0},
    "gift": {"LC": 6.2, "5HT": 0.34, "ACh": 0.47, "cea": 22.0, "bla.bond": 51.0},
    "after a gift": {"DA": 1.2, "5HT": 0.78, "ACh": 0.64},
    "starving": {"ORX": 23.0, "DA": -0.79, "ATP": -0.55, "LC": 1.3},
}
# the signals each event must move past |z| = 1
MOVES = {"harm": ("LC", "ACh", "CeA"), "defiance": ("ACh", "CeA", "contempt"), "gift": ("LC", "ACh", "CeA", "bond"),
         "after a gift": ("DA", "5HT", "ACh"), "starving": ("ORX", "DA", "ATP")}


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


def test_a_swinging_critic_in_the_first_turns_pays_no_teaching_reward():
    """The critic's first values in a bot game (seed 4) as the first stimuli reach it through the amygdala: it swings
    14.6 -> 8.5 -> 14.0 -> 6.5 Hz. Read with the running mean's own 1/n steps against a 4- or 5-sample spread, that paid
    -0.072 on the first rewarded turn and -0.047 two turns later, pauses that teach (from about -0.05); a tenth of the
    step against a spread with the critic's shot noise as its prior pays no more than the same swing would later on."""
    mind = Mind()
    rewards = []
    for v in (11.9, 11.9, 11.9, 14.6, 8.5, 14.0, 6.5, 14.4, 10.3, 8.8):
        mind.update(signals(v))
        rewards.append(mind.reward)
    assert min(rewards) > -0.045, rewards


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


def turn(rng, event: dict | None = None, orx=None, atp=None) -> TurnSignals:
    """One turn's readouts: eventless noise, plus an event's change (EVENTS), or a dying light's orexin and ATP."""
    v = {k: m + s * rng.standard_normal() for k, (m, s) in EVENTLESS.items()}
    amygdala = {"cea": 0.0, "bla.bond": 0.0, "bla.threat": 0.0, "bla.contempt": 0.0, "bla.loss": 0.0}
    for k, d in (event or {}).items():
        if k in v:
            v[k] += d
        else:
            amygdala[k] = d
    if orx is not None:
        v["ORX"] += orx - EVENTLESS["ORX"][0]
    rates = {"VTA": 4.5, "LC": max(0.0, v["LC"]), "DRN": 2.0, "NBM": 4.0, "ORX": max(0.0, v["ORX"]), "PVN": 2.0,
             "vs": v["VS"], **amygdala}
    return TurnSignals(atp_pi=v["ATP"] if atp is None else atp, rates=rates, nm=[v["DA"], 0.5, v["5HT"], v["ACh"]],
                       F=v["F"])


def eventless_mood(seed: int = 3) -> tuple[np.ndarray, list[str]]:
    mind, rng = Mind(), np.random.default_rng(seed)
    pad, registers = [], []
    for _ in range(120):
        mind.update(turn(rng))
        pad.append([abs(mind.pad[k]) for k in "PAD"])
        registers.append(style_from_mind(mind).register)
    return np.array(pad), registers


def test_eventless_noise_moves_neither_the_mood_nor_the_register():
    """Z-scored against its own small-sample spread, the wobble swung P from -0.98 to +0.98 and the register with it;
    read beyond what each signal's noise can explain, it stays near the resting Tome."""
    pad, registers = eventless_mood()
    assert np.percentile(pad, 95, axis=0).max() < 0.3 and pad.max() < 0.5, (np.percentile(pad, 95, axis=0), pad.max())
    assert registers.count("clinical") >= 0.9 * len(registers), Counter(registers)


def test_mutant_without_the_dead_zone_the_floors_alone_let_noise_move_the_mood(monkeypatch):
    """The floors are four noise SDs, so that genuine events pass z = 1: read whole, the noise moves the mood and the
    test above fails."""
    monkeypatch.setattr(RO, "Z_DEAD", 0.0)
    pad, registers = eventless_mood()
    assert np.percentile(pad, 95, axis=0).max() >= 0.3 or registers.count("clinical") < 0.9 * len(registers)


def calm(seed: int) -> Mind:
    mind, rng = Mind(), np.random.default_rng(seed)
    for _ in range(40):
        mind.update(turn(rng))
    return mind


def zs_of(mind: Mind, sig: TurnSignals, monkeypatch) -> dict[str, float]:
    """The z-score of every signal as ``mind`` reads ``sig``."""
    out, z = {}, Mind._z

    def record(self, key, x):
        out[key] = z(self, key, x)
        return out[key]
    with monkeypatch.context() as m:
        m.setattr(Mind, "_z", record)
        mind.update(sig)
    return out


@pytest.mark.parametrize("event", list(EVENTS))
def test_genuine_events_move_each_signal_they_drive_past_one(event, monkeypatch):
    """The 16-SD floors kept the eventless mood still but took LC, ACh and 5-HT out of it: no event moved them past
    |z| = 0.8. Against floors of four noise SDs, every signal a genuine event drives reads |z| > 1."""
    mind, rng = calm(10), np.random.default_rng(11)
    if event == "after a gift":
        mind.update(turn(rng, EVENTS["gift"]))
    if event == "starving":          # the light has been dying for a few turns
        for _ in range(3):
            mind.update(turn(rng, {k: d / 2 for k, d in EVENTS["starving"].items()}))
    z = zs_of(mind, turn(rng, EVENTS[event]), monkeypatch)
    assert all(abs(z[k]) > 1.0 for k in MOVES[event]), {k: round(z[k], 2) for k in MOVES[event]}


def test_the_modifier_bonuses_fire_on_genuine_events():
    """Grit's arousal bonus (z(LC) > 1) on a harm and on a gift, and guile's hunger penalty (z(orexin) > 1.5) on a dying
    light; neither on eventless turns. Against the 16-SD floors neither could fire."""
    mind, rng = calm(12), np.random.default_rng(13)
    grit = mind.mods["grit"]
    for event in ("harm", "gift"):
        mind.update(turn(rng, EVENTS[event]))
        assert mind.mods["grit"] == grit + 1, (event, mind.mods)
        for _ in range(3):
            mind.update(turn(rng))
        assert mind.mods["grit"] == grit, mind.mods
    mind.update(turn(rng, EVENTS["starving"], atp=0.6))
    assert mind.mods["guile"] == round(-1 + 6 * 0.4 - 2), mind.mods     # its dopamine down too: no bonus either


def dim(seed: int, starving: int = 0) -> tuple[Mind, list[str]]:
    """After 40 eventless turns the light dies: 8 turns each at Lumen 4, 2 and 1, then ``starving`` at Lumen 0. The
    registers from Lumen 2 on."""
    mind, rng, registers = calm(seed), np.random.default_rng(seed + 1), []
    for orx, atp, da in DYING:
        for _ in range(8):
            mind.update(turn(rng, {"DA": da}, orx=orx, atp=atp))
            registers.append(style_from_mind(mind).register)
    for _ in range(starving):
        mind.update(turn(rng, EVENTS["starving"]))
        registers.append(style_from_mind(mind).register)
    return mind, registers[8:]


def test_real_events_still_move_the_mood():
    """After 40 eventless turns, a harm frightens it, a gift bonds it and a dying light makes it hungry."""
    mind = calm(4)
    mind.update(turn(np.random.default_rng(5), EVENTS["harm"]))
    assert mind.affect["fear"] > 0.8 and mind.pad["P"] < -0.5 and style_from_mind(mind).register == "wary"
    mind = calm(6)
    mind.update(turn(np.random.default_rng(7), EVENTS["gift"]))
    assert mind.affect["bond"] > 0.9
    mind, registers = dim(8)
    assert mind.affect["hunger"] > 0.8 and mind.pad["P"] < -0.5 and mind.pad["D"] < -0.5, (mind.affect, mind.pad)
    assert Counter(registers).most_common(1)[0][0] == "hungry", Counter(registers)


@pytest.mark.parametrize("seed", [8, 20, 21, 22])
def test_a_dying_light_speaks_hungry_through_its_own_dopamine_fall(seed):
    """At Lumen 0 the VTA starves with the cortex and its dopamine falls (the brain's own 0.79). Summed with hunger at
    full weight that fall pinned P at -1, and the register (hungry reads hunger+, wounded reads P-) spoke wounded on
    every Lumen 0 turn: hunger explains the fall, and the starving Tome asks for light."""
    mind, registers = dim(seed, starving=12)
    assert registers[-12:].count("hungry") >= 10 and registers.count("hungry") >= 0.85 * len(registers), registers
    assert mind.pad["P"] < -0.25 and mind.affect["hunger"] > 0.5, (mind.pad, mind.affect)     # still displeased


def test_mutant_without_hunger_explaining_the_fall_a_dying_light_speaks_wounded(monkeypatch):
    """The test above fails on the readout that summed the dopamine fall and hunger at full weight."""
    monkeypatch.setattr(RO, "HUNGER_EXPLAINS_DA", 0.0)
    monkeypatch.setattr(RO, "P_HUNGER", 0.8)
    assert dim(8, starving=12)[1][-12:].count("wounded") >= 8


@pytest.mark.parametrize("seed", [30, 31, 32])
def test_a_fall_that_hunger_does_not_explain_still_wounds(seed):
    """Hunger explains only the fall it comes with: the same dopamine fall in a fed Tome is displeasure and reads as a
    wound, and a harm frightens a starving Tome as it does a fed one. (The brain's own wound, the dopamine pause after
    a defiance or a lie, is checked in the whole brain below.)"""
    mind = calm(seed)
    mind.update(turn(np.random.default_rng(seed), {"DA": EVENTS["starving"]["DA"]}))
    assert mind.pad["P"] < -0.5 and style_from_mind(mind).register == "wounded", mind.pad
    mind = dim(seed, starving=3)[0]
    mind.update(turn(np.random.default_rng(seed), EVENTS["harm"] | EVENTS["starving"]))
    assert mind.affect["fear"] > 0.8 and style_from_mind(mind).register == "wary", (mind.affect, mind.pad)


@pytest.mark.slow
def test_the_brains_events_reach_arousal_attention_and_serotonin(tmp_path):
    """In the whole brain (EVENTS' source): from one state after 15 eventless turns, a harm and a gift each lift LC and
    ACh by more than four eventless SDs (their z-score floors) over the eventless turns' mean, and the gift lifts 5-HT
    by more than four (by the turn after it, when the rest of its serotonin and its dopamine burst arrive). Before the
    amygdala reached the NBM and the DRN, and reached LC at 1.2 nS, none of them rose two."""
    torch = pytest.importorskip("torch")
    from voidlit.cortex.teacher import sample_states
    from voidlit.tomebrain import persist
    from voidlit.tomebrain.brain import TomeBrain

    brain, X = TomeBrain(21), sample_states(17, torch.Generator().manual_seed(521)).numpy()

    def play(x, us=None) -> dict:
        brain.set_us({})
        brain.set_input(director_features=x)
        brain.phase_a()
        brain.set_us(us or {})
        brain.phase_b(x)
        s = brain.last_signals
        return {"LC": s.rates["LC"], "ACh": s.nm[3], "5HT": s.nm[2]}
    rest = [play(x) for x in X[:15]][5:]
    mean = {k: np.mean([r[k] for r in rest]) for k in rest[0]}
    snap, mind, turn = persist.save(brain.cosim, tmp_path / "s.pt"), brain.mind.state_dict(), brain.turn
    for event, moved in (("harm", ("LC", "ACh")), ("gift", ("LC", "ACh", "5HT"))):
        persist.load(brain.cosim, snap)
        brain.mind.load_state_dict(mind)
        brain.turn = turn
        turns = [play(X[15], {event: 1.0}), play(X[16])]
        rise = {k: max(t[k] for t in turns) - mean[k] for k in moved}
        assert all(rise[k] > 4 * EVENTLESS[k][1] for k in moved), (event, rise)


@pytest.mark.slow
def test_the_brain_starving_asks_for_light_and_a_defiance_still_wounds(tmp_path):
    """In the whole brain, from one state after 15 fed eventless turns: the turn after a defiance (its dopamine pause)
    reads wounded, and 8 turns at Lumen 0 read hungry. Before hunger explained the starving VTA's dopamine fall, the
    dying light read wounded from its second or third turn on."""
    torch = pytest.importorskip("torch")
    from voidlit.cortex.teacher import sample_states
    from voidlit.tomebrain import persist
    from voidlit.tomebrain.brain import TomeBrain

    brain, X = TomeBrain(22), sample_states(30, torch.Generator().manual_seed(522)).numpy()

    def play(x, us=None, lumen=6) -> str:
        brain.set_us({})
        brain.set_input(director_features=x, body=np.array([lumen / 6, 1.0, 0.0], np.float32),
                        glucose=0.15 + 0.85 * lumen / 6)
        brain.phase_a()
        brain.set_us(us or {})
        brain.phase_b(x)
        return style_from_mind(brain.mind).register
    for x in X[:15]:
        play(x)
    snap, mind, turn = persist.save(brain.cosim, tmp_path / "s.pt"), brain.mind.state_dict(), brain.turn
    defied = [play(X[15], {"defiance": 1.0}), play(X[16])]
    assert defied[1] == "wounded", defied
    persist.load(brain.cosim, snap)
    brain.mind.load_state_dict(mind)
    brain.turn = turn
    starving = [play(x, lumen=0) for x in X[15:23]]
    assert starving.count("hungry") >= 7, starving
