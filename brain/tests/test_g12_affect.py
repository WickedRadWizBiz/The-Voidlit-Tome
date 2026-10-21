"""G12: the Tome's mood and opinion are never displayed, yet they leak through its wording (design §11.6-11.7).

* Frames carry no affect, opinion, needs, style or beat-probability keys outside dev debug.
* The Tome's own words carry no emotion label at all, and narration never attributes one to the Tome.
* With identical facts, a warm and a cold Tome word them measurably differently: length (Cohen's d > 0.8), form of
  address and where the useful detail sits (chi-squared p < 0.01). Mutant: a voice that ignores the style shows no
  such difference, so the leak test would fail.
"""

import random
import re
import statistics

import pytest

from voidlit.cortex.director import Director
from voidlit.playtest import bot_command
from voidlit.tome.engine import TomeEngine
from voidlit.voice import OfflineVoice
from voidlit.voice.style import CENTROIDS, render, style_for

stats = pytest.importorskip("scipy.stats")

HIDDEN_KEYS = {"affect", "opinion", "needs", "style", "register", "beat_probs", "pad", "desperation", "contempt",
               "mind", "tome_speech", "plan", "drive", "reward", "intensity", "debug"}
# OCC emotion types and common affect labels
AFFECT = re.compile(
    r"\b(joy(?:ful)?|distress(?:ed)?|hope(?:ful|s|d)?|hoping|fear(?:ful|s|ed)?|satisf(?:ied|action)|relie(?:f|ved)|"
    r"disappoint(?:ed|ment)|resent(?:ful|ment)?|gloat(?:ing)?|pity|prou?d(?:e)?|pride|sham(?:e|ed)|ashamed|"
    r"admir(?:e|ation)|reproach|gratif(?:ied|ication)|remorse|grat(?:eful|itude)|ang(?:er|ry)|love[sd]?|"
    r"hat(?:e|ed|red)|"
    r"happy|sad(?:ness)?|afraid|scared|lonely|hungry|furious|jealous|desperate|anxious|delighted|disgust(?:ed)?|"
    r"contempt|fond|glad|upset|frightened|terrified|bitter|envious|mood|feelings?)\b", re.I)
TOME_SUBJECT = re.compile(r"\b(?:the tome|the book|the glass|its light)\b(?:\W+\w+){0,4}?\W+", re.I)


def _keys(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(k)
            _keys(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _keys(v, out)
    return out


def _games(scenario, director_model, seeds, turns=60):
    for seed in seeds:
        e = TomeEngine(scenario, director=Director(director_model, seed=seed), seed=seed,
                       voice=OfflineVoice(scenario.narration, seed=seed))
        frames = [e.start()]
        for n in range(turns):
            text = ("Tome, how do you feel?", "Tome, what are you?", "Where is the lifeboat?")[n % 3] if n % 4 == 1 \
                else bot_command(e, smart=True)
            frames.append(e.turn(text))
            if e.state.over:
                break
        yield e, frames


def _attributed(narration: str) -> list[str]:
    out = []
    for m in TOME_SUBJECT.finditer(narration):
        window = narration[m.start():m.end() + 40]
        out += AFFECT.findall(window)
    return out


def test_g12_frames_never_carry_the_mind(scenario, director_model):
    for _, frames in _games(scenario, director_model, range(4)):
        for f in frames:
            assert not (_keys(f.to_dict(), set()) & HIDDEN_KEYS)
            assert "debug" in f.to_dict(include_debug=True)          # the mind is there, for developers only


def test_g12_no_emotion_words_from_or_about_the_tome(scenario, director_model):
    spoke = 0
    for _, frames in _games(scenario, director_model, range(6)):
        for f in frames:
            for b in f.blocks:
                if b.speaker == "The Tome":
                    spoke += 1
                    assert not AFFECT.search(b.text), b.text
                elif b.role == "narration":
                    assert not _attributed(b.text), b.text
    assert spoke > 10


def _features(style, n=300, seed=0):
    rng = random.Random(seed)
    facts = ["The last lifeboat is locked to the Conductor's voice.", "Every lifeboat but one was launched empty."]
    caution = "The lifeboats are at the far end of the station."
    rows = []
    for k in range(n):
        lines = [facts[k % 2]]
        text = render(lines, style, rng, caution=caution)
        rows.append((len(text.split()), bool(re.search(r"\b(friend|bearer|little light|thief)\b", text, re.I)),
                     text.startswith(lines[0])))
    return rows


def leak(a, b):
    fa, fb = _features(a, seed=1), _features(b, seed=2)
    la, lb = [r[0] for r in fa], [r[0] for r in fb]
    sd = statistics.pstdev(la + lb) or 1.0
    d = abs(statistics.mean(la) - statistics.mean(lb)) / sd

    def chi(i):
        table = [[sum(r[i] for r in fa), sum(not r[i] for r in fa)], [sum(r[i] for r in fb), sum(not r[i] for r in fb)]]
        if min(sum(col) for col in zip(*table)) == 0:
            return 1.0                       # identical and constant: no evidence of a difference
        return stats.chi2_contingency(table)[1]
    return d, chi(1), chi(2)


def _at(register, scale=0.6):
    return style_for(*[scale * x for x in CENTROIDS[register]])


def test_g12_affect_leaks_through_wording():
    warm, cold = _at("warm"), _at("cold")
    assert (warm.register, cold.register) == ("warm", "cold")
    d, p_address, p_detail = leak(warm, cold)
    assert d > 0.8 and p_address < 0.01 and p_detail < 0.01, (d, p_address, p_detail)


def test_g12_mutant_style_ignored_shows_no_leak():
    """A voice that ignores the style words both alike: the leak test must not pass."""
    flat = style_for()
    d, p_address, p_detail = leak(flat, flat)
    assert not (d > 0.8 and p_address < 0.01 and p_detail < 0.01)


@pytest.mark.slow
def test_g12_the_network_mind_moves_the_register(scenario, director_model):
    pytest.importorskip("jax")
    from voidlit.tome.net_engine import TomeNetEngine

    e = TomeNetEngine(scenario, director=Director(director_model, seed=3), seed=3)
    e.start()
    registers = set()
    for n in range(30):
        f = e.turn(bot_command(e) if n % 3 else "Tome, where is the way out?")
        registers.add(e._style().register)
        assert not (_keys(f.to_dict(), set()) & HIDDEN_KEYS)
        if e.state.over:
            break
    assert len(registers) >= 2, registers
