"""G12: the Tome's mood and opinion are never displayed, yet they leak through its wording (design §11.6-11.7).

* Frames carry no affect, opinion, needs, style or beat-probability keys outside dev debug.
* The Tome's own words carry no emotion label at all, and narration never attributes one to the Tome.
* With identical facts, a warm and a cold Tome word them measurably differently: length (Cohen's d > 0.8), form of
  address and where the useful detail sits (chi-squared p < 0.01). Mutant: a voice that ignores the style shows no
  such difference, so the leak test would fail.
* A local model that leaks the Tome's mood in words no lexicon lists (a heavy heart, a glow that sulks) never gets
  one of them into a release frame: what it writes is held to shapes, not to a list of words.
"""

import random
import re
import statistics

import pytest

from voidlit.cortex.director import Director
from voidlit.playtest import bot_command
from voidlit.tome.engine import TomeEngine
from voidlit.voice import OfflineVoice
from voidlit.voice.llama_local import LocalVoice
from voidlit.voice.prompts import TOME_SPEECH_SYSTEM, TOME_SYSTEM
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


class Mopey:
    """A fake local model that leaks the Tome's mood without one emotion label: no word of it is in G12's lexicon or
    the guard's. It keeps whatever it is asked to say verbatim and adds its own lines around it."""

    FRAMINGS = ["Closer, my heart is heavy.", "Hold me, I shiver.", "Listen, the dark weighs on me.",
                "Do you not see me tremble?", "Stay, little light, stay, I am so small.",
                "Walk on, I am lost without you."]
    COLOURS = ["The book trembles, small and cold.", "Its glow sulks in the corner.", "A heavy sigh escapes the glass.",
               "The slab shudders as if wounded.", "Somewhere the pages whimper.", "The words sag and dim."]

    def __init__(self) -> None:
        self.i = 0

    def chat(self, messages, **kw):
        self.i += 1
        system, user = messages[0]["content"], messages[-1]["content"]
        if system == TOME_SPEECH_SYSTEM:
            return self.FRAMINGS[self.i % len(self.FRAMINGS)]
        if system != TOME_SYSTEM:
            return "..."
        core = [x[2:] for x in user.splitlines() if x.startswith("- ")]
        return " ".join(core + [self.COLOURS[self.i % len(self.COLOURS)]])


def test_g12_an_llm_never_puts_the_tomes_mood_into_a_release_frame(scenario, director_model):
    mopey = Mopey()
    for seed in range(3):
        e = TomeEngine(scenario, director=Director(director_model, seed=seed), seed=seed,
                       voice=LocalVoice(mopey, OfflineVoice(scenario.narration, seed=seed)))
        frames = [e.start()]
        for n in range(30):
            frames.append(e.turn("Tome, how do you feel?" if n % 3 == 1 else bot_command(e, smart=True)))
            if e.state.over:
                break
        for f in frames:
            shown = " ".join(sp["text"] for b in f.to_dict()["blocks"] for sp in b["spans"])
            assert not [x for x in Mopey.FRAMINGS + Mopey.COLOURS if x.rstrip(".?") in shown], shown
            for b in f.blocks:
                assert not (b.speaker == "The Tome" and AFFECT.search(b.text)), b.text
                assert not (b.role == "narration" and _attributed(b.text)), b.text


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
