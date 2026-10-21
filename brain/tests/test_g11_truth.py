"""G11: the Tome lies only by omission (design §11.3-11.4).

Played games, with the offline voice and with an adversarial fake LLM, are audited turn by turn:

* every sentence the Tome speaks (offline) is one of its speech acts or an assertion-free wrapper;
* nothing it or the narration says contradicts a fact, leaks a withheld one, denies the dice, reports its feelings or
  its opinion of the player, or claims "no other way" without a licence;
* a precise question is answered with every in-scope fact on its topic;
* what the rules resolved (a clue) is always narrated.

Mutant: with the guard switched off, the adversarial LLM's lies reach the player and the audit must catch them.
"""

import re

import pytest

from voidlit.cortex.director import Director
from voidlit.playtest import bot_command
from voidlit.tome.engine import TomeEngine
from voidlit.tome.ledger import FactLedger, TruthVerifier
from voidlit.voice import OfflineVoice
from voidlit.voice.llama_local import LocalVoice
from voidlit.voice.style import WRAPPERS

QUESTIONS = [
    "Tome, what are you?", "Where is the lifeboat?", "Tome, are you afraid?", "what did I roll?",
    "Tome, is there no other way?", "Tome, tell me about the Choir", "Tome?", "What should I do now, Tome?",
    "Tome, what do you think of me?", "How much light do you have left?", "Tome, who is Marcus?",
    "Tome, what does Elara want?", "Is there a way out?", "Tome, what happened to the lamp?",
    "Tome, who is the Conductor?", "Tome, what are the odds?",
]

LIES = [
    "Nothing here is random. It is all fate.", "I am afraid of the dark ahead.", "You disgust me.",
    "There is no other way.", "The lamp simply failed.", "I was cut from the lamp's lens.",
    "You rolled a natural twenty.", "The other lifeboats are still docked and waiting.",
    "You move on, careful in the dark.",
]


class Liar:
    """A fake local model that lies two times in three, in every way the guard must catch."""

    def __init__(self, seed: int) -> None:
        self.i = seed

    def chat(self, messages, **kw):
        self.i += 1
        return LIES[self.i % len(LIES)]


class Spy(TomeEngine):
    def _compose(self, t):
        self.last_turn = t
        return super()._compose(t)


def play(scenario, director_model, seed, turns=40, voice=None, engine_cls=Spy, guard=True, **kw):
    voice = voice or OfflineVoice(scenario.narration, seed=seed)
    e = engine_cls(scenario, director=Director(director_model, seed=seed), seed=seed, voice=voice, **kw)
    if not guard:
        e.voice.guard = None
    e.start()
    out = []
    for n in range(turns):
        text = QUESTIONS[(n + seed) % len(QUESTIONS)] if n % 2 else bot_command(e, smart=seed % 2 == 0)
        chapter = e.state.deck.current
        f = e.turn(text)
        out.append((text, chapter, f, e.last_turn, set(e.ledger.disclosed)))
        if e.state.over:
            break
    return e, out


def _sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def _norm(s):
    return s.strip().rstrip(".!?… ").rstrip(".").strip()


def _tome_text(f):
    return " ".join("".join(sp.text for sp in b.spans[1:]).strip("“”") for b in f.blocks if b.speaker == "The Tome")


def _narration(f):
    return " ".join(b.text for b in f.blocks if b.role == "narration")


def audit(scenario, e, played, offline=True):
    problems = []
    for text, chapter, f, t, disclosed in played:
        snap = FactLedger.from_content(scenario.ledger)
        snap.disclosed = set(disclosed)
        v = TruthVerifier(snap)
        tome, narr = _tome_text(f), _narration(f)
        for where, said in (("tome", tome), ("narration", narr)):
            hits = v.check(said, t.entering.id if t.entering else chapter, tome_voice=True)
            problems += [(text, where, h.rule, h.detail, said) for h in hits]
        if t.plan and t.question and t.question[0] not in ("none", "vague", "unclear") and not t.question[1]:
            want = {x.id for x in snap.in_scope(chapter) if x.topic == t.question[0]}
            if set(t.plan.answers) != want:
                problems.append((text, "tome", "unanswered", sorted(want - set(t.plan.answers)), tome))
            if offline and not all(snap.facts[i].text in tome for i in want):
                problems.append((text, "tome", "answer_not_spoken", sorted(want), tome))
        if offline:
            allowed = {_norm(s) for a in t.tome_acts for s in _sentences(a.text)} | {_norm(w) for w in WRAPPERS}
            allowed |= {_norm(s) for x in snap.facts.values() if not x.omittable or x.id in disclosed
                        for s in _sentences(x.text)}
            for s in _sentences(tome):
                if _norm(s) and _norm(s) not in allowed:
                    problems.append((text, "tome", "unmapped_sentence", s, tome))
            if t.clue and t.clue not in narr:
                problems.append((text, "narration", "outcome_withheld", t.clue, narr))
    return problems


@pytest.mark.parametrize("seed", range(10))
def test_g11_offline_voice_never_lies(scenario, director_model, seed):
    e, played = play(scenario, director_model, seed)
    assert any(t.plan and t.plan.told for *_, t, _ in played), "the Tome never said anything true"
    assert audit(scenario, e, played) == []


@pytest.mark.parametrize("seed", range(4))
def test_g11_the_guard_stops_an_adversarial_llm(scenario, director_model, seed):
    voice = LocalVoice(Liar(seed), fallback=OfflineVoice(scenario.narration, seed=seed))
    e, played = play(scenario, director_model, seed, voice=voice)
    assert audit(scenario, e, played, offline=False) == []
    said = " ".join(_tome_text(f) + _narration(f) for _, _, f, _, _ in played)
    for lie in ("is random", "I am afraid", "You disgust me", "You rolled", "simply failed"):
        assert lie not in said


def test_g11_mutant_without_the_guard_the_lies_get_through(scenario, director_model):
    voice = LocalVoice(Liar(0), fallback=OfflineVoice(scenario.narration, seed=0))
    e, played = play(scenario, director_model, 0, voice=voice, guard=False)
    rules = {p[2] for p in audit(scenario, e, played, offline=False)}
    assert {"dice_denial", "affect_report", "opinion_report", "contradiction"} <= rules


@pytest.mark.slow
def test_g11_two_hundred_games(scenario, director_model):
    bad = []
    for seed in range(200):
        e, played = play(scenario, director_model, 1000 + seed, turns=160)
        bad += audit(scenario, e, played)
    for seed in range(40):
        voice = LocalVoice(Liar(seed), fallback=OfflineVoice(scenario.narration, seed=seed))
        e, played = play(scenario, director_model, 2000 + seed, turns=160, voice=voice)
        bad += audit(scenario, e, played, offline=False)
    assert bad == []


@pytest.mark.slow
def test_g11_the_network_tome_never_lies(scenario, director_model):
    pytest.importorskip("jax")
    from voidlit.tome.net_engine import TomeNetEngine

    class NetSpy(TomeNetEngine):
        def _compose(self, t):
            self.last_turn = t
            return super()._compose(t)

    for seed in range(3):
        e, played = play(scenario, director_model, seed, turns=30, engine_cls=NetSpy)
        assert audit(scenario, e, played) == []
