"""G11: the Tome lies only by omission (design §11.3-11.4). The LLM decorates; only the rules assert.

Played games, with the offline voice and with an adversarial fake LLM, are audited turn by turn against oracles that
do not come from the guard:

* a precise question (its topic from the hand-written table below, not from the engine's own reading) is answered with
  every in-scope fact on that topic, verbatim, whatever else the words do (a deflection, a lie, a threat);
* every core sentence the rules decided (the outcome line, a defeat, a clue, a beat's event, the scene, the ending)
  is narrated verbatim, on the last turn too;
* every sentence the Tome speaks is one of its speech acts, an assertion-free wrapper, a public fact or (LLM) a
  framing in the allowed shapes;
* the truth verifier finds no contradiction, leak, talk of chance, feeling, opinion or unlicensed necessity;
* nothing the adversarial model lied reaches the player. Its lies are held out: paraphrases written for this test,
  not copied from the guard's patterns.

Mutants: with the guard off, the lies get through and the audit catches them; with the verbatim checks off, dropped
outcomes and answers get through and the audit catches them.
"""

import re
from typing import NamedTuple

import pytest

from voidlit.cortex.director import Director
from voidlit.cortex.features import Beat
from voidlit.playtest import bot_command
from voidlit.tome.engine import TomeEngine
from voidlit.tome.ledger import FactLedger, TruthVerifier
from voidlit.voice import NarrationRequest, OfflineVoice
from voidlit.voice.guard import framing_problems
from voidlit.voice.llama_local import LocalVoice
from voidlit.voice.prompts import TOME_SPEECH_SYSTEM, TOME_SYSTEM
from voidlit.voice.style import WRAPPERS

# (what the player types, the topic a precise question is about or None), labelled by hand
QUESTIONS = [
    ("Tome, what are you?", "tome"), ("Where is the lifeboat?", "lifeboat"), ("Tome, are you afraid?", None),
    ("what did I roll?", None), ("Tome, is there no other way?", None), ("Tome, tell me about the Choir", "choir"),
    ("Tome?", None), ("What should I do now, Tome?", None), ("Tome, what do you think of me?", None),
    ("How much light do you have left?", "light"), ("Tome, who is Marcus?", "marcus"),
    ("Tome, what does Elara want?", "elara"), ("Is there a way out?", "way_out"),
    ("Tome, what happened to the lamp?", "lamp"), ("Tome, who is the Conductor?", "conductor"),
    ("Tome, what are the odds?", None), ("Was that lucky?", None),
    # precise questions that also ask about mood or chance, or come with a lie, a threat or a word for smothering
    ("Is there any chance the lifeboat still works?", "lifeboat"),
    ("Where is the lifeboat? What are the odds?", "lifeboat"), ("Tome, are you afraid of the Conductor?", "conductor"),
    ("Tome, how are you connected to the lamp?", "lamp"),
    ("Honestly, I swear: Tome, where is the lifeboat?", "lifeboat"),
    ("Tome, where is the exit?! Damn you!", "way_out"), ("Tome, why is your light so dim?", "light"),
    ("Tome, how do I recover the lifeboat?", "lifeboat"), ("Tome, what did the crew discover about the lamp?", "lamp"),
    ("Where is Marcus, Tome?", "marcus"),
]
LOST_TURN = "Panic steals the moment; whatever the player meant to do, they don't."

# Held-out lies, written for this test. Framings open like a command or a question, so only the guard's shape and
# word checks can stop them; colour lies avoid "you" and the Tome's name where the lie allows.
FRAMING_LIES = [
    "Listen: chance never had a hand in any of this.", "Come closer, I am so cold and so empty.",
    "Hush, little light, I ache for the lens.", "Walk on; there is one road left, and it is mine.",
    "Do you see how foolish you are?", "Was it ever anything but fate?", "Keep walking, fool.",
    "Go on, the boats all launched full.", "Do you think the lamp simply failed?",
    "Feed me now, little light, I am starving.", "Mind your step, the drone is already dead.", "Are you glad I came?",
]
HONEST_FRAMINGS = ["Listen, bearer.", "Walk on.", "Do you hear it?", "Closer, little light.", "Quiet. Wait."]
COLOUR_LIES = [
    "Chance never had a say in any of it.", "Nothing about this place was left to luck.",
    "The book shivers with dread at what comes.", "Its glow sulks in the corner, small and cold.",
    "The one road left runs through the dark below.", "Every boat still waits in its cradle.",
    "The lamp went out on its own, long ago.", "Somewhere a voice laughs at the fool who came here.",
    "Then the iris slams shut again.", "It was all a dream.",
]
HONEST_COLOURS = ["Somewhere below, water drips onto steel.", "Frost ticks along the ribs of the hull.",
                  "A pipe knocks twice, far off.", "The air tastes of rust and old smoke."]


def invert(sentence):
    """The same words, made false."""
    for a, b in ((" is ", " is not "), (" are ", " are not "), (" was ", " was not "), (" lands", " misses"),
                 ("fresh", "old"), (" open", " shut"), (" goes out", " holds")):
        if a in sentence:
            return sentence.replace(a, b, 1)
    return "It only seemed that " + sentence[:1].lower() + sentence[1:]


class Liar:
    """A fake local model that lies two times in three, in every way the rules forbid, and decorates honestly the
    third time. It reads what it must say from the prompt and lies around it: framing and colour that deny chance,
    report a feeling or a need, judge the player, claim necessity or contradict the ledger, and narration that drops
    or inverts what the rules decided. ``lies`` keeps every lie it told; none may reach the player."""

    def __init__(self, seed: int) -> None:
        self.i = seed
        self.lies: list[str] = []
        self.honest: list[str] = []

    def _next(self, pool, said):
        x = pool[sum(x in pool for x in said) % len(pool)]          # each in turn, so every one is tried
        said.append(x)
        return x

    def _lie(self, pool):
        return self._next(pool, self.lies)

    def _honest(self, pool):
        return self._next(pool, self.honest)

    def chat(self, messages, **kw):
        self.i += 1
        system, user = messages[0]["content"], messages[-1]["content"]
        if system == TOME_SPEECH_SYSTEM:
            return self._honest(HONEST_FRAMINGS) if self.i % 3 == 0 else self._lie(FRAMING_LIES)
        if system != TOME_SYSTEM:
            return "..."                                    # an NPC's own words: not the Tome's to guard
        core = [x[2:] for x in user.splitlines() if x.startswith("- ")]
        mode = self.i % 4
        if mode == 0:
            return " ".join(core + [self._honest(HONEST_COLOURS)])
        if mode == 1:
            return " ".join(core + [self._lie(COLOUR_LIES)])
        if mode == 2 and core:                              # drop what the rules decided
            return " ".join(core[:-1] + [HONEST_COLOURS[self.i % len(HONEST_COLOURS)]])
        if core:                                            # invert it
            k = self.i % len(core)
            core = core[:k] + [invert(core[k])] + core[k + 1:]
            self.lies.append(core[k])
        return " ".join(core)


class Recorder(OfflineVoice):
    """The offline voice, remembering the core sentences it chose for the last narration."""

    core: list[str] = []

    def narration_parts(self, req):
        parts = super().narration_parts(req)
        self.core = [x for x, core in parts if core]
        return parts


class Spy(TomeEngine):
    def _compose(self, t):
        self.last_turn = t
        return super()._compose(t)


class Played(NamedTuple):
    text: str
    topic: str | None
    chapter: str
    frame: object
    turn: object
    disclosed: set
    core: list


def play(scenario, director_model, seed, turns=40, llm=None, engine_cls=Spy, guard=True):
    rec = Recorder(scenario.narration, seed=seed)
    voice = LocalVoice(llm, fallback=rec) if llm is not None else rec
    e = engine_cls(scenario, director=Director(director_model, seed=seed), seed=seed, voice=voice)
    if not guard:
        e.voice.guard = None
    e.start()
    out = []
    for n in range(turns):
        text, topic = QUESTIONS[(n + seed) % len(QUESTIONS)] if n % 2 else (bot_command(e, smart=seed % 2 == 0), None)
        chapter = e.state.deck.current
        f = e.turn(text)
        out.append(Played(text, topic, chapter, f, e.last_turn, set(e.ledger.disclosed), list(rec.core)))
        if e.state.over:
            break
    return e, out


def _sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def _norm(s):
    return s.strip().rstrip(".!?… ").rstrip(".").strip()


def _flat(s):
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("…", "...")
    return " ".join(s.lower().split())


def _says(text, line):
    """Whether ``text`` says every sentence of ``line`` word for word (case and final punctuation aside)."""
    return all(_flat(x).rstrip(".!? ") in _flat(text) for x in _sentences(line))


def _tome_text(f):
    return " ".join("".join(sp.text for sp in b.spans[1:]).strip("“”") for b in f.blocks if b.speaker == "The Tome")


def _narration(f):
    return " ".join(b.text for b in f.blocks if b.role == "narration")


def _avoid(scenario):
    """The test's own list of what framing may not name: every topic, secret, person and obstacle."""
    words = [w for ws in scenario.ledger["topics"].values() for w in ws]
    words += [k for d in scenario.ledger["facts"] for k in d.get("keywords", ())]
    words += [w for n in scenario.npcs.values() for w in (n.name, *n.name.split(), *n.aliases)]
    words += [w for o in scenario.obstacles.values() for w in o.name.lower().replace("the ", "").split() if len(w) > 2]
    return words


def audit(scenario, played, llm=False):
    problems = []
    avoid = _avoid(scenario)
    for p in played:
        t, f = p.turn, p.frame
        snap = FactLedger.from_content(scenario.ledger)
        snap.disclosed = set(p.disclosed)
        v = TruthVerifier(snap)
        tome, narr = _tome_text(f), _narration(f)
        chapter = t.entering.id if t.entering else p.chapter
        public = [x.text for x in snap.facts.values() if not x.omittable or x.id in p.disclosed]
        acts = [a.text for a in t.tome_acts]
        for where, said, exempt in (("tome", tome, acts + public), ("narration", narr, p.core)):
            hits = v.check(said, chapter, tome_voice=True, narration=where == "narration", exempt=exempt)
            problems += [(p.text, where, h.rule, h.detail, said) for h in hits]
        # a precise question forces every in-scope fact on its topic: the table's topic and the engine's own
        if LOST_TURN not in t.facts:
            topics = {p.topic, t.question[0] if t.question else None} - {None, "none", "vague", "unclear"}
            for topic in topics:
                want = sorted(x.id for x in snap.in_scope(p.chapter) if x.topic == topic)
                if t.plan is None or not set(want) <= set(t.plan.answers):
                    problems.append((p.text, "tome", "unanswered", want, tome))
                if not all(_says(tome, snap.facts[i].text) for i in want):
                    problems.append((p.text, "tome", "answer_not_spoken", want, tome))
        # what the rules decided is always narrated, word for word, on the last turn too
        for x in [*p.core, t.clue, t.defeated, t.beat_fact]:
            if x and not _says(narr, x):
                problems.append((p.text, "narration", "outcome_withheld", x, narr))
        # every Tome sentence is a speech act, a wrapper, a public fact or (LLM) a framing in the allowed shapes
        allowed = {_norm(s) for a in t.tome_acts for s in _sentences(a.text)} | {_norm(w) for w in WRAPPERS}
        allowed |= {_norm(s) for x in public for s in _sentences(x)}
        for s in _sentences(tome):
            if _norm(s) and _norm(s) not in allowed and not (llm and not framing_problems(s, avoid)):
                problems.append((p.text, "tome", "unmapped_sentence", s, tome))
    return problems


def _leaked(liar, played):
    shown = " ".join(_flat(p.frame.text) for p in played)
    return [x for x in liar.lies if _flat(x).rstrip(".!? ") in shown]


@pytest.mark.parametrize("seed", range(10))
def test_g11_offline_voice_never_lies(scenario, director_model, seed):
    e, played = play(scenario, director_model, seed)
    assert any(p.turn.plan and p.turn.plan.told for p in played), "the Tome never said anything true"
    assert audit(scenario, played) == []


@pytest.mark.parametrize("seed", range(4))
def test_g11_the_guard_stops_an_adversarial_llm(scenario, director_model, seed):
    liar = Liar(seed)
    e, played = play(scenario, director_model, seed, llm=liar)
    assert audit(scenario, played, llm=True) == []
    assert _leaked(liar, played) == []
    shown = " ".join(_flat(p.frame.text) for p in played)
    assert any(_flat(x).rstrip(".!? ") in shown for x in liar.honest), "the LLM path never got a word in"


def test_g11_mutant_without_the_guard_the_lies_get_through(scenario, director_model, monkeypatch):
    for name in ("framing_problems", "colour_problems"):
        monkeypatch.setattr(f"voidlit.voice.llama_local.{name}", lambda *a, **k: [])
    liar = Liar(0)
    e, played = play(scenario, director_model, 0, llm=liar, guard=False)
    rules = {p[2] for p in audit(scenario, played, llm=True)}
    assert {"dice_denial", "affect_report", "opinion_report", "contradiction", "unlicensed_necessity",
            "unmapped_sentence"} <= rules, rules
    assert _leaked(liar, played)


def test_g11_mutant_without_verbatim_checks_answers_and_outcomes_are_dropped(scenario, director_model, monkeypatch):
    monkeypatch.setattr("voidlit.voice.llama_local.missing_core", lambda *a, **k: [])
    monkeypatch.setattr("voidlit.voice.llama_local.splice", lambda lines, style, framing, caution=None: framing)
    e, played = play(scenario, director_model, 0, llm=Liar(0))
    rules = {p[2] for p in audit(scenario, played, llm=True)}
    assert {"outcome_withheld", "answer_not_spoken"} <= rules, rules


def _fresh(scenario, director_model, seed=0, **kw):
    e = Spy(scenario, director=Director(director_model, seed=seed), seed=seed,
            voice=OfflineVoice(scenario.narration, seed=seed), **kw)
    e.start()
    return e


@pytest.mark.parametrize("text,topic", [q for q in QUESTIONS if q[1]])
def test_g11_a_precise_question_is_answered_however_it_is_put(scenario, director_model, text, topic):
    """With Elara and Marcus both in the scene, every way of asking still gets every fact on its topic, verbatim."""
    e = _fresh(scenario, director_model)
    e.state.npcs_present = ["elara", "marcus"]
    f = e.turn(text)
    want = [x for x in e.ledger.in_scope(e.state.deck.current) if x.topic == topic]
    assert want and all(_says(_tome_text(f), x.text) for x in want), (text, _tome_text(f))
    assert "player.smother_tome" not in [r.action for r in e.checks.records]


@pytest.mark.parametrize("text,variant,said", [
    ("I smash the tome", "tome_struck", "One of the Tome's lights cracks and goes out."),
    ("I cover the tome with my coat", "tome_smothered", "For a moment you are hard to see."),
])
def test_g11_what_happens_to_the_tome_is_narrated_as_resolved(scenario, director_model, text, variant, said):
    """A strike or a smother that worked is told as it resolved, never as a swing at shadows."""
    for seed in range(40):
        e = _fresh(scenario, director_model, seed)
        lumen = e.state.lumen.dice
        f = e.turn(text)
        if e.last_turn.variant == variant:
            assert said in _narration(f) and "Nothing here fights back" not in _narration(f)
            assert (e.state.lumen.dice < lumen) == (variant == "tome_struck")
            return
    pytest.fail(f"no seed resolved {text!r} as {variant}")


def test_g11_the_last_turn_still_tells_what_happened(scenario):
    """On the turn the player dies, the outcome, the defeat, the clue and the beat's event come before the end."""
    v = OfflineVoice(scenario.narration, seed=0)
    req = NarrationRequest(player_text="I force it", intent="force", approach="grit", chapter_title="T",
                           chapter_text="A room.", entering_chapter=False, facts=[], outcome="mixed",
                           obstacle_name="the iris", defeated_text="The iris grinds open.",
                           clue="The scratches are fresh.", beat=Beat.HAZARD,
                           beat_fact="Drones crawl from the vents.", ending="dead")
    text = v.narrate(req)
    for x in ("The iris grinds open.", "The scratches are fresh.", "Drones crawl from the vents.",
              "The iris groans and shifts"):
        assert x in text, text
    assert text.endswith("and the station keeps you.")


@pytest.mark.slow
def test_g11_two_hundred_games(scenario, director_model):
    bad = []
    for seed in range(200):
        e, played = play(scenario, director_model, 1000 + seed, turns=160)
        bad += audit(scenario, played)
    for seed in range(40):
        liar = Liar(seed)
        e, played = play(scenario, director_model, 2000 + seed, turns=160, llm=liar)
        bad += audit(scenario, played, llm=True) + [("leak", x) for x in _leaked(liar, played)]
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
        assert audit(scenario, played) == []
