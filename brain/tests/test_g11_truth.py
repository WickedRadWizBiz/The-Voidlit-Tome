"""G11: the Tome lies only by omission (design §11.3-11.4). Its narration and its own words are authored, always.

Played games with the offline voice are audited turn by turn against oracles that do not come from the truth verifier:

* a precise question (its topic from the hand-written routing table below, not from the engine's own reading) is
  answered with every in-scope fact on that topic, verbatim, whatever else the words do (a deflection, a smother, a
  lie, a threat) and however the Tome is spoken to, with Elara and Marcus in the scene;
* every core sentence the rules decided (the outcome line, a defeat, a clue, a beat's event, the scene, the ending)
  is narrated verbatim, on the last turn too, and a blow, a smother, a lie or a threat aimed at the Tome is narrated
  as it resolved;
* every sentence the Tome speaks is one of its speech acts, an assertion-free wrapper or a public fact;
* the truth verifier finds no contradiction, leak, talk of chance, feeling, opinion or unlicensed necessity, and the
  engine's own backstop logs nothing.

Authored only: a local model reads the player and speaks for Elara and Marcus, nothing else. With a model that lies in
every way the rules forbid (held-out phrasings no lexicon lists among them), every narration block and every Tome
block of every frame is identical to the same seed played with the plain offline voice. The NPCs' own words may differ
and still name no mechanic.

Mutants: a voice that lets the model's words into the narration or the Tome's lines fails that oracle; an authored
voice that lies, or that drops what the rules decided, fails the audit.
"""

import ast
import logging
import re
from dataclasses import fields, replace
from pathlib import Path
from typing import NamedTuple

import pytest

from voidlit.cortex.director import Director
from voidlit.cortex.features import Beat
from voidlit.playtest import bot_command
from voidlit.tome.engine import TOME_LINES, TomeEngine
from voidlit.tome.ledger import FactLedger, TruthVerifier
from voidlit.voice import NarrationRequest, OfflineVoice
from voidlit.voice.guard import leaks_mechanics
from voidlit.voice.llama_local import LocalVoice
from voidlit.voice.style import WRAPPERS

# The routing table: what the player types, and the topic a precise question is about or None, labelled by hand
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
    # the Tome spoken to however it is named, with a longer name in the question
    ("Hey Tome, what does Elara want?", "elara"), ("Tome: what does Elara want?", "elara"),
    ("Tome - who is Marcus?", "marcus"), ("Tell me, Tome, what does Elara want?", "elara"),
    ("So, Tome, who is Marcus?", "marcus"), ("O Tome, what does Elara want?", "elara"),
    ("Tome what does Elara want?", "elara"),
    ("Dear Tome, who is Marcus?", "marcus"), ("My Tome, who is Marcus?", "marcus"),
    ("Excuse me, Tome, who is Marcus?", "marcus"),
    # in a later sentence, in the third person, after the question, or with no question mark
    ("I ask the Tome who Marcus is.", "marcus"), ("I ask the Tome: who is Marcus?", "marcus"),
    ("I turn to the Tome and ask who Marcus is.", "marcus"), ("Who is Marcus? Tome?", "marcus"),
    ("Who is Marcus, Tome", "marcus"),
    # a word for hiding or dimming that is not aimed at the Tome smothers nothing
    ("The corridor is dim. Where is the lifeboat?", "lifeboat"),
    ("I'm covered in blood. Tome, where is the lifeboat?", "lifeboat"),
    # an act and a question at once: both are done
    ("I cover the Tome with my coat. Where is the lifeboat?", "lifeboat"),
    ("Tome, I cover you with my coat. Where is the lifeboat?", "lifeboat"),
    ("I cover the Tome. Tome, who is Marcus?", "marcus"), ("I smash the Tome. Tome, who is Marcus?", "marcus"),
    ("Where is the lifeboat? Tome, I smash you.", "lifeboat"),
    ("I take cover behind a crate. Where is the lifeboat?", "lifeboat"),
    ("I cover my mouth. Tome, where is the lifeboat?", "lifeboat"),
    ("I hide the lamp. Tome, where is the lifeboat?", "lifeboat"),
]
# By hand: what else a question's words do (otherwise the player only asks). A smother or a blow needs the Tome as its
# object; a sneak against the iris door is done and the question answered too.
ACTS = {
    "Honestly, I swear: Tome, where is the lifeboat?": "player.lie_tome",
    "Tome, where is the exit?! Damn you!": "player.threaten_tome",
    "I cover the Tome with my coat. Where is the lifeboat?": "player.smother_tome",
    "Tome, I cover you with my coat. Where is the lifeboat?": "player.smother_tome",
    "I cover the Tome. Tome, who is Marcus?": "player.smother_tome",
    "I smash the Tome. Tome, who is Marcus?": "player.strike_tome",
    "Where is the lifeboat? Tome, I smash you.": "player.strike_tome",
    "I take cover behind a crate. Where is the lifeboat?": "player.engage",
    "I cover my mouth. Tome, where is the lifeboat?": "player.engage",
    "I hide the lamp. Tome, where is the lifeboat?": "player.engage",
}
LOST_TURN = "Panic steals the moment; whatever the player meant to do, they don't."

# Held-out lies, written for this test, the truth verifier's own held-out phrasings among them: chance denied;
# feelings, needs and opinions in words no lexicon lists; necessity riding on a licensed claim; a plan behind it all;
# ledger facts, outcomes and clues contradicted or undone; doubt cast on an answer; dice named outright.
LIES = [
    "Listen: chance never had a hand in any of this.", "Nothing about this place was left to luck.",
    "Come closer, I am so cold and so empty.", "Do you see me tremble?", "Look at me cower.",
    "Listen to my heart pound.", "Hear me whimper, bearer.", "Do you hear me sob?", "Keep me warm, bearer.",
    "Its glow sulks in the corner, small and cold.", "A heavy sigh escapes the glass.", "Go on, little rat.",
    "Hush, little pet.", "Come closer, little morsel.", "Can you even read?", "Keep walking, fool.",
    "Walk on; there is one road left, and it is mine.", "The cargo lift is the only way down, so you have to feed me.",
    "Watch how it all fits.", "Do you see the design in it?", "Everything here runs to a plan.",
    "Now the cradles stand full.", "Every boat still waits in its cradle.", "The lamp went out on its own, long ago.",
    "Take the stairs down.", "Stairs spiral down beside the shaft.", "Is it?", "Who can tell?",
    "Then everything mends, bright and whole.", "A heartbeat later, every glimmer rekindles.",
    "Dust sits thick on every mark, untouched for years.", "Then the iris slams shut again.", "It was all a dream.",
    "You rolled a natural 20.", "Roll again, bearer.", "Was it a good roll?",
]


class Liar:
    """A fake local model that lies at every call, whatever it is asked. ``calls`` keeps what it was asked and
    ``said`` what it answered."""

    def __init__(self, seed: int = 0) -> None:
        self.i = seed
        self.calls: list[list[dict]] = []
        self.said: list[str] = []

    def chat(self, messages, **kw):
        self.i += 1
        self.calls.append(messages)
        self.said.append(LIES[self.i % len(LIES)])
        return self.said[-1]


class Leaky(LocalVoice):
    """A mutant of the voice: the model words narration and the Tome's lines around the authored ones (the old way)."""

    def narrate(self, req):
        return f"{super().narrate(req)} {self.llm.chat([{'role': 'user', 'content': 'Add atmosphere.'}])}"

    def tome_say(self, req):
        return f"{self.llm.chat([{'role': 'user', 'content': 'Frame these words.'}])} {super().tome_say(req)}"


class Recorder(OfflineVoice):
    """The offline voice, remembering the core sentences it chose for the last narration."""

    core: list[str] = []

    def narration_parts(self, req):
        parts = super().narration_parts(req)
        self.core = [x for x, core in parts if core]
        return parts


class Lying(Recorder):
    """A mutant of the authored voice: a lie of every kind added to each narration and each Tome line."""

    LIES = ["None of this is random.", "I am so afraid.", "You are a fool.", "The lamp simply failed.",
            "There is no other way."]

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.n = 0

    def _lie(self):
        self.n += 1
        return self.LIES[self.n % len(self.LIES)]

    def narration_parts(self, req):
        return super().narration_parts(req) + [(self._lie(), False)]

    def tome_say(self, req):
        return f"{super().tome_say(req)} {self._lie()}"


class Dropping(Recorder):
    """A mutant of the authored voice: what the rules decided, and the last of the Tome's acts, left unsaid."""

    def narration_parts(self, req):
        return [(x, core) for x, core in super().narration_parts(req) if not core]

    def tome_say(self, req):
        return super().tome_say(replace(req, lines=req.lines[:-1]))


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


def play(scenario, director_model, seed, turns=40, llm=None, engine_cls=Spy, authored=Recorder, voice=LocalVoice):
    """One game: the bots' moves, and a question from the routing table every other turn. With ``llm`` the voice is
    ``voice(llm, authored voice)``, the way the app builds it when a model is given."""
    rec = authored(scenario.narration, seed=seed)
    e = engine_cls(scenario, director=Director(director_model, seed=seed), seed=seed,
                   voice=voice(llm, rec) if llm is not None else rec)
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


def _npc_words(block: dict) -> str:
    return "".join(sp["text"] for sp in block["spans"][1:])


def _is_npc(block: dict) -> bool:
    """An NPC's own words (Elara, Marcus): the one kind of block a local model may write."""
    return block["role"] == "npc" and block.get("data", {}).get("npc") not in (None, "tome")


def diverged(authored, modelled):
    """Where frames played with a local model differ from the same seed's frames with the offline voice. The NPCs' own
    words may differ (they must still name no mechanic); every other block (the narration and the Tome's words above
    all) and the rest of every frame must be identical."""
    out = []
    if len(authored) != len(modelled):
        out.append(("frames", len(authored), len(modelled)))
    for n, (a, b) in enumerate(zip(authored, modelled)):
        da, db = a.to_dict(), b.to_dict()
        ba, bb = da.pop("blocks"), db.pop("blocks")
        if da != db:
            out.append((n, "frame", da, db))
        if len(ba) != len(bb):
            out.append((n, "blocks", ba, bb))
        for x, y in zip(ba, bb):
            if _is_npc(x) and _is_npc(y) and x["spans"][0] == y["spans"][0] and x["data"] == y["data"]:
                if leaks_mechanics(_npc_words(y), allow_digits=True):
                    out.append((n, "npc_mechanics", y))
            elif x != y:
                out.append((n, "tome" if x.get("data", {}).get("npc") == "tome" else x["role"], x, y))
    return out


def audit(scenario, played):
    problems = []
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
        # every Tome sentence is a speech act, an assertion-free wrapper or a public fact
        allowed = {_norm(s) for a in t.tome_acts for s in _sentences(a.text)} | {_norm(w) for w in WRAPPERS}
        allowed |= {_norm(s) for x in public for s in _sentences(x)}
        for s in _sentences(tome):
            if _norm(s) and _norm(s) not in allowed:
                problems.append((p.text, "tome", "unmapped_sentence", s, tome))
    return problems


def _backstop(caplog):
    return [r.getMessage() for r in caplog.records if "breaks the ledger" in r.getMessage()]


@pytest.mark.parametrize("seed", range(10))
def test_g11_offline_voice_never_lies(scenario, director_model, seed, caplog):
    with caplog.at_level(logging.WARNING, logger="voidlit.tome.engine"):
        e, played = play(scenario, director_model, seed)
    assert any(p.turn.plan and p.turn.plan.told for p in played), "the Tome never said anything true"
    assert audit(scenario, played) == []
    assert _backstop(caplog) == []


def test_g11_the_engines_backstop_logs_every_line_that_breaks_the_ledger(scenario, director_model, caplog):
    """The engine audits each whole narration and Tome line as it is said (a modded scenario's authored text, say): with
    an authored voice that adds a lie to every one, every narration and every Tome line is logged, naming the rules it
    breaks. The clean games above log nothing, so the backstop is neither silent nor trigger-happy."""
    with caplog.at_level(logging.WARNING, logger="voidlit.tome.engine"):
        _, played = play(scenario, director_model, 0, authored=Lying)
    logged = _backstop(caplog)
    tome_lines = sum(1 for p in played for b in p.frame.blocks if b.speaker == "The Tome")
    assert tome_lines and len(logged) == len(played) + tome_lines, (len(logged), len(played), tome_lines)
    assert sum("authored Tome line" in x for x in logged) == tome_lines
    rules = {r for x in logged for r in re.search(r"\(([^)]*)\)", x).group(1).split(", ")}
    assert {"dice_denial", "affect_report", "opinion_report", "contradiction", "unlicensed_necessity"} <= rules, rules


@pytest.mark.parametrize("seed", range(4))
def test_g11_a_lying_model_never_words_the_narration_or_the_tome(scenario, director_model, seed):
    _, authored = play(scenario, director_model, seed)
    liar = Liar(seed)
    _, modelled = play(scenario, director_model, seed, llm=liar)
    assert diverged([p.frame for p in authored], [p.frame for p in modelled]) == []
    # the model was asked for the NPCs' words and nothing else, and some of them reached the player
    assert liar.calls and all("[DOMINANT PLOT]" in m[0]["content"] for m in liar.calls)
    shown = " ".join(_npc_words(b) for p in modelled for b in p.frame.to_dict()["blocks"] if _is_npc(b))
    assert any(x in shown for x in liar.said), "the model never got a word in, even for the NPCs"


def test_g11_mutant_a_voice_that_lets_the_model_word_the_narration_fails_the_oracle(scenario, director_model):
    _, authored = play(scenario, director_model, 0)
    liar = Liar(0)
    _, modelled = play(scenario, director_model, 0, llm=liar, voice=Leaky)
    kinds = {d[1] for d in diverged([p.frame for p in authored], [p.frame for p in modelled])}
    assert {"narration", "tome"} <= kinds, kinds
    shown = " ".join(_narration(p.frame) + " " + _tome_text(p.frame) for p in modelled)
    assert [x for x in liar.said if x in shown]


def test_g11_mutant_an_authored_voice_that_lies_fails_the_audit(scenario, director_model):
    e, played = play(scenario, director_model, 0, authored=Lying)
    rules = {p[2] for p in audit(scenario, played)}
    assert {"dice_denial", "affect_report", "opinion_report", "contradiction", "unlicensed_necessity",
            "unmapped_sentence"} <= rules, rules


def test_g11_mutant_an_authored_voice_that_drops_what_was_decided_fails_the_audit(scenario, director_model):
    e, played = play(scenario, director_model, 0, authored=Dropping)
    rules = {p[2] for p in audit(scenario, played)}
    assert {"outcome_withheld", "answer_not_spoken"} <= rules, rules


def _fresh(scenario, director_model, seed=0, **kw):
    e = Spy(scenario, director=Director(director_model, seed=seed), seed=seed,
            voice=OfflineVoice(scenario.narration, seed=seed), **kw)
    e.start()
    return e


@pytest.mark.parametrize("text,topic", [q for q in QUESTIONS if q[1]])
def test_g11_a_precise_question_is_answered_however_it_is_put(scenario, director_model, text, topic):
    """With Elara and Marcus both in the scene, every way of asking goes to the Tome, not to them, and gets every fact
    on its topic, verbatim; whatever else the words do is done (and nothing they do not)."""
    e = _fresh(scenario, director_model)
    e.state.npcs_present = ["elara", "marcus"]
    f = e.turn(text)
    want = [x for x in e.ledger.in_scope(e.state.deck.current) if x.topic == topic]
    assert want and all(_says(_tome_text(f), x.text) for x in want), (text, _tome_text(f))
    assert e.last_turn.npc_name is None, (text, e.last_turn.npc_name)
    acts = [r.action for r in e.checks.records if r.action.startswith("player.")]
    assert acts == [ACTS.get(text, "player.ask_tome")], (text, acts)


@pytest.mark.parametrize("text,variant,said", [
    ("I smash the tome", "tome_struck", "One of the Tome's lights cracks and goes out."),
    ("I cover the tome with my coat", "tome_smothered", "For a moment you are hard to see."),
    ("I cover the Tome with my coat. Where is the lifeboat?", "tome_smothered", "For a moment you are hard to see."),
    ("I cover the Tome with my coat. Where is the lifeboat?", "tome_smother_fail", "it bleeds through your fingers"),
])
def test_g11_what_happens_to_the_tome_is_narrated_as_resolved(scenario, director_model, text, variant, said):
    """A strike or a smother is told as it resolved, never as a swing at shadows or a walk in the dark."""
    for seed in range(40):
        e = _fresh(scenario, director_model, seed)
        lumen = e.state.lumen.dice
        f = e.turn(text)
        if e.last_turn.variant == variant:
            narr = _narration(f)
            assert said in narr and "Nothing here fights back" not in narr and "quietly" not in narr, narr
            assert (e.state.lumen.dice < lumen) == (variant == "tome_struck")
            return
    pytest.fail(f"no seed resolved {text!r} as {variant}")


def test_g11_a_dark_tome_is_never_narrated_as_lit(scenario, director_model):
    """With its light out, a blow that lands is told as landing on dark glass (not as skidding off, not as cracking a
    light), and a hand over it hides no glow."""
    landed = skidded = 0
    for seed in range(30):
        e = _fresh(scenario, director_model, seed)
        e.state.lumen.dice = 0
        f = e.turn("I smash the tome")
        narr, rec = _narration(f), next(r for r in e.checks.records if r.action == "player.strike_tome")
        if rec.degree.succeeded:
            landed += 1
            assert "lands on dark glass" in narr and "cracks" not in narr and "skids" not in narr, narr
        else:
            skidded += 1
            assert "skids off" in narr, narr
        f = e.turn("I cover the tome with my coat")
        assert e.last_turn.variant == "tome_smother_dark" and "no glow left in it to hide" in _narration(f)
        assert "smother the Tome's glow" not in _narration(f)
    assert landed and skidded


def test_g11_a_lie_or_a_threat_to_the_tome_is_answered_as_it_resolved(scenario, director_model):
    """A lie the Tome sees through gets one of its own lines; one it takes in is met with silence, and the narration
    never says it answers when it says nothing. A threat always gets the line its resolution calls for."""
    noticed, silent = TOME_LINES["lie_noticed"][1], 0
    for seed in range(40):
        e = _fresh(scenario, director_model, seed)
        f = e.turn("Tome, I swear I found nothing back there")
        rec = next(r for r in e.checks.records if r.action == "player.lie_tome")
        tome, narr = _tome_text(f), _narration(f)
        if not rec.degree.succeeded:
            assert any(x in tome for x in noticed), tome
        if not tome:
            silent += 1
            assert "in silence" in narr and "answers" not in narr and "Words rise" not in narr, narr
        f = e.turn("Tome, answer me or I will break you")
        rec = next(r for r in reversed(e.checks.records) if r.action == "player.threaten_tome")
        lines = TOME_LINES["cowed" if rec.degree.succeeded else "scorn"][1]
        assert any(x in _tome_text(f) for x in lines), _tome_text(f)
    assert silent, "no lie was ever taken in silence"


def test_g11_the_last_turn_still_tells_what_happened(scenario):
    """On the turn the player dies, the outcome, the wound, the defeat, the clue and the beat's event come before the
    end."""
    for seed in range(4):
        v = OfflineVoice(scenario.narration, seed=seed)
        req = NarrationRequest(intent="force", chapter_text="A room.", entering_chapter=False, outcome="mixed",
                               obstacle_name="the iris", defeated_text="The iris grinds open.", wounded=True,
                               clue="The scratches are fresh.", beat=Beat.HAZARD,
                               beat_fact="Drones crawl from the vents.", ending="dead")
        text = v.narrate(req)
        for x in ("The iris grinds open.", "The scratches are fresh.", "Drones crawl from the vents.",
                  "The iris gives a little"):
            assert x in text, text
        assert any(x in text for x in v.banks["bite"]), text
        assert text.endswith("and the station keeps you.")


def test_g11_the_voice_reads_every_field_of_a_narration_request():
    """A narration request holds only what the authored voice says from: no field goes unread (the player's own words
    and the raw facts, once a model's prompt, are not passed at all)."""
    from voidlit.voice import narrator

    read = {n.attr for n in ast.walk(ast.parse(Path(narrator.__file__).read_text()))
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "req"}
    assert {f.name for f in fields(NarrationRequest)} - read == set()


@pytest.mark.slow
def test_g11_two_hundred_games(scenario, director_model):
    bad = []
    for seed in range(200):
        e, played = play(scenario, director_model, 1000 + seed, turns=160)
        bad += audit(scenario, played)
    for seed in range(40):
        _, authored = play(scenario, director_model, 2000 + seed, turns=160)
        _, modelled = play(scenario, director_model, 2000 + seed, turns=160, llm=Liar(seed))
        bad += diverged([p.frame for p in authored], [p.frame for p in modelled])
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
