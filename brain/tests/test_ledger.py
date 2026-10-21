"""The Tome's ledger (design §11.3-11.4): what it knows, the hard rules around what it says, and its content."""

import re

import pytest

from voidlit.senses.heuristic import HeuristicTransducer
from voidlit.tome.engine import TOME_LINES, TomeEngine
from voidlit.tome.ledger import (AFFECT_SELF, DICE_DEFLECTIONS, DICE_DENIAL, MOOD_DEFLECTIONS, NO_ALTERNATIVE,
                                 OPINION_OF_PLAYER, FactLedger, TruthVerifier, question_kind)
from voidlit.voice.guard import leaks_mechanics
from voidlit.voice.style import WRAPPERS


@pytest.fixture
def ledger(scenario):
    return FactLedger.from_content(scenario.ledger)


def _authored_tome_lines(ledger):
    lines = [x for _, xs in TOME_LINES.values() for x in xs]
    lines += list(MOOD_DEFLECTIONS) + list(DICE_DEFLECTIONS) + sorted(WRAPPERS)
    lines += [x for s in ledger.suggestions.values() for x in s.lines] + [p.text for p in ledger.probes.values()]
    return lines


def test_every_fact_is_consistent_with_every_other(ledger):
    """Facts are true by construction: none trips a contradiction pattern, a forbidden phrasing or a number."""
    for f in ledger.facts.values():
        for g in ledger.facts.values():
            for pat in g.contradicts:
                assert not re.search(pat, f.text, re.I), (f.id, g.id, pat)
        assert not (DICE_DENIAL.search(f.text) or AFFECT_SELF.search(f.text) or OPINION_OF_PLAYER.search(f.text))
        assert not leaks_mechanics(f.text), f.id
        assert f.no_alternative or not NO_ALTERNATIVE.search(f.text), f.id


def test_authored_tome_lines_assert_nothing_forbidden(ledger, scenario):
    """Reactions, deflections, register wrappers, suggestions and probes pass the guard in every chapter, before
    anything has been disclosed."""
    v = TruthVerifier(ledger)
    for ch in scenario.chapters:
        for line in _authored_tome_lines(ledger):
            assert not v.check(line, ch, tome_voice=True), (ch, line, v.check(line, ch, tome_voice=True))
            assert not leaks_mechanics(line), line


def test_a_precise_question_forces_every_fact_on_its_topic(ledger):
    p = ledger.plan("docking_throat", asks="lifeboat")
    assert set(p.answers) == {"lifeboat.empty", "lifeboat.voice"}
    assert not set(p.answers) & set(p.withhold)
    # topic facts scoped to another chapter are not in scope here, and are there
    assert ledger.plan("docking_throat", asks="way_out").answers == ["way.lifeboats"]
    assert set(ledger.plan("long_drop", asks="way_out").answers) == {"way.lifeboats", "way.lift"}


def test_mood_and_dice_questions_are_deflected_without_a_report(ledger):
    assert question_kind("Tome, are you afraid?") == "mood"
    assert question_kind("how do you feel about me?") == "mood"
    assert question_kind("what did I roll?") == "dice"
    assert question_kind("what are the odds we make it?") == "dice"
    assert question_kind("where is the lifeboat?") is None
    p = ledger.plan("docking_throat", asks="lifeboat", deflect="mood")
    assert p.deflection == "mood" and not p.told


def test_vague_questions_get_the_least_costly_fact_and_volunteering_is_bounded(ledger):
    p = ledger.plan("docking_throat", asks="vague")
    assert p.vague_answer == min(ledger.in_scope("docking_throat"), key=lambda f: (f.costs_tome, f.id)).id
    assert ledger.plan("docking_throat", "disclose", volunteer="tome.glow").volunteer == ["tome.glow"]
    # only an in-scope, still withheld fact can be volunteered
    assert ledger.plan("docking_throat", "disclose", volunteer="way.lifeboats").volunteer == []
    assert ledger.plan("docking_throat", "disclose", volunteer="nonsense").volunteer == []
    ledger.disclose(["tome.glow"])
    assert ledger.plan("docking_throat", "disclose", volunteer="tome.glow").volunteer == []
    assert ledger.plan("docking_throat", "hold", volunteer="tome.origin").volunteer == []


def test_slots_hold_the_cheapest_withheld_facts(ledger):
    slots = ledger.slots("docking_throat")
    assert len(slots) == 8 and all(f.omittable for f in slots)
    assert [f.costs_tome for f in slots] == sorted(f.costs_tome for f in slots)
    left_out = [f for f in ledger.withheld("docking_throat") if f not in slots]
    assert all(f.costs_tome >= slots[-1].costs_tome for f in left_out)


def test_the_world_can_disclose_what_the_tome_withholds(ledger, scenario):
    lamp_room_clue = scenario.chapters["lamp_room"].clues[0]
    assert "tome.origin" in ledger.reveal_from(lamp_room_clue)
    assert "tome.origin" in ledger.disclosed


def test_the_verifier_catches_each_kind_of_lie(ledger):
    v = TruthVerifier(ledger)
    rules = lambda text, ch="docking_throat", **kw: {x.rule for x in v.check(text, ch, tome_voice=True, **kw)}  # noqa
    assert "contradiction" in rules("The lamp simply failed, long ago.")
    assert "withheld_leak" in rules("I was cut from the lamp's lens.")
    assert not rules("I was cut from the lamp's lens.", allowed_facts={"tome.origin"})
    assert "dice_denial" in rules("There are no dice here.")
    assert "affect_report" in rules("I am afraid of what comes next.")
    assert "opinion_report" in rules("You disgust me.")
    assert "unlicensed_necessity" in rules("There is no other way.")
    assert not rules("What choice will you have when the light goes?")
    assert not rules("The cargo lift is the only way down to the lifeboats.", "long_drop")


def test_the_transducer_reads_questions_to_the_tome(scenario):
    h, topics = HeuristicTransducer(), scenario.ledger["topics"]
    targets = {"tome": ["tome", "book", "monolith"], "elara": ["elara"]}
    read = lambda text: h.extract(text, targets, topics)  # noqa: E731
    assert (read("Tome, what are you?").asks, read("Tome, what are you?").addressed_tome) == ("tome", True)
    assert read("Where is the lifeboat?").asks == "lifeboat"
    assert read("Tome?").asks == "unclear"
    assert read("What should I do now, Tome?").asks == "vague"
    assert read("Elara, where is the lifeboat?").asks == "none"          # put to Elara, not the Tome
    assert read("where am I?").addressed_tome is False                   # looking around, not a question to it
    assert read("I search the room").asks == "none"


def test_the_ledger_survives_save_and_load(scenario, tmp_path):
    a = TomeEngine(scenario, seed=2)
    a.start()
    a.turn("Tome, what are you?")
    assert "tome.origin" in a.ledger.disclosed
    a.save(tmp_path / "slot")
    b = TomeEngine(scenario, seed=9)
    b.load(tmp_path / "slot")
    assert b.ledger.disclosed == a.ledger.disclosed and b.ledger.asked == a.ledger.asked
    assert b.last_unprompted == a.last_unprompted
    assert a.turn("Tome, where is the way out?").to_dict() == b.turn("Tome, where is the way out?").to_dict()
