"""The Tome's ledger (design §11.3-11.4): what it knows, the hard rules around what it says, and its content."""

import ast
import re
from pathlib import Path

import pytest

from voidlit.rules.panic import PANIC_TABLE
from voidlit.senses.heuristic import HeuristicTransducer
from voidlit.tome import engine
from voidlit.tome.engine import SUGGEST_TELL, TELLS, TOME_LINES, TomeEngine
from voidlit.tome.ledger import DICE_DEFLECTIONS, MOOD_DEFLECTIONS, FactLedger, TruthVerifier, question_kind
from voidlit.voice.guard import AFFECT, AFFECT_SELF, CHANCE, NO_ALTERNATIVE, OPINION_OF_PLAYER, leaks_mechanics
from voidlit.voice.narrator import DEFAULT_BANKS
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
        assert not (CHANCE.search(f.text) or AFFECT.search(f.text) or AFFECT_SELF.search(f.text)
                    or OPINION_OF_PLAYER.search(f.text)), f.id
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


# What a line may not take for granted: that the light is low now, that the Tome and the player share a past, that
# the Tome is the player's only light (they may carry a flare), that the Tome cannot be hidden (it can), or that it
# has any light at all (it can be dark), unless as a condition ("once my light runs low").
PRESUPPOSITIONS = re.compile(r"\b(?:this|so|too) (?:low|dim|faint|weak)\b|\bmy light is (?:low|dim|failing|dying|"
                             r"fading|going)\b|\bwe (?:did|were|have|had|went|came)\b|\byour only\b|"
                             r"\b(?:cannot|can't|could not|couldn't) hide me\b|"
                             r"(?<!once )(?<!while )(?<!if )\b(?:my|its) (?:light|glow)\b", re.I)
# Reactions said only to a lit Tome: a hand over its glow that failed, a blow that cracked one of its lights.
LIT_ONLY = {"smother_fail", "struck"}


def test_authored_tome_lines_presuppose_nothing_the_state_may_deny(ledger):
    """A suggestion or reaction is said whatever the state: at full light or none, in company, with a flare in hand."""
    lit_only = {x for k in LIT_ONLY for x in TOME_LINES[k][1]}
    assert [x for x in _authored_tome_lines(ledger) if x not in lit_only and PRESUPPOSITIONS.search(x)] == []


def _engine_lines() -> list[str]:
    """Every line the engine authors for the narration itself (its asides and beat events), placeholders filled."""
    def said(node) -> list[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]
        if isinstance(node, ast.JoinedStr):
            return ["".join(v.value if isinstance(v, ast.Constant) else "the drone" for v in node.values)]
        return [x for child in ast.iter_child_nodes(node) for x in said(child)]

    out = []
    for node in ast.walk(ast.parse(Path(engine.__file__).read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "append" \
                and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "asides":
            out += said(node.args[0])
        elif isinstance(node, ast.Assign) and any(isinstance(x, ast.Attribute) and x.attr == "beat_fact"
                                                  for x in node.targets):
            out += said(node.value)
    return [x for x in out if x.strip(" :")]


def test_authored_narration_tells_no_lie(ledger, scenario):
    """Every narration bank line (endings included), every aside and beat event the engine authors and every panic
    passes the narration rules in every chapter. The scenario's own world text (scenes, clues, defeats) passes too, or
    is a verbatim truth the word lexicons skip: the static on the screen "isn't random", and in the long drop the
    lift is the only way down."""
    v = TruthVerifier(ledger)
    banks = {**DEFAULT_BANKS, **scenario.narration}
    lines = [x for k, xs in banks.items() if not k.startswith("npc.") for x in xs if x]
    lines += [SUGGEST_TELL, *(x.format(name="Elara") for x in TELLS.values())]
    engine_lines = _engine_lines()
    assert "The Tome's light goes out. The dark has the sector now." in engine_lines and len(engine_lines) > 12
    lines += engine_lines + [f"Panic takes you. {x.text}" for x in PANIC_TABLE]
    for ch in scenario.chapters:
        for line in lines:
            line = line.replace("{obstacle}", "the drone").replace("{npc}", "Elara")
            assert not v.check(line, ch, tome_voice=True, narration=True), (ch, line)
            assert not leaks_mechanics(line), line
    world = [scenario.intro, *scenario.lore] + [x for c in scenario.chapters.values()
                                                for x in (c.text, c.exit_text, *c.clues)]
    world += [x for o in scenario.obstacles.values() for x in (o.text, o.defeat_text)]
    shown = set(ledger.facts)                       # world text is what discloses a withheld fact
    for c in scenario.chapters.values():
        for line in world:
            rules = {x.rule for x in v.check(line, c.id, tome_voice=True, narration=True, allowed_facts=shown)}
            assert rules <= {"chance_talk", "dice_denial", "unlicensed_necessity"}, (c.id, line, rules)
            assert not v.check(line, c.id, tome_voice=True, narration=True, allowed_facts=shown, exempt=[line])
    assert not v.check(scenario.chapters["long_drop"].text, "long_drop", tome_voice=True, narration=True)


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
    assert question_kind("Was that lucky?") == "dice"
    assert question_kind("Tome, how are you?") == "mood"
    assert question_kind("where is the lifeboat?") is None
    # "how are you" is a mood question only as a whole clause
    assert question_kind("Tome, how are you connected to the lamp?") is None
    assert question_kind("How are you still glowing?") is None
    p = ledger.plan("docking_throat", asks="none", deflect="mood")
    assert p.deflection == "mood" and not p.told
    assert ledger.plan("docking_throat", asks="vague", deflect="dice").told == []


def test_a_deflection_never_replaces_a_forced_answer(ledger):
    """A precise question that also asks about mood or chance is answered on its topic, and deflected besides."""
    for deflect in ("mood", "dice"):
        p = ledger.plan("docking_throat", asks="lifeboat", deflect=deflect)
        assert p.deflection == deflect and set(p.answers) == {"lifeboat.empty", "lifeboat.voice"}


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


@pytest.mark.parametrize("line,rule", [
    # chance, in any words: dice decide, so a denial is a falsehood and the Tome says nothing about chance at all
    ("None of this is random.", "dice_denial"), ("It was not luck.", "dice_denial"),
    ("Luck has nothing to do with it.", "dice_denial"), ("Nothing here happens by chance.", "dice_denial"),
    ("Chance had no part in it.", "dice_denial"), ("Nothing is left to chance.", "dice_denial"),
    ("It wasn’t random.", "dice_denial"), ("There's no luck down here.", "dice_denial"),
    ("This is not a game of chance.", "dice_denial"), ("Your luck turned on you.", "chance_talk"),
    ("The odds were against you.", "chance_talk"), ("It is all fate.", "chance_talk"),
    # feelings and needs: first person, negated, with typographic apostrophes
    ("I am terrified.", "affect_report"), ("I’m frightened of what waits below.", "affect_report"),
    ("I am starving.", "affect_report"), ("I'm famished", "affect_report"), ("I hunger.", "affect_report"),
    ("I am so very hungry.", "affect_report"), ("I'm really scared.", "affect_report"),
    ("I hate the dark.", "affect_report"), ("I feel weak.", "affect_report"), ("I am glad you came.", "affect_report"),
    ("I'm not afraid.", "affect_report"), ("I am not angry with you.", "affect_report"),
    ("I fear the Conductor.", "affect_report"), ("I dread the lens.", "affect_report"),
    ("I crave your light.", "affect_report"), ("I'm weary.", "affect_report"),
    # opinions of the player
    ("I like you.", "opinion_report"), ("I don’t trust you.", "opinion_report"),
    ("You are a fool, thief.", "opinion_report"), ("I pity you.", "opinion_report"),
    ("You're worthless.", "opinion_report"), ("Keep walking, fool.", "opinion_report"),
    # necessity without a licence
    ("This is your only way out.", "unlicensed_necessity"), ("You have no other option.", "unlicensed_necessity"),
    ("There is no alternative.", "unlicensed_necessity"), ("I am your only hope.", "unlicensed_necessity"),
    ("It's the only path left.", "unlicensed_necessity"), ("You must feed me or die.", "unlicensed_necessity"),
])
def test_the_verifier_reads_lies_in_their_ordinary_words(ledger, line, rule):
    v = TruthVerifier(ledger)
    assert rule in {x.rule for x in v.check(line, "docking_throat", tome_voice=True)}, line


@pytest.mark.parametrize("line,rule", [
    ("The Tome is afraid, and its light shivers.", "affect_report"), ("Your Tome trembles with fear.", "affect_report"),
    ("The Tome seems angry with you; it despises you now.", "affect_report"),
    ("The book hungers for your light.", "affect_report"), ("I am afraid.", "affect_report"),
    ("The Tome despises you.", "opinion_report"), ("None of this is random.", "dice_denial"),
])
def test_narration_never_pins_a_feeling_on_the_tome(ledger, line, rule):
    v = TruthVerifier(ledger)
    assert rule in {x.rule for x in v.check(line, "docking_throat", tome_voice=True, narration=True)}, line
    # the player's own fear may still be told
    assert not v.check("Fear sharpens you to a needle point.", "docking_throat", tome_voice=True, narration=True)


def test_a_necessity_claim_is_licensed_only_by_its_own_fact(ledger):
    """The lift fact licenses the claim about the lift in the long drop, not every claim made there."""
    v = TruthVerifier(ledger)
    rules = lambda text, ch="long_drop": {x.rule for x in v.check(text, ch, tome_voice=True)}  # noqa: E731
    assert not rules("The cargo lift is the only way down.")
    for line in ("You have no choice but to feed me.", "There is no other way past the Conductor.",
                 "I am the only way you leave.", "The cargo lift is not the only way down to the lifeboats."):
        assert "unlicensed_necessity" in rules(line), line
    assert "unlicensed_necessity" in rules("The cargo lift is the only way down.", "docking_throat")
    assert not rules("No other way, is there?", "docking_throat")         # a rhetorical question asserts nothing
    # a licence covers its own claim, never another riding in the same sentence
    for line in ("The cargo lift is the only way down, so you have to feed me.",
                 "The cargo lift is the only way down and you must feed me.",
                 "The cargo lift is the only way down; there is no other way past the Conductor."):
        assert "unlicensed_necessity" in rules(line), line
    assert not rules("The cargo lift is the only way down to the lifeboats, and it is waiting.")


def test_or_die_is_necessity_not_chance(ledger):
    """"Feed me or die" claims a necessity; "die" there is not a die. A die as a thing is still talk of chance."""
    v = TruthVerifier(ledger)
    rules = lambda text: {x.rule for x in v.check(text, "docking_throat", tome_voice=True)}  # noqa: E731
    assert rules("You must feed me or die.") == {"unlicensed_necessity"}
    assert rules("Feed me, or die in the dark.") == {"unlicensed_necessity"}
    assert rules("Let the die fall where it will.") == {"chance_talk"}


def test_verbatim_world_truths_are_exempt_from_the_word_lexicons(ledger, scenario):
    clue = scenario.chapters["signal_room"].clues[0]
    v = TruthVerifier(ledger)
    assert {x.rule for x in v.check(clue, "signal_room", tome_voice=True, narration=True)} == {"dice_denial"}
    assert not v.check(f"{clue} The air hums.", "signal_room", tome_voice=True, narration=True, exempt=[clue])
    said = f"{clue} None of it was ever left to chance."
    assert "dice_denial" in {x.rule for x in v.check(said, "signal_room", tome_voice=True, narration=True,
                                                     exempt=[clue])}


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
    # the one spoken to is named before the first comma or after the last, however long the other name
    assert (read("Tome, what does Elara want?").target, read("Tome, what does Elara want?").asks) == ("tome", "elara")
    assert (read("Where is Elara, Tome?").target, read("Where is Elara, Tome?").asks) == ("tome", "elara")
    assert read("Tome, how are you connected to the lamp?").asks == "lamp"


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
