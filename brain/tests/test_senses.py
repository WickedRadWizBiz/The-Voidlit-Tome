import json

import pytest

from voidlit.rules import Approach
from voidlit.senses import (HeuristicTransducer, Intent, SensoryParseError, SensoryState, build_sensory_gbnf,
                            parse_sensory_json)

TARGETS = {"elara": ["elara", "voss"], "marcus": ["marcus", "warden"]}


def test_grammar_locks_targets_and_enums():
    g = build_sensory_gbnf(["Elara", "marcus", "bad id!"])
    assert 'target ::= "none" | "bad_id_" | "elara" | "marcus"' in g
    for intent in Intent:
        assert f'"{intent.value}"' in g
    assert 'unit ::= ("0." [0-9] [0-9]?) | "1.0"' in g
    assert g.startswith("root ::=")


def test_parse_valid_output_is_clamped():
    out = ('{"intent": "talk", "approach": "guile", "target": "elara", "aggression": 0.10, "deception": 0.9, '
           '"leverage": 0.0, "exposure": 0.25, "rapport": -0.5, "boldness": 1.0}')
    s = parse_sensory_json("noise " + out + " trailing", TARGETS)
    assert s.intent is Intent.TALK and s.approach is Approach.GUILE and s.target == "elara"
    assert s.deception == pytest.approx(0.9) and s.rapport == pytest.approx(-0.5)


def test_parse_rejects_off_schema_and_unknown_targets():
    with pytest.raises(SensoryParseError):
        parse_sensory_json("no json here")
    with pytest.raises(SensoryParseError):
        parse_sensory_json('{"intent": "dance", "approach": "grit", "aggression": 0, "deception": 0, '
                           '"leverage": 0, "exposure": 0, "rapport": 0, "boldness": 0}')
    d = {"intent": "look", "approach": "none", "target": "ghost", "aggression": 0, "deception": 0,
         "leverage": 0, "exposure": 0, "rapport": 3.0, "boldness": 0}
    s = parse_sensory_json(json.dumps(d), TARGETS)
    assert s.target == "none" and s.rapport == 1.0


@pytest.mark.parametrize("text,intent,approach,target", [
    ('"Elara, are you hurt? We get out together."', Intent.TALK, Approach.NONE, "elara"),
    ("I shoot the drone!", Intent.FIGHT, Approach.GUILE, "none"),
    ("I pry the hatch open", Intent.FORCE, Approach.GRIT, "none"),
    ("I carefully sneak past", Intent.SNEAK, Approach.GUILE, "none"),
    ("I analyze the console", Intent.SEARCH, Approach.INSIGHT, "none"),
    ("press on", Intent.ADVANCE, Approach.NONE, "none"),
])
def test_heuristic_intents(text, intent, approach, target):
    s = HeuristicTransducer().extract(text, TARGETS)
    assert (s.intent, s.approach, s.target) == (intent, approach, target)


def test_heuristic_psychology():
    h = HeuristicTransducer()
    probe = h.extract("Marcus, what are you hiding? I saw you with the Choir.", TARGETS)
    assert probe.exposure > 0.6 and probe.target == "marcus"
    threat = h.extract("Tell me where the codes are or else I kill you, Elara!", TARGETS)
    assert threat.aggression > 0.6 and threat.rapport < 0
    warm = h.extract("Thank you, friend. I trust you.", TARGETS)
    assert warm.rapport > 0.5 and warm.aggression == 0


def test_npc_channels_split_rapport():
    s = SensoryState(rapport=-0.4, aggression=0.2)
    assert s.npc_channels() == [0.2, 0.0, 0.0, 0.0, 0.0, 0.4]
    assert s.shock == pytest.approx(0.4)


@pytest.mark.parametrize("text,target", [
    ("Tome, what does Elara want?", "tome"), ("Where is Elara, Tome?", "tome"), ("Tome, who is Marcus?", "tome"),
    ("Elara, where is Marcus?", "elara"), ("Is Marcus armed, Elara?", "elara"), ("I shoot Elara", "elara"),
    ("Marcus, I swear I am on your side", "marcus"),
])
def test_the_one_named_as_spoken_to_is_the_target(text, target):
    """A question is for the name before its first comma or after its last, however long another name is."""
    targets = {"tome": ["tome", "book", "monolith"], **TARGETS}
    assert HeuristicTransducer().extract(text, targets).target == target
