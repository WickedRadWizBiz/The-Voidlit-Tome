import json

import pytest

from voidlit.rules import Stat
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
    out = ('{"intent": "talk", "stat": "grace", "target": "elara", "aggression": 0.10, "deception": 0.9, '
           '"leverage": 0.0, "exposure": 0.25, "rapport": -0.5, "boldness": 1.0}')
    s = parse_sensory_json("noise " + out + " trailing", TARGETS)
    assert s.intent is Intent.TALK and s.stat is Stat.GRACE and s.target == "elara"
    assert s.deception == pytest.approx(0.9) and s.rapport == pytest.approx(-0.5)


def test_parse_rejects_off_schema_and_unknown_targets():
    with pytest.raises(SensoryParseError):
        parse_sensory_json("no json here")
    with pytest.raises(SensoryParseError):
        parse_sensory_json('{"intent": "dance", "stat": "might", "aggression": 0, "deception": 0, '
                           '"leverage": 0, "exposure": 0, "rapport": 0, "boldness": 0}')
    d = {"intent": "look", "stat": "none", "target": "ghost", "aggression": 0, "deception": 0,
         "leverage": 0, "exposure": 0, "rapport": 3.0, "boldness": 0}
    s = parse_sensory_json(json.dumps(d), TARGETS)
    assert s.target == "none" and s.rapport == 1.0


@pytest.mark.parametrize("text,intent,stat,target", [
    ('"Elara, are you hurt? We get out together."', Intent.TALK, Stat.NERVE, "elara"),
    ("I shoot the drone!", Intent.FIGHT, Stat.GRACE, "none"),
    ("I pry the hatch open", Intent.FORCE, Stat.MIGHT, "none"),
    ("I carefully sneak past", Intent.SNEAK, Stat.GRACE, "none"),
    ("I analyze the console", Intent.SEARCH, Stat.WITS, "none"),
    ("I hold steady and face it", Intent.OTHER, Stat.NERVE, "none"),
])
def test_heuristic_intents_and_stats(text, intent, stat, target):
    s = HeuristicTransducer().extract(text, TARGETS)
    assert (s.intent, s.effective_stat, s.target) == (intent, stat, target)


def test_heuristic_psychology():
    h = HeuristicTransducer()
    probe = h.extract("Marcus, what are you hiding? I saw you with the Hymn.", TARGETS)
    assert probe.exposure > 0.6 and probe.target == "marcus"
    threat = h.extract("Tell me where the codes are or else I kill you, Elara!", TARGETS)
    assert threat.aggression > 0.6 and threat.rapport < 0
    warm = h.extract("Thank you, friend. I trust you.", TARGETS)
    assert warm.rapport > 0.5 and warm.aggression == 0


def test_npc_channels_split_rapport():
    s = SensoryState(rapport=-0.4, aggression=0.2)
    assert s.npc_channels() == [0.2, 0.0, 0.0, 0.0, 0.0, 0.4]
    assert s.shock == pytest.approx(0.4)
