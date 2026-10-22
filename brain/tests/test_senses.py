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


SCENE = {"tome": ["tome", "book", "monolith"], **TARGETS, "jammed_iris": ["jammed iris door", "jammed", "iris", "door"],
         "warden_turret": ["warden turret", "warden", "turret"]}
T, F = True, False
TALK, FIGHT, FORCE, SNEAK, LOOK = Intent.TALK, Intent.FIGHT, Intent.FORCE, Intent.SNEAK, Intent.LOOK
# The routing table, labelled by hand: what the player types (with Elara, Marcus, the iris door and the turret in the
# scene), what they do, whom or what it is aimed at, the topic of a question put to the Tome, and whether the words are
# aimed at the Tome at all.
ROUTING = [
    # the one named as spoken to, at the head, after an opener, before a colon, a dash or a question word, or last
    ("Tome, what does Elara want?", TALK, "tome", "elara", T), ("Where is Elara, Tome?", TALK, "tome", "elara", T),
    ("Elara, where is Marcus?", TALK, "elara", "none", F), ("Is Marcus armed, Elara?", TALK, "elara", "none", F),
    ("Marcus, I swear I am on your side", TALK, "marcus", "none", F),
    ("Hey Tome, what does Elara want?", TALK, "tome", "elara", T),
    ("Tome: what does Elara want?", TALK, "tome", "elara", T),
    ("Tome - who is Marcus?", TALK, "tome", "marcus", T), ("Tome — who is Marcus?", TALK, "tome", "marcus", T),
    ("Tell me, Tome, what does Elara want?", TALK, "tome", "elara", T),
    ("So, Tome, who is Marcus?", TALK, "tome", "marcus", T),
    ("O Tome, what does Elara want?", TALK, "tome", "elara", T),
    ("Tome what does Elara want?", TALK, "tome", "elara", T),
    ("What does Elara want, O Tome?", TALK, "tome", "elara", T),
    ("Hey Marcus, where is Elara?", TALK, "marcus", "none", F), ("Elara where is Marcus?", TALK, "elara", "none", F),
    ("Look, Marcus, put the gun down.", TALK, "marcus", "none", F),
    ("Dear Tome, who is Marcus?", TALK, "tome", "marcus", T), ("My Tome, who is Marcus?", TALK, "tome", "marcus", T),
    ("Excuse me, Tome, who is Marcus?", TALK, "tome", "marcus", T),
    ("Hey there Tome, who is Marcus?", TALK, "tome", "marcus", T),
    # ... in a later sentence (of several, the one a question is put to), or in the third person
    ("I cover the Tome. Tome, who is Marcus?", SNEAK, "tome", "marcus", T),
    ("I smash the Tome. Tome, who is Marcus?", FORCE, "tome", "marcus", T),
    ("Elara, Marcus is armed. Tome, who is Marcus?", TALK, "tome", "marcus", T),
    ("Who is Marcus? Tome?", TALK, "tome", "marcus", T),
    ("Marcus? Where is the lifeboat?", TALK, "marcus", "none", F),
    ("I ask the Tome who Marcus is.", TALK, "tome", "marcus", T),
    ("I ask the Tome: who is Marcus?", TALK, "tome", "marcus", T),
    ("I turn to the Tome and ask who Marcus is.", TALK, "tome", "marcus", T),
    ("I ask the Tome about the lifeboat.", TALK, "tome", "lifeboat", T),
    ("I tell Marcus to ask the Tome", TALK, "marcus", "none", F),
    ("I ask Elara who Marcus is.", TALK, "elara", "none", F),
    # of two spoken to, the questions the Tome hears, not those put to the other
    ("Tome, is Marcus lying? Marcus, where is the lifeboat?", TALK, "tome", "marcus", T),
    # a question with no question mark is still a question
    ("Who is Marcus, Tome", TALK, "tome", "marcus", T), ("Tome what does Elara want", TALK, "tome", "elara", T),
    # a name, then the player telling what they do, is that act and not talk
    ("Tome! I smash you.", FORCE, "tome", "none", T), ("Tome: I smash you.", FORCE, "tome", "none", T),
    ("Elara. I attack her.", FIGHT, "elara", "none", F), ("Marcus - I attack him.", FIGHT, "marcus", "none", F),
    ("Hey Marcus, I shoot you.", FIGHT, "marcus", "none", F), ("Marcus! I shoot him.", FIGHT, "marcus", "none", F),
    ("Elara: I attack you.", FIGHT, "elara", "none", F), ("Marcus, I attack you.", FIGHT, "marcus", "none", F),
    ("Tome, I swear I found nothing back there", TALK, "tome", "none", T),
    ("Tome, answer me or I will break you", TALK, "tome", "none", T),
    # a question to no one in particular goes to the Tome, whatever else the line does
    ("Where is the lifeboat?", TALK, "none", "lifeboat", T),
    ("I take cover behind a crate. Where is the lifeboat?", SNEAK, "none", "lifeboat", T),
    ("The corridor is dim. Where is the lifeboat?", TALK, "none", "lifeboat", T),
    ("I force the iris door. Where is the lifeboat?", FORCE, "jammed_iris", "lifeboat", T),
    ("where am I?", LOOK, "none", "none", F), ('"Where is the lifeboat?"', TALK, "none", "none", F),
    # a name in passing is not one spoken to: the first one named (the longer of two that start together), the Tome last
    ("Hey, Elara is lying", TALK, "elara", "none", F), ("So the tome is lying to Marcus", TALK, "marcus", "none", F),
    ("I shoot Elara", FIGHT, "elara", "none", F), ("Look, the Tome!", LOOK, "tome", "none", T),
    ("I give the lamp to Elara, then Marcus.", TALK, "elara", "none", F),
    ("I shoot the warden turret", FIGHT, "warden_turret", "none", F),
    ("I shoot the door near Elara", FIGHT, "jammed_iris", "none", F),
]


@pytest.mark.parametrize("text,intent,target,asks,addressed", ROUTING)
def test_the_routing_table(scenario, text, intent, target, asks, addressed):
    """Words are for the one they are spoken to, however they are named and wherever in the line; a name followed by
    what the player does is that act; a question to no one is the Tome's."""
    s = HeuristicTransducer().extract(text, SCENE, scenario.ledger["topics"])
    assert (s.intent, s.target, s.asks, s.addressed_tome) == (intent, target, asks, addressed), text


@pytest.mark.parametrize("text,topic", [
    ("Hey Tome, what does Elara want?", "elara"), ("Tome: what does Elara want?", "elara"),
    ("Tome - who is Marcus?", "marcus"), ("Tell me, Tome, what does Elara want?", "elara"),
    ("So, Tome, who is Marcus?", "marcus"), ("O Tome, what does Elara want?", "elara"),
    ("Tome what does Elara want", "elara"), ("I cover the Tome with my coat. Where is the lifeboat?", "lifeboat"),
    ("Hey Tome?", "unclear"),
    # the topic of the question, not of whatever else the line names
    ("I hide behind the lifeboat. Tome, who is Marcus?", "marcus"), ("Marcus. Tell me about him, Tome.", "marcus"),
])
def test_a_question_put_to_the_tome_however_it_is_named(scenario, text, topic):
    s = HeuristicTransducer().extract(text, SCENE, scenario.ledger["topics"])
    assert (s.target, s.addressed_tome, s.asks) == ("tome", True, topic)
