import random

import pytest

from voidlit.rules import Attribute, CheckLog, Character, Item, Outcome, as_pool, d6, d20, degree_for, mod
from voidlit.rules.checks import d6_table
from voidlit.rules.panic import panic_check
from voidlit.rules.relevance import D6, D20, NO_CHECK, table_for


def test_modifiers_follow_scores():
    assert [mod(s) for s in range(5)] == [-2, 0, 2, 4, 6]


@pytest.mark.parametrize("natural,total,dc,expected", [
    (10, 22, 12, Outcome.CRITICAL), (10, 21, 12, Outcome.SUCCESS), (10, 12, 12, Outcome.SUCCESS),
    (10, 11, 12, Outcome.MIXED), (10, 7, 12, Outcome.MIXED), (10, 6, 12, Outcome.FAIL),
    (20, 12, 12, Outcome.CRITICAL), (20, 6, 12, Outcome.MIXED), (1, 12, 12, Outcome.MIXED),
    (1, 7, 12, Outcome.FAIL), (1, 30, 12, Outcome.SUCCESS),
])
def test_degrees_and_natural_shifts(natural, total, dc, expected):
    assert degree_for(natural, total, dc) is expected


def test_d20_records_every_roll_and_rejects_unknown_actions():
    log = CheckLog()
    rec = d20(random.Random(1), log, "player", "player.search", "insight", 2, 14)
    assert log.records == [rec] and rec.die == "d20" and rec.total == rec.natural + 2
    with pytest.raises(KeyError):
        d20(random.Random(1), log, "player", "player.teleport", "grit", 0, 10)
    with pytest.raises(ValueError):  # searching does not oppose the Tome: no upper-hand die
        d20(random.Random(1), log, "player", "player.search", "insight", 2, 14, upper_hand=True)


def test_upper_hand_adds_a_d6_for_the_actor_or_to_the_defence():
    rng = random.Random(2)
    a = [d20(rng, None, "player", "player.strike_tome", "grit", 0, 12, upper_hand=True) for _ in range(300)]
    assert all(1 <= r.upper_hand_d6 <= 6 and r.total == r.natural + r.upper_hand_d6 for r in a)
    b = [d20(rng, None, "tome", "tome.escalate", "grit", 2, 12, defender_upper_hand=True) for _ in range(300)]
    assert all(1 <= r.defence_d6 <= 6 and r.dc == 12 + r.defence_d6 for r in b)


def test_upper_hand_raises_success_rate():
    def rate(uh):
        rng = random.Random(3)
        rs = [d20(rng, None, "player", "player.strike_tome", "grit", 0, 16, upper_hand=uh) for _ in range(4000)]
        return sum(r.degree.succeeded for r in rs) / len(rs)
    assert rate(True) > rate(False) + 0.12


def test_d6_passive_checks_and_tables():
    log = CheckLog()
    rng = random.Random(4)
    for _ in range(200):
        r = d6(rng, log, "player", "passive.panic", "resolve", 5, 6, strict=True)
        assert r.degree.succeeded == (r.natural + 5 > 6)
    with pytest.raises(KeyError):
        d6(rng, log, "player", "player.engage", "grit", 1, 4)  # a direct action is never a d6
    seen = {d6_table(rng, log, "passive.twist_pip", 3) for _ in range(200)}
    assert seen == {0, 1, 2}


def test_adapter_maps_degrees_onto_obstacle_hits():
    rng = random.Random(5)
    for _ in range(300):
        rec = d20(rng, None, "player", "player.engage", "grit", 0, 12)
        pool = as_pool(rec)
        assert pool.outcome is rec.degree


def test_character_modifier_includes_items_and_conditions():
    c = Character("Ash", "Salvager", grit=3, guile=1, insight=1, hull_max=5, resolve=2,
                  inventory=[Item("Cutter", bonus={"grit": 1})])
    assert c.modifier_for(Attribute.GRIT)[0] == 4 + 2
    c.conditions |= {"shaken", "bleeding"}
    assert c.modifier_for(Attribute.GRIT)[0] == 6 - 2 - 2
    assert c.modifier_for(Attribute.RESOLVE)[0] == 2 - 2


def test_resolve_holds_panic():
    rng = random.Random(6)
    steady = Character("A", "x", 1, 1, 1, 5, resolve=4, stress=7)
    shaky = Character("B", "x", 1, 1, 1, 5, resolve=0, stress=7)
    held = lambda c: sum(panic_check(c, rng).held for _ in range(600)) / 600  # noqa: E731
    assert held(steady) == 1.0          # d6 + 4 + 3 > 7 always
    assert 0.25 < held(shaky) < 0.42    # d6 + 0 + 3 > 7 only on a 5 or 6


def test_every_action_has_exactly_one_home():
    keys = list(D20) + list(D6) + list(NO_CHECK)
    assert len(keys) == len(set(keys))
    for k in keys:
        table_for(k)
    with pytest.raises(KeyError):
        table_for("player.teleport")
