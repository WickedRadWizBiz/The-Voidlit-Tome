import random

from voidlit.rules import (Approach, Chapter, Character, Clock, Lumen, Obstacle, Outcome, PoolRoll, SectorDeck,
                           deplete, panic_check, resolve_exchange, roll_pool)
from voidlit.rules.panic import PANIC_TABLE


def roll(dice, pool=None):
    pool = len(dice) if pool is None else pool
    hits = sum(1 for d in dice if (d == 6 if pool == 0 else d >= 5))
    return PoolRoll(tuple(dice), pool, hits, sum(1 for d in dice if d == 6))


def test_outcomes_follow_hit_counts():
    assert roll([1, 2, 3]).outcome is Outcome.FAIL
    assert roll([5, 2]).outcome is Outcome.MIXED
    assert roll([5, 6]).outcome is Outcome.SUCCESS
    assert roll([6, 6, 1]).outcome is Outcome.CRITICAL


def test_zero_pool_only_counts_sixes():
    rng = random.Random(1)
    for _ in range(200):
        r = roll_pool(0, rng)
        assert len(r.dice) == 1 and r.pool == 0
        assert r.hits == (1 if r.dice[0] == 6 else 0)


def test_pool_is_clamped_and_hits_are_fives_and_sixes():
    rng = random.Random(2)
    r = roll_pool(20, rng)
    assert len(r.dice) == 8
    assert r.hits == sum(d >= 5 for d in r.dice)


def test_depletion_only_loses_ones():
    rng = random.Random(3)
    for _ in range(50):
        d = deplete(6, rng)
        assert d.after == 6 - sum(1 for x in d.dice if x == 1)


def test_harm_overflows_into_last_light():
    rng = random.Random(4)
    c = Character("Ash", "Salvager", grit=3, guile=1, insight=1, hull_max=3)
    r = c.take_harm(2, rng)
    assert c.hull == 1 and r.last_light is None
    r = c.take_harm(3, rng)
    assert c.hull == 0 and r.last_light is not None
    assert r.last_light.die == "d6" and r.last_light.dc == 5 + 2  # overflow 2
    assert r.dead == (r.last_light.natural + 3 < 7)
    if not r.dead:
        assert c.grit == 2 and "bleeding" in c.conditions


def test_stress_triggers_panic_threshold_and_panic_rows_are_valid():
    c = Character("Ash", "Medic", 1, 1, 1, 4, resolve=1)
    assert not c.add_stress(3)
    assert c.add_stress(1)
    rng = random.Random(5)
    c.stress = 10
    checks = [panic_check(c, rng) for _ in range(100)]
    panicked = [p for p in checks if not p.held]
    assert len(panicked) == 100, "d6 + Resolve 1 + 3 can never beat Stress 10"
    assert all(0 <= p.row < len(PANIC_TABLE) for p in panicked)


def test_obstacle_exchange_matches_pips_and_bites():
    ob = Obstacle("drone", "a drone", "enemy", ["grit", "guile"], harm=2, stress=1)
    ex = resolve_exchange(ob, Approach.INSIGHT, roll([5, 1, 1]))
    assert not ex.matched and ob.pips == ["grit", "guile"] and ex.bite_harm == 0  # one off-approach hit: nothing
    ex = resolve_exchange(ob, Approach.INSIGHT, roll([5, 6, 1]))
    assert ex.removed == ["grit"] and ob.pips == ["guile"]  # two off-approach hits strip one pip
    ob.pips = ["grit", "guile"]
    ex = resolve_exchange(ob, Approach.GRIT, roll([5, 1]))
    assert ex.removed == ["grit"] and ex.bite_harm == 0 and ex.bite_stress == 1  # mixed: hits land, costs 1 stress
    ex = resolve_exchange(ob, Approach.GUILE, roll([1, 2]))
    assert ex.removed == [] and ex.bite_harm == 2 and ex.bite_stress == 1


def test_critical_strips_an_extra_pip_and_dark_bites_harder():
    ob = Obstacle("x", "x", "enemy", ["grit", "insight", "guile"], harm=1)
    ex = resolve_exchange(ob, Approach.GRIT, roll([6, 6]))
    assert ex.removed == ["grit", "insight"] and ob.pips == ["guile"]
    ex = resolve_exchange(ob, Approach.GRIT, roll([1]), dark=True)
    assert ex.bite_harm == 2


def test_lumen_ticks_on_interval_and_restores():
    lum = Lumen(dice=6, interval=3)
    rng = random.Random(6)
    assert lum.advance(rng) is None and lum.advance(rng) is None
    assert lum.advance(rng) is not None
    lum.burn(10)
    assert lum.dark and lum.advance(rng) is None
    assert lum.restore(2) == 2 and lum.fraction == 2 / 6


def test_clock_fires_exactly_once():
    c = Clock("c", "Clock", 4)
    assert not c.tick(3)
    assert c.tick(2)
    assert not c.tick(1) and c.filled == 4


def test_sector_deck_structure():
    chapters = {f"{a}-{i}": Chapter(f"{a}-{i}", a, "passage", "t", "x") for a in range(5) for i in range(3)}
    deck = SectorDeck.build(chapters, per_act=2, rng=random.Random(7))
    acts = [chapters[c].act for c in deck.order]
    assert acts == [0, 1, 1, 2, 2, 3, 3, 4]
    assert len(set(deck.order)) == len(deck.order)
    while deck.advance():
        pass
    assert deck.at_end and deck.progress == 1.0
