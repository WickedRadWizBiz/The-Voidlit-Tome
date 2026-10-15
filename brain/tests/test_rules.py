import random

from voidlit.rules import (ACCEPTS, KINDS, LUMEN_MAX, STATS, Campaign, Kind, PoolRoll, Result, Stat, Trial,
                           ask, build_deck, make_bearer, opening_roll, push, roll_pool)
from voidlit.rules.trial import CRUCIBLE_DEPTH, DEPTH


def bearer(codex, cls="breacher", seed=0):
    return make_bearer(codex.classes[cls], "Ash", random.Random(seed), codex.entries("origins"))


def test_hits_are_fives_and_sixes_and_difficulty_decides():
    assert PoolRoll((5, 6, 1), 3, difficulty=2).result is Result.SUCCESS
    assert PoolRoll((5, 2, 1), 3, difficulty=2).result is Result.SHORT
    assert PoolRoll((1, 2, 4), 3, difficulty=1).result is Result.FAIL


def test_zero_pool_rolls_one_die_that_hits_only_on_six():
    rng = random.Random(1)
    for _ in range(200):
        r = roll_pool(0, rng)
        assert len(r.dice) == 1 and r.hits == (1 if r.dice[0] == 6 else 0)


def test_pool_is_capped_at_six():
    assert len(roll_pool(11, random.Random(2)).dice) == 6


def test_push_rerolls_misses_and_ones_cost_stress():
    rng = random.Random(3)
    for _ in range(100):
        r = roll_pool(4, rng)
        p = push(r, rng)
        assert p.pushed and sum(1 for d in p.dice if d >= 5) >= sum(1 for d in r.dice if d >= 5)
        assert p.push_ones == p.dice.count(1)       # kept dice are Hits, so every 1 is a reroll


def test_classes_match_the_forge(codex):
    for cid, c in codex.classes.items():
        b = bearer(codex, cid)
        assert b.grit == 5 and b.stress == b.min_stress
        assert {s.value: b.base(s) for s in STATS} == {k: int(c[k]) for k in ("might", "grace", "wits", "nerve")}
        assert b.find_item("medkit") is not None


def test_opening_roll_never_raises_past_three(codex):
    for seed in range(40):
        b = bearer(codex, "archivist", seed)
        op = opening_roll(b, random.Random(seed))
        assert all(0 <= b.base(s) <= 4 for s in STATS)
        if op.raised:
            assert b.base(op.stat) <= 3


def test_kinds_accept_the_right_stats():
    assert ACCEPTS[Kind.HOSTILE] == (Stat.MIGHT, Stat.GRACE, Stat.NERVE)
    assert ACCEPTS[Kind.HAZARD] == (Stat.GRACE, Stat.MIGHT)
    assert ACCEPTS[Kind.RELIC] == (Stat.WITS, Stat.MIGHT)
    assert ACCEPTS[Kind.WITNESS] == (Stat.NERVE, Stat.WITS)


def test_the_deck_has_54_leaves():
    deck = build_deck(0, random.Random(4))
    assert len(deck) == 54
    chapters = [x for x in deck if x.kind in {k.value for k in KINDS}]
    assert len(chapters) == 36
    assert len([x for x in deck if x.kind == "echo"]) == 12
    assert len([x for x in deck if x.kind == "breather"]) == 4
    assert len(build_deck(4, random.Random(4))) == 55     # a hungry TOME adds an Intervention


def test_grit_soaks_harm_then_wounds_then_last_rites(codex):
    b = bearer(codex, "hauler")
    t = Trial.begin(b, random.Random(5))
    rng = random.Random(6)
    t.harm(3, rng, guardable=False)
    assert b.grit == 2 and not b.wounds
    for _ in range(12):
        if t.pending:
            break
        t.harm(3, rng, guardable=False)
    assert b.dead or (t.pending and t.pending["type"] == "bargain")


def test_two_bargains_then_the_tome_takes_you(codex):
    b = bearer(codex, "breacher")
    t = Trial.begin(b, random.Random(7))
    rng = random.Random(8)
    for n in range(2):
        t.last_rites(rng)
        assert t.pending["type"] == "bargain"
        t.decide_bargain(rng, True)
        assert len(b.brands) == n + 1 and not b.dead
    t.last_rites(rng)
    assert b.dead and t.hunger_gain == 2


def test_stress_at_five_checks_for_panic(codex):
    b = bearer(codex, "archivist")
    t = Trial.begin(b, random.Random(9))
    b.stress = 4
    t.add_stress(1, random.Random(10))
    assert b.stress == 5 and (t.log.has("panic") or t.log.has("held"))


def test_lumen_starts_from_hunger_and_crucible_is_deep(codex):
    b = bearer(codex, "pilgrim")
    assert Trial.begin(b, random.Random(11)).lumen == LUMEN_MAX
    assert Trial.begin(b, random.Random(11), hunger=5).lumen == 2
    assert DEPTH == {"shallow": 2, "deep": 4, "abyssal": 6} and CRUCIBLE_DEPTH == 8


def test_haven_resets_stress_and_grit(codex):
    b = bearer(codex, "fence")
    c = Campaign.begin(b, codex.entries("agendas"), random.Random(12))
    b.stress, b.grit = 8, 1
    c.haven()
    assert b.stress >= b.min_stress and b.stress <= b.min_stress + 1 and b.grit == 5


def test_opening_a_page_feeds_the_hunger(codex):
    b = bearer(codex, "vessel")
    c = Campaign.begin(b, codex.entries("agendas"), random.Random(13))
    b.insight = 4
    ok, _ = c.open_page("ward")
    assert ok and c.hunger == 1 and "ward" in c.pages_open and b.insight == 0


def test_oracle_likelihoods():
    rng = random.Random(14)
    yes = {k: sum(ask(k, rng).yes for _ in range(3000)) / 3000 for k in ("likely", "even", "unlikely")}
    assert yes["likely"] > yes["even"] > yes["unlikely"]
