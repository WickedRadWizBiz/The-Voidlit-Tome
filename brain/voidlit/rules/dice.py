"""D6 dice pools: the only randomness the rules ever consult.

Every roll in the game is a pool of six-sided dice. Each 5 or 6 is a *hit*.

* 0 hits  -> FAIL      (the Tome makes a hard move)
* 1 hit   -> MIXED     (you get it, at a cost)
* 2+ hits -> SUCCESS
* 2+ hits including two or more 6s -> CRITICAL (an extra effect)

An empty pool (0 dice) still rolls one die, but only a 6 counts as a hit.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

MAX_POOL = 8
HIT_MIN_FACE = 5


class Outcome(str, Enum):
    FAIL = "fail"
    MIXED = "mixed"
    SUCCESS = "success"
    CRITICAL = "critical"

    @property
    def succeeded(self) -> bool:
        return self in (Outcome.SUCCESS, Outcome.CRITICAL)


@dataclass(frozen=True)
class PoolRoll:
    dice: tuple[int, ...]
    pool: int
    hits: int
    sixes: int

    @property
    def outcome(self) -> Outcome:
        if self.hits == 0:
            return Outcome.FAIL
        if self.hits == 1:
            return Outcome.MIXED
        if self.sixes >= 2:
            return Outcome.CRITICAL
        return Outcome.SUCCESS

    def to_dict(self) -> dict:
        return {
            "dice": list(self.dice),
            "pool": self.pool,
            "hits": self.hits,
            "outcome": self.outcome.value,
        }


def roll_pool(n: int, rng: random.Random) -> PoolRoll:
    """Roll a pool of ``n`` dice (clamped to ``MAX_POOL``)."""
    n = max(0, min(MAX_POOL, n))
    if n == 0:
        die = rng.randint(1, 6)
        six = int(die == 6)
        return PoolRoll(dice=(die,), pool=0, hits=six, sixes=six)
    dice = tuple(rng.randint(1, 6) for _ in range(n))
    hits = sum(1 for d in dice if d >= HIT_MIN_FACE)
    sixes = sum(1 for d in dice if d == 6)
    return PoolRoll(dice=dice, pool=n, hits=hits, sixes=sixes)


@dataclass(frozen=True)
class DepletionRoll:
    dice: tuple[int, ...]
    before: int
    after: int

    @property
    def lost(self) -> int:
        return self.before - self.after


def deplete(pool: int, rng: random.Random, fail_face: int = 1) -> DepletionRoll:
    """Roll a resource pool. Every die showing ``fail_face`` or lower is lost."""
    pool = max(0, min(MAX_POOL, pool))
    dice = tuple(rng.randint(1, 6) for _ in range(pool))
    lost = sum(1 for d in dice if d <= fail_face)
    return DepletionRoll(dice=dice, before=pool, after=pool - lost)


def roll_2d6(rng: random.Random) -> tuple[int, int]:
    return rng.randint(1, 6), rng.randint(1, 6)
