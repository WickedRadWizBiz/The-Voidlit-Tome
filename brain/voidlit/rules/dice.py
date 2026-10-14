"""D6 dice pools: the only randomness the rules ever consult (TOME: Core Rules 0.2, "Core resolution").

Roll a pool of six-sided dice. Every 5 or 6 is a *Hit*; you need as many Hits as the task's Difficulty.

* Hits >= Difficulty -> SUCCESS (each extra Hit is a Boon)
* some Hits, too few -> SHORT  (it still counts, but the TOME takes a cost: the Bite lands)
* no Hits            -> FAIL   (+1 Stress, and the situation gets worse)

The pool never exceeds 6 dice. A pool reduced to 0 still rolls 1 die, and only a 6 counts.
A missed roll may be Pushed once: reroll every die that was not a Hit, keep the new result, and each 1 on the
reroll is +1 Stress.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

MAX_POOL = 6
HIT_MIN_FACE = 5


class Result(str, Enum):
    FAIL = "fail"
    SHORT = "short"
    SUCCESS = "success"


@dataclass(frozen=True)
class PoolRoll:
    dice: tuple[int, ...]
    pool: int                 # dice asked for, after the cap (0 = the desperate single die)
    difficulty: int = 1
    pushed: bool = False
    push_ones: int = 0        # 1s rolled on the Push reroll (each is +1 Stress)

    @property
    def zero(self) -> bool:
        return self.pool == 0

    def is_hit(self, face: int) -> bool:
        return face == 6 if self.zero else face >= HIT_MIN_FACE

    @property
    def hits(self) -> int:
        return sum(1 for d in self.dice if self.is_hit(d))

    @property
    def result(self) -> Result:
        if self.hits == 0:
            return Result.FAIL
        return Result.SUCCESS if self.hits >= self.difficulty else Result.SHORT

    @property
    def missed(self) -> bool:
        return self.hits < self.difficulty

    @property
    def boons(self) -> int:
        return max(0, self.hits - self.difficulty)

    def to_dict(self) -> dict:
        return {"dice": list(self.dice), "pool": self.pool, "hits": self.hits, "difficulty": self.difficulty,
                "result": self.result.value, "pushed": self.pushed, "push_ones": self.push_ones}

    @classmethod
    def from_dict(cls, d: dict) -> "PoolRoll":
        return cls(dice=tuple(d["dice"]), pool=d["pool"], difficulty=d.get("difficulty", 1),
                   pushed=d.get("pushed", False), push_ones=d.get("push_ones", 0))


def roll_pool(n: int, rng: random.Random, difficulty: int = 1) -> PoolRoll:
    """Roll ``n`` dice (clamped to 0..MAX_POOL; 0 rolls one die that only hits on a 6)."""
    n = max(0, min(MAX_POOL, n))
    dice = tuple(rng.randint(1, 6) for _ in range(max(1, n)))
    return PoolRoll(dice=dice, pool=n, difficulty=difficulty)


def push(roll: PoolRoll, rng: random.Random) -> PoolRoll:
    """Reroll every die that was not a Hit. The new result stands even if it is worse."""
    if roll.pushed:
        raise ValueError("a roll can only be pushed once")
    kept = [d for d in roll.dice if roll.is_hit(d)]
    fresh = [rng.randint(1, 6) for _ in range(len(roll.dice) - len(kept))]
    return PoolRoll(dice=tuple(kept + fresh), pool=roll.pool, difficulty=roll.difficulty, pushed=True,
                    push_ones=sum(1 for d in fresh if d == 1))


def better(a: PoolRoll, b: PoolRoll) -> PoolRoll:
    """The Borrow Page: keep the better of two rolls."""
    return b if b.hits > a.hits else a


def d6(rng: random.Random) -> int:
    return rng.randint(1, 6)


def roll_2d6(rng: random.Random) -> tuple[int, int]:
    return rng.randint(1, 6), rng.randint(1, 6)
