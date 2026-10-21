"""Hybrid invisible dice: d20 + modifier for direct actions, d6 for passive events, +d6 for a significant upper hand.

Degrees map onto the existing hit/pip machinery, so obstacles, harm and the voice stay as they are:
  total < DC -> FAIL (0 hits) | DC..DC+4 -> MIXED (1 hit) | DC+5..DC+9 -> SUCCESS (2 hits) | >= DC+10 -> CRITICAL
  natural 1 -> FAIL always; natural 20 -> at least SUCCESS (CRITICAL if it also clears DC).
Modifiers: attribute score a (0..4) -> mod = 2a - 2 (0 -> -2, 2 -> +2, 4 -> +6); each former "+1 die" -> +2.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from voidlit.rules.dice import PoolRoll


def attr_mod(score: int) -> int:
    return 2 * score - 2


@dataclass(frozen=True)
class Check:
    d20: int
    mod: int
    bonus_d6: int        # 0 when no upper hand
    dc: int

    @property
    def total(self) -> int:
        return self.d20 + self.mod + self.bonus_d6

    @property
    def degree(self) -> int:     # 0 fail, 1 mixed, 2 success, 3 critical
        if self.d20 == 1:
            return 0
        m = self.total - self.dc
        deg = 0 if m < 0 else 1 if m < 5 else 2 if m < 10 else 3
        if self.d20 == 20:
            deg = max(deg, 2) if m < 0 else 3
        return deg

    def as_pool(self) -> PoolRoll:
        """Express the degree as the PoolRoll the obstacle rules already understand (never shown to anyone)."""
        hits = {0: 0, 1: 1, 2: 2, 3: 3}[self.degree]
        sixes = 2 if self.degree == 3 else 0
        dice = tuple([6] * sixes + [5] * (hits - sixes) + [1] * max(0, 1 - hits))
        return PoolRoll(dice=dice, pool=len(dice), hits=hits, sixes=sixes)


def check(rng: random.Random, mod: int, dc: int, upper_hand: bool = False) -> Check:
    d = rng.randint(1, 20)
    b = rng.randint(1, 6) if upper_hand else 0
    return Check(d, mod, b, dc)


def passive(rng: random.Random, mod: int, target: int) -> tuple[int, bool]:
    """d6 passive event (notice, panic resistance, environment): d6 + mod >= target."""
    d = rng.randint(1, 6)
    return d, d + mod >= target
