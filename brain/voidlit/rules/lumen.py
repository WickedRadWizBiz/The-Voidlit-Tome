"""The Lumen: the Tome's own light, and the run's countdown.

The Lumen is a pool of up to six dice. Every ``interval`` turns the Tome rolls
it, and each die showing a 1 gutters out. Some events burn Lumen directly.
At zero the Dark has the sector: every hazard bites harder, and the Director
is pushed to escalate every turn.

The renderer reads ``fraction`` straight off the frame and dims the voxels
with it, so the light running out is something the player *sees*.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .dice import DepletionRoll, deplete

LUMEN_MAX = 6


@dataclass
class Lumen:
    dice: int = LUMEN_MAX
    interval: int = 5
    turns_since_tick: int = 0

    @property
    def fraction(self) -> float:
        return self.dice / LUMEN_MAX

    @property
    def dark(self) -> bool:
        return self.dice <= 0

    def advance(self, rng: random.Random) -> DepletionRoll | None:
        """Call once per turn. Returns the depletion roll when one happens."""
        if self.dark:
            return None
        self.turns_since_tick += 1
        if self.turns_since_tick < self.interval:
            return None
        self.turns_since_tick = 0
        roll = deplete(self.dice, rng)
        self.dice = roll.after
        return roll

    def burn(self, n: int = 1) -> int:
        before = self.dice
        self.dice = max(0, self.dice - n)
        return before - self.dice

    def restore(self, n: int = 1) -> int:
        before = self.dice
        self.dice = min(LUMEN_MAX, self.dice + n)
        return self.dice - before

    def to_dict(self) -> dict:
        return {"dice": self.dice, "interval": self.interval, "turns_since_tick": self.turns_since_tick}

    @classmethod
    def from_dict(cls, d: dict) -> "Lumen":
        return cls(**d)
