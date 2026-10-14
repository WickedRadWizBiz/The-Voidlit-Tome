"""Progress clocks: segmented countdowns for threats the player can't see coming."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Clock:
    id: str
    name: str
    segments: int
    filled: int = 0
    consequence: str = ""  # an obstacle id, "lumen", or "" for narration only
    fired: bool = False

    @property
    def fraction(self) -> float:
        return self.filled / self.segments if self.segments else 0.0

    @property
    def full(self) -> bool:
        return self.filled >= self.segments

    def tick(self, n: int = 1) -> bool:
        """Advance the clock. Returns True exactly once, on the tick that fills it."""
        if self.fired:
            return False
        self.filled = min(self.segments, self.filled + n)
        if self.full:
            self.fired = True
            return True
        return False

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "segments": self.segments, "filled": self.filled,
                "consequence": self.consequence, "fired": self.fired}

    @classmethod
    def from_dict(cls, d: dict) -> "Clock":
        return cls(**d)
