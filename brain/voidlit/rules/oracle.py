"""Asking the TOME: a yes/no oracle with a twist (Core Rules 0.2, "Asking the TOME")."""

from __future__ import annotations

import random
from dataclasses import dataclass

YES_ON = {"likely": 3, "even": 4, "unlikely": 5}


@dataclass(frozen=True)
class Answer:
    likelihood: str
    answer_die: int
    twist_die: int

    @property
    def yes(self) -> bool:
        return self.answer_die >= YES_ON[self.likelihood]

    @property
    def twist(self) -> str:
        return {1: "complication", 6: "benefit"}.get(self.twist_die, "")

    @property
    def unasked(self) -> bool:
        """Matching dice: the TOME tells you something you did not ask."""
        return self.answer_die == self.twist_die

    def to_dict(self) -> dict:
        return {"likelihood": self.likelihood, "dice": [self.answer_die, self.twist_die], "yes": self.yes,
                "twist": self.twist, "unasked": self.unasked}


def likelihood_of(text: str) -> str:
    low = text.lower()
    if "unlikely" in low or "doubt" in low:
        return "unlikely"
    if "likely" in low or "probably" in low or "surely" in low:
        return "likely"
    return "even"


def ask(likelihood: str, rng: random.Random) -> Answer:
    return Answer(likelihood if likelihood in YES_ON else "even", rng.randint(1, 6), rng.randint(1, 6))
