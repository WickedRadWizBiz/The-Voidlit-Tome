"""Obstacles: enemies, hazards, locks and horrors share one mechanic.

An obstacle carries a row of *pips*, each marked with an approach (Grit,
Guile or Insight). The player picks how to tackle it; every hit from a roll
using that approach strips one matching pip. Hits that find no matching pip
are not wasted, just inefficient: every two of them strip one pip of any
kind (brute force works, slowly). A critical strips one extra pip of any kind.
Clear every pip and the obstacle is overcome.

If the obstacle isn't cleared, a FAIL lets it *bite* for its full harm and
stress. A MIXED roll still lands its hits, but the exchange rattles you:
1 Stress (when the obstacle can hurt you at all). Hitting with an approach
that matches no pip still keeps you from being bitten on a success.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .character import Approach
from .dice import Outcome, PoolRoll


@dataclass
class Obstacle:
    id: str
    name: str
    kind: str  # enemy | hazard | lock | horror
    pips: list[str]
    harm: int = 1
    stress: int = 0
    text: str = ""
    defeat_text: str = ""
    blocking: bool = True

    @property
    def cleared(self) -> bool:
        return not self.pips

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "kind": self.kind, "pips": list(self.pips),
                "harm": self.harm, "stress": self.stress, "text": self.text,
                "defeat_text": self.defeat_text, "blocking": self.blocking}

    @classmethod
    def from_dict(cls, d: dict) -> "Obstacle":
        return cls(id=d["id"], name=d["name"], kind=d["kind"], pips=list(d["pips"]),
                   harm=d.get("harm", 1), stress=d.get("stress", 0), text=d.get("text", ""),
                   defeat_text=d.get("defeat_text", ""), blocking=d.get("blocking", True))


@dataclass
class Exchange:
    approach: Approach
    roll: PoolRoll
    removed: list[str] = field(default_factory=list)
    matched: bool = False
    cleared: bool = False
    bite_harm: int = 0
    bite_stress: int = 0

    def to_dict(self) -> dict:
        return {"approach": self.approach.value, "roll": self.roll.to_dict(), "removed": self.removed,
                "matched": self.matched, "cleared": self.cleared,
                "bite_harm": self.bite_harm, "bite_stress": self.bite_stress}


def resolve_exchange(obstacle: Obstacle, approach: Approach, roll: PoolRoll,
                     dark: bool = False) -> Exchange:
    ex = Exchange(approach=approach, roll=roll)
    ex.matched = approach.value in obstacle.pips
    leftover = 0
    for _ in range(roll.hits):
        if approach.value in obstacle.pips:
            obstacle.pips.remove(approach.value)
            ex.removed.append(approach.value)
        else:
            leftover += 1
    for _ in range(leftover // 2):
        if obstacle.pips:
            ex.removed.append(obstacle.pips.pop(0))
    if roll.outcome is Outcome.CRITICAL and obstacle.pips:
        ex.removed.append(obstacle.pips.pop(0))
    ex.cleared = obstacle.cleared
    if not ex.cleared:
        outcome = roll.outcome
        extra = 1 if dark else 0
        if outcome is Outcome.FAIL:
            ex.bite_harm = obstacle.harm + (extra if obstacle.harm else 0)
            ex.bite_stress = obstacle.stress + extra
        elif outcome is Outcome.MIXED and (obstacle.harm or obstacle.stress):
            ex.bite_stress = 1
    return ex
