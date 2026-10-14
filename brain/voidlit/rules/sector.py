"""The sector deck: each run is a short stack of chapters that escalates by act.

A run is one start chapter, ``per_act`` chapters drawn from each of acts
I-III (each act deadlier than the last), and one escape chapter. You win by
getting through the escape chapter alive.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

START_ACT = 0
FINAL_ACT = 4


@dataclass
class Chapter:
    id: str
    act: int
    kind: str  # passage | hazard | encounter | discovery | respite | horror | escape
    title: str
    text: str
    obstacles: list[str] = field(default_factory=list)
    npcs: list[str] = field(default_factory=list)
    clues: list[str] = field(default_factory=list)
    respite: bool = False
    light: int = 0  # Lumen dice a successful search can recover here
    exit_text: str = ""


@dataclass
class SectorDeck:
    order: list[str]
    index: int = 0

    @classmethod
    def build(cls, chapters: dict[str, Chapter], per_act: int, rng: random.Random) -> "SectorDeck":
        by_act: dict[int, list[str]] = {}
        for ch in chapters.values():
            by_act.setdefault(ch.act, []).append(ch.id)
        for ids in by_act.values():
            ids.sort()
        if not by_act.get(START_ACT) or not by_act.get(FINAL_ACT):
            raise ValueError("a sector needs at least one start (act 0) and one escape (act 4) chapter")
        order = [rng.choice(by_act[START_ACT])]
        for act in (1, 2, 3):
            pool = list(by_act.get(act, []))
            rng.shuffle(pool)
            order.extend(pool[:per_act])
        order.append(rng.choice(by_act[FINAL_ACT]))
        return cls(order=order)

    @property
    def current(self) -> str:
        return self.order[self.index]

    @property
    def progress(self) -> float:
        return self.index / max(1, len(self.order) - 1)

    @property
    def at_end(self) -> bool:
        return self.index >= len(self.order) - 1

    def advance(self) -> str | None:
        if self.at_end:
            return None
        self.index += 1
        return self.current

    def to_dict(self) -> dict:
        return {"order": list(self.order), "index": self.index}

    @classmethod
    def from_dict(cls, d: dict) -> "SectorDeck":
        return cls(order=list(d["order"]), index=d["index"])
