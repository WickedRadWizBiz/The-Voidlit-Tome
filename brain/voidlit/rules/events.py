"""Rule events: what happened, in order, for the voice (facts) and the voxel surface (tones)."""

from __future__ import annotations

from dataclasses import dataclass, field

# Tones follow the TOME's voxel language: amber is light, cyan a Hit, crimson harm and Stress,
# violet the TOME's own will, gold a Page. "stone" is raised black (the TOME's words), "absence" sunken black.
TONES = ("lumen", "hit", "harm", "stress", "tome", "page", "stone", "absence")


@dataclass
class Event:
    kind: str                # machine tag: harm, wound, stress, panic, lumen, progress, cleared, echo, ...
    text: str                # short line the surface can raise ("-1 GRIT")
    tone: str = "stone"
    fact: str = ""           # a plain sentence the voice must narrate (empty: surface only)
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text, "tone": self.tone, "fact": self.fact, "data": self.data}


class Log(list):
    def add(self, kind: str, text: str, tone: str = "stone", fact: str = "", /, **data) -> Event:
        e = Event(kind, text, tone, fact, data)
        self.append(e)
        return e

    def facts(self) -> list[str]:
        return [e.fact for e in self if e.fact]

    def has(self, kind: str) -> bool:
        return any(e.kind == kind for e in self)
