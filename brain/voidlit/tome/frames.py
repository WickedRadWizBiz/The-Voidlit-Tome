"""Tome frames: the render protocol between the Tome's mind and its voxel body.

A frame is semantic, not pixels: blocks of colored spans with effect hints.
The Unreal client typesets them onto the voxel grid. The terminal client
prints them with ANSI colour. See docs/PROTOCOL.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PROTOCOL_VERSION = 2

PALETTE: dict[str, str] = {
    "title": "#ffe9a8",
    "narration": "#c9d1e0",
    "echo": "#7f8796",
    "system": "#6f7a8f",
    "grit": "#ff6a3d",
    "guile": "#b26bff",
    "insight": "#3dd6ff",
    "none": "#c9d1e0",
    "fail": "#ff3d5a",
    "mixed": "#ffd23d",
    "success": "#7dffb2",
    "critical": "#ffffff",
    "hull": "#9fe6a0",
    "stress": "#ff4d6d",
    "lumen": "#ffe9a8",
    "clue": "#e8d9ff",
    "dread": "#9a1f40",
}

@dataclass
class Span:
    text: str
    color: str = PALETTE["narration"]
    glow: float = 0.0       # 0 = matte; 1 = brilliant emissive
    effect: str = "rise"    # rise | flicker | pulse | ripple | none

    def to_dict(self) -> dict:
        return {"text": self.text, "color": self.color, "glow": round(self.glow, 3), "effect": self.effect}


@dataclass
class Block:
    role: str               # title | echo | narration | npc | prompt | system
    spans: list[Span] = field(default_factory=list)
    speaker: str | None = None
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"role": self.role, "spans": [s.to_dict() for s in self.spans]}
        if self.speaker:
            d["speaker"] = self.speaker
        if self.data:
            d["data"] = self.data
        return d

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans)


@dataclass
class TomeFrame:
    seq: int
    blocks: list[Block]
    status: dict
    mood: dict
    effects: list[dict] = field(default_factory=list)
    over: bool = False
    debug: dict = field(default_factory=dict)

    def to_dict(self, include_debug: bool = False) -> dict:
        d = {
            "type": "frame", "protocol": PROTOCOL_VERSION, "seq": self.seq,
            "blocks": [b.to_dict() for b in self.blocks], "status": self.status, "mood": self.mood,
            "effects": self.effects, "over": self.over,
        }
        if include_debug:
            d["debug"] = self.debug
        return d

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks)


def text_block(role: str, text: str, color_key: str | None = None, glow: float = 0.0,
               effect: str = "rise", speaker: str | None = None, color: str | None = None) -> Block:
    c = color or PALETTE.get(color_key or role, PALETTE["narration"])
    return Block(role=role, spans=[Span(text=text, color=c, glow=glow, effect=effect)], speaker=speaker)
