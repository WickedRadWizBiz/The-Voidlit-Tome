"""Tome frames: the render protocol between the Tome's mind and its voxel body.

A frame is semantic, not pixels: blocks of colored spans with effect hints.
The Unreal client typesets them onto the voxel grid. The terminal client
prints them with ANSI colour. See docs/PROTOCOL.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PROTOCOL_VERSION = 1

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

APPROACH_GLYPH = {"grit": "▲", "guile": "◆", "insight": "●"}  # ▲ ◆ ●


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
    role: str               # title | narration | echo | dice | npc | event | clue | status | prompt
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


def dice_block(dice: list[int], pool: int, hits: int, outcome: str, approach: str, notes: list[str]) -> Block:
    zero = pool == 0
    spans: list[Span] = [Span(f"{APPROACH_GLYPH.get(approach, '')} {approach.upper()} ",
                              color=PALETTE.get(approach, PALETTE["none"]), glow=0.6)]
    for d in dice:
        hit = d == 6 if zero else d >= 5
        spans.append(Span(f"[{d}]", color=PALETTE[outcome] if hit else PALETTE["system"],
                          glow=0.9 if hit else 0.05, effect="pulse" if hit else "rise"))
    spans.append(Span(f"  {outcome.upper()}", color=PALETTE[outcome], glow=0.8))
    if notes:
        spans.append(Span(f"  ({', '.join(notes)})", color=PALETTE["system"]))
    return Block(role="dice", spans=spans,
                 data={"dice": dice, "pool": pool, "hits": hits, "outcome": outcome, "approach": approach})
