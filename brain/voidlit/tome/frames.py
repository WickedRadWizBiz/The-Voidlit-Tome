"""Tome frames: the render protocol between the Tome's mind and its voxel body.

A frame is semantic, not pixels: blocks of colored spans with effect hints.
The Unreal client typesets them onto the voxel grid. The terminal client
prints them with ANSI colour. See docs/PROTOCOL.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PROTOCOL_VERSION = 2

# The TOME's voxel language (TOME: Core Rules 0.2, "The voxel language"). Colors have fixed meanings: amber is
# light (the Lumen), cyan a Hit, crimson harm and Stress, violet the TOME's own will (Echoes, Interventions, the
# Agenda), gold a Page. Everything else is matte black, read by relief: raised for the TOME's words and glyphs,
# sunken for absence (what is missing, Depth still to clear, a Wound). The hex values for black relief are what a
# flat screen (the terminal) shows instead; the voxel client renders them as black stone.
PALETTE: dict[str, str] = {
    "lumen": "#ffb23d",     # amber
    "hit": "#3de0ff",       # cyan
    "harm": "#d8213b",      # crimson
    "stress": "#d8213b",    # crimson
    "tome": "#9b5cff",      # violet
    "page": "#e8c547",      # gold
    "stone": "#d9d4c7",     # raised black: the TOME's words
    "absence": "#5c5a57",   # sunken black
    "title": "#d9d4c7",
    "narration": "#c9c4b8",
    "echo": "#6f6c66",
    "system": "#6f6c66",
}
# Relief per tone: raised (words, offers), sunken (absence, cost, danger) or flush (silence).
RELIEF: dict[str, str] = {"absence": "sunken", "harm": "sunken", "stress": "sunken", "echo": "flush"}
# Color-on-black tones: everything else is matte black.
COLORED = {"lumen", "hit", "harm", "stress", "tome", "page"}

STAT_GLYPH = {"might": "▲", "grace": "◆", "wits": "●", "nerve": "✦", "none": "·"}
SHAPE_GLYPH = {"spines": "✶", "fracture": "⟋", "facets": "◇", "ring": "○", "smooth": "◯"}


@dataclass
class Span:
    text: str
    color: str = PALETTE["narration"]
    glow: float = 0.0       # 0 = matte; 1 = brilliant emissive
    effect: str = "rise"    # rise | flicker | pulse | ripple | none
    relief: str = "raised"  # raised | sunken | flush (voxels about 1 mm up, 1 mm down, or level)
    tone: str = "stone"     # the voxel-language meaning: lumen | hit | harm | stress | tome | page | stone | absence

    def to_dict(self) -> dict:
        return {"text": self.text, "color": self.color, "glow": round(self.glow, 3), "effect": self.effect,
                "relief": self.relief, "tone": self.tone}


@dataclass
class Block:
    role: str               # title | narration | echo | dice | npc | event | clue | leaf | draw | prompt
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


def span(text: str, tone: str = "stone", glow: float | None = None, effect: str = "rise",
         color: str | None = None) -> Span:
    if glow is None:
        glow = 0.7 if tone in COLORED else 0.0
    return Span(text=text, color=color or PALETTE.get(tone, PALETTE["stone"]), glow=glow, effect=effect,
                relief=RELIEF.get(tone, "raised"), tone=tone if tone in PALETTE else "stone")


def text_block(role: str, text: str, tone: str | None = None, glow: float | None = None, effect: str = "rise",
               speaker: str | None = None, color: str | None = None) -> Block:
    t = tone or (role if role in PALETTE else "stone")
    s = span(text, t, glow, effect, color)
    if tone is None and role in ("narration", "title", "echo", "system"):
        s.color = color or PALETTE[role]
    return Block(role=role, spans=[s], speaker=speaker)


def dice_block(dice: list[int], pool: int, hits: int, difficulty: int, result: str, stat: str,
               notes: list[str], label: str = "") -> Block:
    zero = pool == 0
    spans: list[Span] = [span(f"{STAT_GLYPH.get(stat, '')} {label or stat.upper()} ", "stone", 0.2)]
    for d in dice:
        hit = d == 6 if zero else d >= 5
        spans.append(span(f"[{d}]", "hit" if hit else "absence", 0.9 if hit else 0.0,
                          "pulse" if hit else "rise"))
    tone = {"success": "hit", "short": "stress", "fail": "absence"}.get(result, "stone")
    spans.append(span(f"  {hits}/{difficulty} {result.upper()}", tone))
    if notes:
        spans.append(span(f"  ({', '.join(notes)})", "system", 0.0))
    return Block(role="dice", spans=spans,
                 data={"dice": dice, "pool": pool, "hits": hits, "difficulty": difficulty, "result": result,
                       "stat": stat})
