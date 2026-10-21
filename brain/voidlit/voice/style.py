"""The hidden affect channel (design §11.6): how the Tome words things, never what it feels.

``VoiceStyle`` is never serialised into a frame or a protocol message. The register is the nearest centroid in the
space (P, A, D, bond, contempt, hunger, fear); this mapping is designed, not learned. The style reaches the player
only as wording: the form of address, where the useful detail sits, how often the register's wrappers (its commands,
fragments and openers) are added, and a warm Tome's unasked caution. The Tome's words are always authored: its speech
acts, verbatim, inside the wrappers below, which assert nothing.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

AXES = ("P", "A", "D", "bond", "contempt", "hunger", "fear")

# centroid per register over AXES (contempt and the rest are signed, -1..1; 0 is the resting Tome)
CENTROIDS: dict[str, tuple[float, ...]] = {
    "warm":     (0.6, 0.0, 0.0, 0.6, -0.4, 0.0, 0.0),
    "clinical": (0.0, -0.5, 0.4, 0.0, 0.0, 0.0, 0.0),
    "cold":     (0.0, 0.0, 0.5, -0.2, 0.6, 0.0, 0.0),
    "hungry":   (0.0, 0.2, 0.0, 0.0, 0.0, 0.8, 0.0),
    "wounded":  (-0.6, 0.0, -0.2, 0.5, 0.0, 0.0, 0.0),
    "gloating": (0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0),
    "wary":     (0.0, 0.6, -0.2, 0.0, 0.0, 0.0, 0.6),
}

# register -> (address, detail position)
SHAPES: dict[str, tuple[str | None, str]] = {
    "warm":     ("friend", "first"),
    "clinical": (None, "first"),
    "cold":     ("bearer", "last"),
    "hungry":   ("little light", "last"),
    "wounded":  (None, "last"),
    "gloating": ("thief", "last"),
    "wary":     (None, "first"),
}

# Offline wrappers: interjections, imperatives and forms of address only. None of them asserts anything, so a Tome
# line is always its speech acts plus these (gate G11 checks every other sentence against the acts).
LEADS: dict[str, tuple[str, ...]] = {
    "warm": ("Listen, {addr}.", "Come closer, {addr}.", "Listen."),
    "clinical": (),
    "cold": ("{Addr}.",),
    "hungry": ("Closer, {addr}.", "{Addr}."),
    "wounded": ("Listen...",),
    "gloating": ("Oh, {addr}.",),
    "wary": ("Quiet.", "Wait."),
}
TRAILS: dict[str, tuple[str, ...]] = {
    "warm": ("Mind your step.", "Go gently, {addr}.", "Keep close to me."),
    "clinical": (),
    "cold": (),
    "hungry": ("Feed me.", "Find me light.", "Bring me light."),
    "wounded": (),
    "gloating": ("Walk on.", "Keep walking."),
    "wary": ("Careful.", "Slowly."),
}
WRAPPERS: frozenset[str] = frozenset(
    w.replace("{addr}", a or "").replace("{Addr}", (a or "").capitalize()).replace(" ,", ",").replace(", .", ".")
    for reg, (a, _) in SHAPES.items() for w in LEADS[reg] + TRAILS[reg])


@dataclass(frozen=True)
class VoiceStyle:
    register: str = "clinical"
    intensity: float = 0.0             # how clearly the register won: how often its wrappers are added
    address: str | None = None
    detail_position: str = "first"
    unasked_caution: bool = False      # warm only; the caution must be a true ledger fact


def style_for(P: float = 0.0, A: float = 0.0, D: float = 0.0, bond: float = 0.0, contempt: float = 0.0,
              hunger: float = 0.0, fear: float = 0.0) -> VoiceStyle:
    x = (P, A, D, bond, contempt, hunger, fear)
    dist = {r: math.dist(x, c) for r, c in CENTROIDS.items()}
    order = sorted(dist, key=lambda r: (dist[r], r))
    reg = order[0]
    # how clearly this register won: 0 on a tie with the runner-up, 1 when far closer
    intensity = 1.0 - dist[reg] / max(dist[order[1]], 1e-9)
    addr, pos = SHAPES[reg]
    return VoiceStyle(register=reg, intensity=round(min(1.0, max(0.0, intensity)), 3), address=addr,
                      detail_position=pos, unasked_caution=reg == "warm")


def style_from_mind(mind) -> VoiceStyle:
    """From the Tome's network readouts (``tomebrain.readout.Mind``)."""
    a = getattr(mind, "affect", {}) or {}
    return style_for(P=mind.pad["P"], A=mind.pad["A"], D=mind.pad["D"], bond=a.get("bond", 0.0),
                     contempt=2.0 * mind.contempt - 1.0, hunger=a.get("hunger", 0.0), fear=a.get("fear", 0.0))


def style_from_game(lumen: float, tension: float) -> VoiceStyle:
    """The Director engine has no physiology: a designed proxy from the game state (a dying light reads as hunger,
    tension as arousal and fear)."""
    hunger = max(0.0, min(1.0, 1.6 * (0.6 - lumen) / 0.6))
    return style_for(A=tension - 0.3, hunger=hunger, fear=max(0.0, tension - 0.4))


def _fill(w: str, addr: str | None) -> str:
    return (w.replace("{addr}", addr or "").replace("{Addr}", (addr or "").capitalize())
            .replace(" ,", ",").replace(", .", "."))


def _body(lines: list[str], style: VoiceStyle, caution: str | None) -> list[str]:
    """The Tome's speech acts as they will be said: never altered except for the wounded register's trailing
    ellipsis, plus a warm Tome's unasked caution (a true, public fact)."""
    body = list(lines)
    if style.register == "wounded":
        body = [b[:-1] + "..." if b.endswith(".") else b for b in body]
    if caution and style.unasked_caution and caution not in body:
        body.append(caution)
    return body


def render(lines: list[str], style: VoiceStyle, rng: random.Random, caution: str | None = None) -> str:
    """Wrap the Tome's speech acts (already true, already chosen) in its register."""
    if not lines:
        return ""
    body = _body(lines, style, caution)
    reg = style.register
    p = 0.5 + 0.5 * style.intensity
    lead = [_fill(rng.choice(LEADS[reg]), style.address)] if LEADS[reg] and rng.random() < p else []
    trail = [_fill(rng.choice(TRAILS[reg]), style.address)] if TRAILS[reg] and rng.random() < p else []
    # detail first: the true lines open and a remark may follow; detail last: address and commands come first
    parts = body + trail if style.detail_position == "first" else lead + trail + body
    return " ".join(x for x in parts if x).strip()


__all__ = ["VoiceStyle", "style_for", "style_from_mind", "style_from_game", "render", "WRAPPERS", "CENTROIDS"]
