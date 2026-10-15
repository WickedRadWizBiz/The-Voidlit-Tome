"""The Director's view of the table: game state squeezed into [0, 1] features."""

from __future__ import annotations

from dataclasses import astuple, dataclass, fields
from enum import Enum

import torch


class Beat(str, Enum):
    # Narrative beats, chosen after a moment resolves. None of them changes a number.
    CALM = "calm"              # let it breathe
    FORESHADOW = "foreshadow"  # something is coming; say so without saying what
    CLUE = "clue"              # reveal a truth of this place
    NPC = "npc"                # someone present acts on their own agenda
    # How the TOME asks to be touched when a Leaf is ready (Core Rules 0.2, "The Draw").
    DOT = "dot"                # a single pulsing point of color
    SHAPES = "shapes"          # two or three shapes: the player chooses, the rest are lost
    DARK = "dark"              # nothing appears; the TOME waits


BEATS: tuple[Beat, ...] = tuple(Beat)
NARRATIVE: frozenset[Beat] = frozenset({Beat.CALM, Beat.FORESHADOW, Beat.CLUE, Beat.NPC})
DRAWS: frozenset[Beat] = frozenset({Beat.DOT, Beat.SHAPES, Beat.DARK})


@dataclass
class DirectorFeatures:
    lumen: float = 1.0          # Lumen dice / 6
    grit: float = 1.0           # Grit / 5
    stress: float = 0.0         # Stress / 10
    wounds: float = 0.0         # open Wounds / 3
    progress: float = 0.0       # chapters cleared / chapters needed
    tension: float = 0.2
    in_chapter: float = 0.0
    depth_left: float = 0.0     # what is left of the Depth ring
    crucible: float = 0.0
    last_fail: float = 0.0
    last_short: float = 0.0
    last_success: float = 0.0
    aggression: float = 0.0
    boldness: float = 0.0
    npc_present: float = 0.0
    npc_hostility: float = 0.0
    hunger: float = 0.0         # Hunger / 6

    def tensor(self) -> torch.Tensor:
        return torch.tensor(astuple(self), dtype=torch.float).clamp(0.0, 1.0)

    @classmethod
    def size(cls) -> int:
        return len(fields(cls))

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "DirectorFeatures":
        return cls(*[float(x) for x in t.tolist()])


N_FEATURES = DirectorFeatures.size()
