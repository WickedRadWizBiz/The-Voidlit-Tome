"""The Director's view of the table: game state squeezed into [0, 1] features."""

from __future__ import annotations

from dataclasses import astuple, dataclass, fields
from enum import Enum

import torch


class Beat(str, Enum):
    CALM = "calm"          # let it breathe: atmosphere, no new pressure
    ESCALATE = "escalate"  # a threat clock ticks; danger closes in
    CLUE = "clue"          # reveal something true about the sector
    NPC = "npc"            # someone acts on their own agenda
    HAZARD = "hazard"      # the environment turns on the player
    RESPITE = "respite"    # a chance to recover
    TWIST = "twist"        # a complication or reversal
    ADVANCE = "advance"    # nudge the story toward the next chapter


BEATS: tuple[Beat, ...] = tuple(Beat)


@dataclass
class DirectorFeatures:
    lumen: float = 1.0
    hull: float = 1.0
    stress: float = 0.0
    progress: float = 0.0
    tension: float = 0.2
    since_threat: float = 0.0
    since_respite: float = 0.0
    chapter_turns: float = 0.0
    obstacle_active: float = 0.0
    obstacle_left: float = 0.0
    last_fail: float = 0.0
    last_mixed: float = 0.0
    last_success: float = 0.0
    aggression: float = 0.0
    boldness: float = 0.0
    clock_max: float = 0.0
    npc_present: float = 0.0
    npc_hostility: float = 0.0

    def tensor(self) -> torch.Tensor:
        return torch.tensor(astuple(self), dtype=torch.float).clamp(0.0, 1.0)

    @classmethod
    def size(cls) -> int:
        return len(fields(cls))

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "DirectorFeatures":
        return cls(*[float(x) for x in t.tolist()])


N_FEATURES = DirectorFeatures.size()
