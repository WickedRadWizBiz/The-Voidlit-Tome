"""What the voice is asked to say. The rules have already decided everything here."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..cortex.features import Beat
from ..limbic.npc_brain import BrainDecision, NPCSpec


@dataclass
class NarrationRequest:
    player_text: str
    intent: str
    stat: str
    register: str                      # plain | uneasy | eldritch | breakthrough (from the Bearer's Stress)
    register_style: str                # the register's style note, from content
    facts: list[str]                   # mechanical truths, already resolved, in order
    place: str = ""
    leaf_title: str = ""
    leaf_kind: str = ""                # hostile | hazard | relic | witness | crucible | ""
    leaf_text: str = ""                # the Leaf's own prose in this register, when it has one
    entering: bool = False             # a Leaf just rose: describe it
    outcome: str | None = None         # fail | short | success | None (no cast)
    variant: str | None = None         # special cases: lost, refused, npc, alone, look, wait, ...
    npc_name: str | None = None
    cleared_text: str | None = None    # what the place says when the ring fills
    beat: Beat = Beat.CALM
    beat_fact: str = ""
    beat_note: str = ""
    clue: str | None = None
    tension: float = 0.0
    lumen: float = 1.0
    history: list[str] = field(default_factory=list)  # last few narrated lines, oldest first
    ending: str | None = None          # verdict | dead | taken | campaign | None


@dataclass
class DialogueRequest:
    npc: NPCSpec
    decision: BrainDecision
    system_prompt: str
    player_text: str
    player_name: str
    unprompted: bool = False
    source_name: str | None = None     # who the rumor came from, if any
    history: list[str] = field(default_factory=list)
