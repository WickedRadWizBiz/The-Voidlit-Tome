"""What the voice is asked to say. The rules have already decided everything here."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..cortex.features import Beat
from ..limbic.npc_brain import BrainDecision, NPCSpec


@dataclass
class NarrationRequest:
    player_text: str
    intent: str
    approach: str
    chapter_title: str
    chapter_text: str
    entering_chapter: bool
    facts: list[str]                   # mechanical truths, already resolved, in order
    outcome: str | None = None         # fail | mixed | success | critical | None (no roll)
    variant: str | None = None         # special case of the intent: blocked, denied, light, heal, npc, ...
    npc_name: str | None = None        # who answered, when the player was talking to someone
    obstacle_name: str | None = None
    obstacle_kind: str | None = None
    obstacle_text: str | None = None
    defeated_text: str | None = None
    beat: Beat = Beat.CALM
    beat_fact: str = ""                # a concrete in-world event caused by the beat (both voices use it)
    beat_note: str = ""                # what the beat means, for the LLM only
    clue: str | None = None
    tension: float = 0.0
    lumen: float = 1.0
    history: list[str] = field(default_factory=list)  # last few narrated lines, oldest first
    ending: str | None = None          # "won" | "dead" | None


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
