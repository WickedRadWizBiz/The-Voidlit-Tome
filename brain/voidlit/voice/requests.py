"""What the voice is asked to say. The rules have already decided everything here.

Narration and the Tome's speech are always worded by the authored (offline) voice; only a dialogue request
reaches a local model, for an NPC's own words."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..cortex.features import Beat
from ..limbic.npc_brain import BrainDecision, NPCSpec


@dataclass
class NarrationRequest:
    intent: str
    chapter_text: str
    entering_chapter: bool             # the scene is told (on entering it, or on a look around)
    outcome: str | None = None         # fail | mixed | success | critical | None (no roll)
    variant: str | None = None         # special case of the intent: blocked, denied, light, heal, npc, ...
    npc_name: str | None = None        # who answered, when the player was talking to someone
    obstacle_name: str | None = None
    defeated_text: str | None = None
    wounded: bool = False              # the player was hurt this turn
    beat: Beat = Beat.CALM
    beat_fact: str = ""                # a concrete in-world event caused by the beat
    clue: str | None = None
    ending: str | None = None          # "won" | "dead" | None
    register: str = ""                 # the Tome's hidden register (wording only; never shown)


@dataclass
class TomeSpeechRequest:
    """The Tome's own words this turn: speech acts already chosen and checked by the ledger (tome.ledger)."""

    lines: list[str]                   # each a ledger fact, deflection, prediction, opinion or question: said verbatim
    style: object                      # voice.style.VoiceStyle (hidden; wording only)
    caution: str | None = None         # a true, already-public ledger fact a warm Tome may add unasked


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
