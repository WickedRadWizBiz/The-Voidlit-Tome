"""Offline voice: authored phrase banks, chosen by seeded dice. No model needed.

Place packs can add or override any bank key. Keys look like ``action.<kind>.<outcome>`` (kind is hostile,
hazard, relic, witness or crucible; outcome is fail, short or success), ``action.any.<outcome>``,
``variant.<name>``, ``register.<register>``, ``beat.<beat>``, ``ending.<ending>``, ``npc.rumor`` and
``npc.regret``. Facts from the rules are already written in the TOME's voice and are always included.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

from ..cortex.features import Beat
from .requests import DialogueRequest, NarrationRequest

DEFAULT_BANKS: dict[str, list[str]] = {
    "action.any.fail": ["It goes wrong at once.", "The dark does not cooperate.", "Nothing gives except your nerve."],
    "action.any.short": ["It gives, a little, and takes something back.", "Not enough. It costs you."],
    "action.any.success": ["It works. For once, the dark lets you have this.", "Clean. Quick. Done."],
    "action.hostile.fail": ["You go at it and meet only its answer.", "It was waiting for exactly that."],
    "action.hostile.short": ["You hurt it, and it pays you back in kind."],
    "action.hostile.success": ["You drive into it, and something gives.", "It reels."],
    "action.hazard.fail": ["The place does not care what you meant to do."],
    "action.hazard.short": ["You get partway through, and the place takes its toll."],
    "action.hazard.success": ["You find the rhythm of the place and move with it."],
    "action.relic.fail": ["It does not open to you. It looks back."],
    "action.relic.short": ["Something turns over inside it, and something in you turns with it."],
    "action.relic.success": ["Your fingers find what your eyes missed."],
    "action.witness.fail": ["Your words come out wrong, and the silence after them is worse."],
    "action.witness.short": ["You hold its gaze, mostly."],
    "action.witness.success": ["You stand there and take it, and you are still yourself afterward."],
    "action.crucible.fail": ["The Crucible turns your strength back on you."],
    "action.crucible.short": ["You hurt it. It hurts you more."],
    "action.crucible.success": ["Something in the Crucible breaks, and it is not you."],
    "variant.look": ["You take it in, slowly. The TOME's light trembles over every surface."],
    "variant.wait": ["The stone is still. It is waiting for you to choose how."],
    "variant.alone": ["You speak into the dark. The dark keeps its own counsel."],
    "variant.npc": ["{npc} studies you before answering.", "{npc} glances at the slab, then back to you."],
    "variant.npc_attack": ["{npc} twists away from you, eyes wide, and the dark takes them."],
    "variant.lost": ["Your body will not answer you."],
    "variant.touch": ["The stone ripples under your touch."],
    "variant.haven": ["The TOME is quiet. For now, so is everything else."],
    "register.uneasy": ["Something about it is slightly wrong, and stays wrong.",
                        "You look at it twice and it is the same both times, which is worse."],
    "register.eldritch": ["Its angles do not agree with one another, and the longer you look the older it becomes.",
                          "It has the patience of a thing that was here before the stars learned to burn."],
    "register.breakthrough": ["MINE, the stone says, inside the sentence, in violet. You keep reading.",
                              "The words on the slab rearrange themselves while you read them. YOU ARE LATE."],
    "beat.calm": ["Somewhere far off, metal ticks as it cools.", "For a moment, only the hum of dying machines."],
    "beat.foreshadow": ["Somewhere ahead, something hears you.", "The TOME's surface stirs, then lies flat again."],
    "beat.clue": ["Something here wants to be understood."],
    "beat.npc": [""],
    "beat.dot": [""], "beat.shapes": [""], "beat.dark": [""],
    "npc.rumor": ["Funny. I heard a different story about you.", "People talk, even out here."],
    "npc.regret": ["I wanted to trust you. You made that impossible.", "We could have walked out of here together."],
    "ending.verdict": ["The Crucible sinks into the stone, and the TOME closes under your hand. It lets you go. "
                       "For now."],
    "ending.dead": ["Your light goes out before you do, but not by much. The TOME closes over your name and "
                    "waits for the next pair of hands."],
    "ending.taken": ["You do not die. The TOME simply has you now, the way it has always meant to."],
    "ending.campaign": ["The last glyph translates itself, and you understand what it wanted all along."],
}


def _capitalize(s: str) -> str:
    return s[:1].upper() + s[1:]


class OfflineVoice:
    def __init__(self, banks: Mapping[str, Sequence[str]] | None = None, seed: int = 0) -> None:
        self.banks: dict[str, list[str]] = {k: list(v) for k, v in DEFAULT_BANKS.items()}
        for k, v in (banks or {}).items():
            self.banks[k] = list(v)
        self.rng = random.Random(seed)

    def _pick(self, *keys: str) -> str:
        for k in keys:
            options = [o for o in self.banks.get(k, []) if o]
            if options:
                return self.rng.choice(options)
        return ""

    def narrate(self, req: NarrationRequest) -> str:
        parts: list[str] = []
        if req.entering:
            parts.append(req.leaf_text or req.leaf_title)
            if not req.leaf_text and req.register != "plain":
                parts.append(self._pick(f"register.{req.register}"))
        if req.variant:
            parts.append(self._pick(f"variant.{req.variant}").replace("{npc}", req.npc_name or "They"))
        if req.outcome:
            parts.append(self._pick(f"action.{req.leaf_kind}.{req.outcome}", f"action.any.{req.outcome}"))
        parts += req.facts
        if req.cleared_text:
            parts.append(req.cleared_text)
        if req.clue:
            parts.append(req.clue)
        if req.ending:
            parts.append(self._pick(f"ending.{req.ending}"))
        elif req.beat_fact:
            parts.append(req.beat_fact)
        elif req.beat is not Beat.NPC and not (req.beat is Beat.CLUE and req.clue):
            parts.append(self._pick(f"beat.{req.beat.value}"))
        if req.register == "breakthrough" and not req.entering:
            parts.append(self._pick("register.breakthrough"))
        return " ".join(_capitalize(p) for p in parts if p)

    def speak(self, req: DialogueRequest) -> str:
        tactic = req.npc.tactics[req.decision.tactic]
        line = self.rng.choice(tactic.lines) if tactic.lines else "..."
        role = req.npc.plots[req.decision.dominant_plot].role
        if role == "betray" and req.decision.branch_origin == "ally" and self.rng.random() < 0.5:
            line = f"{self._pick('npc.regret')} {line}"
        elif req.decision.branch_source_npc and self.rng.random() < 0.4:
            line = f"{self._pick('npc.rumor')} {line}"
        return line.strip()
