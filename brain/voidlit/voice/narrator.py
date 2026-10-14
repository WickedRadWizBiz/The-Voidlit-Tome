"""Offline voice: authored phrase banks, chosen by seeded dice. No model needed.

Scenario content can add or override any bank key. Keys look like
``action.<intent>.<outcome>`` (where outcome is fail, mixed, success,
critical or ``none``), ``action.any.<outcome>``, ``beat.<beat>``, ``bite``,
``npc.rumor`` and ``npc.regret``.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

from ..cortex.features import Beat
from .requests import DialogueRequest, NarrationRequest

DEFAULT_BANKS: dict[str, list[str]] = {
    "action.any.fail": ["It goes wrong at once.", "The dark does not cooperate.", "Nothing gives except your nerve."],
    "action.any.mixed": ["It works, but the station takes its toll.", "Something gives, and something else breaks."],
    "action.any.success": ["It works. For once, the station lets you have this.", "Clean. Quick. Done."],
    "action.any.critical": ["Better than you dared. For a breath, you feel unkillable."],
    "action.any.none": ["The Tome's surface ripples, waiting.", "You hold still. The station breathes around you."],
    "action.fight.fail": ["You lunge for {obstacle} and meet only its answer.",
                          "Your strike glances off {obstacle}, which was waiting for exactly that."],
    "action.fight.mixed": ["You hurt {obstacle}, and pay for it in kind.",
                           "Steel finds {obstacle}, but not before {obstacle} finds you."],
    "action.fight.success": ["You drive into {obstacle}, and something gives.", "{obstacle} reels under the blow."],
    "action.fight.critical": ["You find the seam in {obstacle} and tear it wide open."],
    "action.force.fail": ["You strain against {obstacle}. Nothing moves, except something behind it."],
    "action.force.mixed": ["{obstacle} groans and shifts, and the effort costs you."],
    "action.force.success": ["{obstacle} gives way with a shriek of metal."],
    "action.force.critical": ["{obstacle} comes apart in your hands like wet paper."],
    "action.sneak.fail": ["A boot scuffs. {obstacle} turns toward the sound."],
    "action.sneak.mixed": ["You slip past {obstacle}, but not unnoticed."],
    "action.sneak.success": ["You become a shadow. {obstacle} never knows."],
    "action.sneak.critical": ["You move like the dark itself; {obstacle} is behind you before it can blink."],
    "action.search.fail": ["Panels, dust, dead screens. And the feeling of being watched."],
    "action.search.mixed": ["You find something, but it takes too long."],
    "action.search.success": ["Your fingers find what your eyes missed."],
    "action.search.critical": ["It all clicks: the pattern, the purpose, the way through."],
    "action.search.none": ["You look, and the station looks back."],
    "action.look.none": ["You take it in, slowly. The Tome's light trembles over every surface."],
    "action.rest.none": ["You sit with your back to cold metal and let your heart slow."],
    "action.flee.fail": ["You run, and {obstacle} is faster."],
    "action.flee.mixed": ["You break away from {obstacle}, but it marks you as you go."],
    "action.flee.success": ["You put distance and darkness between you and {obstacle}."],
    "action.advance.none": ["You move on. The corridor swallows the light behind you."],
    "action.advance.fail": ["You dash past {obstacle}, and it catches you on the way through."],
    "action.advance.mixed": ["You make it past {obstacle}, heart hammering."],
    "action.advance.success": ["You slip past {obstacle} before it can touch you."],
    "action.advance.critical": ["You're past {obstacle} before it knows you were there."],
    "action.advance.blocked": ["You can't press on past {obstacle}. Not yet."],
    "action.any.unmatched": ["{obstacle} doesn't care about that. You'll need another way in.",
                             "Wrong tool, wrong instinct: {obstacle} shrugs it off."],
    "action.rest.denied": ["You try to rest, but {obstacle} won't let you close your eyes."],
    "action.rest.spent": ["You've taken what rest this place will give. The station won't allow more."],
    "action.use.light": ["Light hisses alive in your hand and you feed it to the Tome. The words burn brighter."],
    "action.use.heal": ["You patch what you can. It isn't pretty, but it holds."],
    "action.fight.npc_attack": ["{npc} twists away from you, eyes wide, and the dark takes them."],
    "action.fight.none": ["You lash out at shadows. Nothing here fights back, yet."],
    "action.force.none": ["You put your shoulder into the station. It doesn't notice."],
    "action.sneak.none": ["You move quietly through the dark, though nothing here is watching. You think."],
    "action.flee.none": ["You back away. Nothing follows. That's almost worse."],
    "action.talk.none": ["Your words hang in the stale air."],
    "action.talk.alone": ["You speak into the dark. The dark keeps its own counsel."],
    "action.talk.npc": ["{npc} studies you before answering.", "{npc} takes a moment with your words.",
                        "{npc} glances at the Tome's light, then back to you."],
    "action.talk.fail": ["Your words come out wrong, and {npc} hears every crack in them."],
    "action.talk.mixed": ["{npc} weighs what you said, unconvinced but listening."],
    "action.talk.success": ["Your words land exactly where you meant them to. {npc} goes still."],
    "action.talk.critical": ["You say it perfectly. {npc} has no answer ready."],
    "action.any.blocked": ["{obstacle} is still in the way."],
    "bite": ["Pain arrives a half-second late.", "It costs you. It always costs you."],
    "beat.calm": ["Somewhere far off, a pipe ticks as it cools.", "For a moment, only the hum of dying machines.",
                  "Frost creeps across a viewport, slow as breath."],
    "beat.escalate": ["Deep in the hull, something answers the noise you made.",
                      "The station shudders. A low choir of static rises and falls.",
                      "Far below, a bulkhead opens on its own."],
    "beat.clue": ["Something here wants to be understood."],
    "beat.npc": [""],
    "beat.hazard": ["The station turns on you.", "Warning lights wake, red and silent."],
    "beat.respite": ["The pressure eases. You can breathe here, briefly.",
                     "A pocket of warmth: a vent still pushing air. Rest, while you can."],
    "beat.twist": ["Then the floor changes its mind.", "Wrong. Something about this is wrong."],
    "beat.advance": ["The way onward stands open, and the dark behind you is growing patient.",
                     "There's nothing more for you here. The Tome's light leans toward the exit."],
    "npc.rumor": ["Funny. I heard a different story about you.", "People talk, even down here."],
    "npc.regret": ["I wanted to trust you. You made that impossible.", "We could have walked out of here together."],
    "ending.won": ["The airlock seals behind you. The Tome's light steadies, then dims to a sleeping glow. "
                   "You are out. Whatever is left in the Hollow Lantern will have to keep its own dark."],
    "ending.dead": ["Your light goes out before you do, but not by much. The Tome closes over your name, "
                    "and the station keeps you."],
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
        if req.entering_chapter:
            parts.append(req.chapter_text)
        if req.ending:
            parts.append(self._pick(f"ending.{req.ending}"))
            return " ".join(p for p in parts if p)
        outcome = req.outcome or "none"
        keys = [f"action.{req.intent}.{req.variant}", f"action.any.{req.variant}"] if req.variant else []
        keys += [f"action.{req.intent}.{outcome}", f"action.any.{outcome}"]
        line = self._pick(*keys)
        if line and not (req.entering_chapter and req.intent == "advance"):
            line = line.replace("{obstacle}", req.obstacle_name or "the dark")
            parts.append(line.replace("{npc}", req.npc_name or "They"))
        if req.defeated_text:
            parts.append(req.defeated_text)
        if req.clue:
            parts.append(req.clue)
        if req.beat_fact:
            parts.append(req.beat_fact)
        elif req.beat is not Beat.NPC and not (req.beat is Beat.CLUE and req.clue):
            parts.append(self._pick(f"beat.{req.beat.value}"))
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
