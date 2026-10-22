"""Offline voice: authored phrase banks, chosen by seeded dice. No model needed.

Scenario content can add or override any bank key. Keys look like
``action.<intent>.<outcome>`` (where outcome is fail, mixed, success,
critical or ``none``), ``action.any.<outcome>``, ``beat.<beat>``, ``bite``
(the player was hurt), ``npc.rumor`` and ``npc.regret``.

A narration is made of *core* parts, which say what the rules decided this
turn (the scene on entering, the action's outcome line, a defeat, a clue, a
beat's event, a wound, the ending), and *colour*. This voice is the only one
that narrates or speaks for the Tome, with or without a local model: every
word of it is authored (voice/llama_local.py gives a model the NPCs'
dialogue only). Each line must be true in every state it can be said in: lit
or dark, whatever the obstacle is made of, and colour (a beat's ambience)
says nothing of the place the player stands in that its scene does not.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence

from ..cortex.features import Beat
from .requests import DialogueRequest, NarrationRequest, TomeSpeechRequest
from .style import render

DEFAULT_BANKS: dict[str, list[str]] = {
    "action.any.fail": ["It goes wrong at once.", "The dark does not cooperate.", "Nothing gives."],
    # An outcome line is said whether or not the obstacle falls and whether or not it bites back (a defeat and a wound
    # are told by their own lines), so none claims either.
    "action.any.mixed": ["It works, but only just.", "Something gives, though not cleanly."],
    "action.any.success": ["It works. For once, the station lets you have this.", "Clean. Quick."],
    "action.any.critical": ["Better than you dared. For a breath, you feel unkillable."],
    "action.any.none": ["The Tome's surface ripples, waiting.", "You hold still. The station breathes around you."],
    "action.fight.fail": ["You lunge for {obstacle} and meet only its answer.",
                          "Your strike glances off {obstacle}, which was waiting for exactly that."],
    "action.fight.mixed": ["You hurt {obstacle}, but it is a near thing.", "Steel finds {obstacle}, and only just."],
    "action.fight.success": ["You drive into {obstacle}, and something gives.", "{obstacle} reels under the blow."],
    "action.fight.critical": ["You find the seam in {obstacle} and drive into it, hard."],
    "action.force.fail": ["You strain against {obstacle}, and nothing moves."],
    "action.force.mixed": ["{obstacle} gives a little, grudgingly."],
    "action.force.success": ["{obstacle} gives under your weight."],
    "action.force.critical": ["{obstacle} buckles before you like wet paper."],
    "action.sneak.fail": ["A boot scuffs. {obstacle} turns toward the sound."],
    "action.sneak.mixed": ["You slip close to {obstacle}, but it is a near thing."],
    "action.sneak.success": ["You become a shadow. {obstacle} never knows."],
    "action.sneak.critical": ["You move like the dark itself; {obstacle} never sees you coming."],
    "action.search.fail": ["Panels, dust, dead screens. And the feeling of being watched."],
    "action.search.mixed": ["You find something, but it takes too long."],
    "action.search.success": ["Your fingers find what your eyes missed."],
    "action.search.critical": ["It all clicks: the pattern, the purpose, the thing everyone missed."],
    "action.search.empty": ["You turn the place over and come away with nothing new."],   # everything is found
    "action.search.none": ["You look, and the station looks back."],
    "action.look.none": ["You take it in, slowly, one surface at a time."],
    "action.rest.none": ["You sit with your back to cold metal and let your heart slow."],
    "action.flee.fail": ["You run, and {obstacle} is faster."],
    "action.flee.mixed": ["You break away from {obstacle}, but it marks you as you go."],
    "action.flee.success": ["You put distance and darkness between you and {obstacle}."],
    "action.advance.none": ["You move on, and the dark folds in behind you."],
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
    "action.use.light_full": ["Light hisses alive in your hand and you feed it to the Tome, but it burns no brighter."],
    "action.use.heal": ["You patch what you can. It isn't pretty, but it holds."],
    # a blow at a person, which they get clear of whether or not it lands (an aside tells a blow that lands)
    "action.fight.npc_attack": ["{npc} twists away from you, eyes wide, and the dark takes them."],
    "action.force.npc_attack": ["You lunge at {npc}, who wrenches clear, eyes wide, and the dark takes them."],
    "action.fight.none": ["You lash out at shadows. Nothing here fights back, yet."],
    "action.force.none": ["You put your shoulder into the station. It doesn't notice."],
    "action.sneak.none": ["You move as quietly as you can, one careful step at a time."],
    "action.use.none": ["You try it, but nothing here answers to it."],
    "action.other.none": ["Nothing in the station answers to that."],
    "action.flee.none": ["You back away. Nothing follows. That's almost worse."],
    "action.talk.none": ["Your words hang in the stale air."],
    "action.talk.alone": ["You speak into the dark. The dark keeps its own counsel."],
    "action.any.lost_turn": ["Panic holds you. The moment passes without you."],
    # the Tome answers (it always does when this is said), or keeps silent (a lie it took in)
    "action.any.tome": ["Words rise under the black glass to meet you.", "The Tome's surface stirs under your hand.",
                        "The glass warms as it answers."],
    "action.any.tome_silent": ["The black glass stays still. The Tome takes your words in silence."],
    # what a blow or a hand over its glow came to; only a lit Tome has a glow to crack or to hide
    "action.any.tome_struck": ["Your blow lands. One of the Tome's lights cracks and goes out."],
    "action.any.tome_struck_dark": ["Your blow lands on dark glass. Every one of its lights is already out."],
    "action.any.tome_unmoved": ["Your blow skids off the Tome's black face."],
    "action.any.tome_smothered": ["You smother the Tome's glow. For a moment you are hard to see."],
    "action.any.tome_smother_fail": ["You try to smother the Tome's glow, but it bleeds through your fingers."],
    "action.any.tome_smother_dark": ["You cover the Tome, but there is no glow left in it to hide."],
    # anything else done to the Tome (lifting it, reading it, using it): nothing comes of it
    "action.any.tome_idle": ["The Tome lets you, and nothing comes of it."],
    "action.talk.npc": ["{npc} studies you before answering.", "{npc} takes a moment with your words.",
                        "{npc} glances at the Tome, then back to you."],
    "action.talk.fail": ["Your words come out wrong, and {npc} hears every crack in them."],
    "action.talk.mixed": ["{npc} weighs what you said, unconvinced but listening."],
    "action.talk.success": ["Your words land exactly where you meant them to. {npc} goes still."],
    "action.talk.critical": ["You say it perfectly. {npc} has no answer ready."],
    "action.any.blocked": ["{obstacle} is still in the way."],
    "bite": ["Pain arrives a half-second late.", "A breath later, the hurt catches up with you."],
    "beat.calm": ["Somewhere far off, a pipe ticks as it cools.", "For a moment, only the hum of dying machines.",
                  "Somewhere above you, the station creaks and settles."],
    # an escalation or a twist moved a danger on out of sight (a clock ticked): told as just that, never as an event in
    # the player's own scene
    "beat.escalate": ["Deep in the hull, something heavy shifts its weight.",
                      "Somewhere out of sight, something that was waiting starts to move.",
                      "Far off, the station stirs, and does not settle again."],
    "beat.clue": ["Something here wants to be understood."],
    "beat.npc": [""],
    "beat.hazard": ["The station turns on you.", "The station presses in on you, and your nerves fray."],
    "beat.respite": ["The pressure eases. You can breathe here, briefly.",
                     "For a moment nothing moves, and you get your breath back."],
    "beat.twist": ["Somewhere out of sight, something has turned against you.",
                   "Wrong. Something about this is wrong."],
    "beat.advance": ["The way onward stands open, and the dark behind you is growing patient.",
                     "Nothing stands between you and the way on now."],
    "npc.rumor": ["Funny. I heard a different story about you.", "People talk, even down here."],
    "npc.regret": ["I wanted to trust you. You made that impossible.", "We could have walked out of here together."],
    "ending.won": ["The airlock seals behind you, and the Tome settles against your chest like a sleeping thing. "
                   "You are out. Whatever is left in the Hollow Lantern will have to keep its own dark."],
    "ending.dead": ["Your light goes out before you do, but not by much. The Tome closes over your name, "
                    "and the station keeps you."],
}


def _capitalize(s: str) -> str:
    return s[:1].upper() + s[1:]


def _fill(line: str, slot: str, name: str) -> str:
    """A bank line with a name in its slot, capitalised where the slot opens a sentence ("A boot scuffs. The jammed
    iris door turns toward the sound.")."""
    return re.sub(rf"(^|[.!?]\s+)?{re.escape(slot)}",
                  lambda m: m.group(1) + _capitalize(name) if m.group(1) is not None else name, line)


class OfflineVoice:
    def __init__(self, banks: Mapping[str, Sequence[str]] | None = None, seed: int = 0) -> None:
        self.banks: dict[str, list[str]] = {k: list(v) for k, v in DEFAULT_BANKS.items()}
        for k, v in (banks or {}).items():
            self.banks[k] = list(v)
        self.rng = random.Random(seed)

    def _pick(self, *keys: str, register: str = "") -> str:
        if register:                                   # a register variant first: ``beat.calm@cold``
            keys = tuple(x for k in keys for x in (f"{k}@{register}", k))
        for k in keys:
            options = [o for o in self.banks.get(k, []) if o]
            if options:
                return self.rng.choice(options)
        return ""

    def narration_parts(self, req: NarrationRequest) -> list[tuple[str, bool]]:
        """This turn's narration as (text, core) parts, in order, chosen once. Core parts are what the rules decided;
        the rest is colour. On the last turn the outcome, a wound, the defeat, clue and beat event are told before the
        ending."""
        parts: list[tuple[str, bool]] = []
        if req.entering_chapter:
            parts.append((req.chapter_text, True))
        outcome = req.outcome or "none"
        if not req.ending or req.outcome or req.variant:
            keys = [f"action.{req.intent}.{req.variant}", f"action.any.{req.variant}"] if req.variant else []
            keys += [f"action.{req.intent}.{outcome}", f"action.any.{outcome}"]
            line = self._pick(*keys)
            if line and not (req.entering_chapter and req.intent == "advance"):
                line = _fill(line, "{obstacle}", req.obstacle_name or "the dark")
                parts.append((_fill(line, "{npc}", req.npc_name or "They"), True))
        if req.wounded:
            parts.append((self._pick("bite"), True))
        parts += [(x, True) for x in (req.defeated_text, req.clue, req.beat_fact) if x]
        if req.ending:
            parts.append((self._pick(f"ending.{req.ending}"), True))
        elif not req.beat_fact and req.beat is not Beat.NPC and not (req.beat is Beat.CLUE and req.clue):
            parts.append((self._pick(f"beat.{req.beat.value}", register=req.register), False))
        return [(_capitalize(x), core) for x, core in parts if x]

    @staticmethod
    def join(parts: list[tuple[str, bool]]) -> str:
        return " ".join(x for x, _ in parts if x)

    def narrate(self, req: NarrationRequest) -> str:
        return self.join(self.narration_parts(req))

    def tome_say(self, req: TomeSpeechRequest) -> str:
        return render(req.lines, req.style, self.rng, caution=req.caution)

    def speak(self, req: DialogueRequest) -> str:
        tactic = req.npc.tactics[req.decision.tactic]
        line = self.rng.choice(tactic.lines) if tactic.lines else "..."
        role = req.npc.plots[req.decision.dominant_plot].role
        if role == "betray" and req.decision.branch_origin == "ally" and self.rng.random() < 0.5:
            line = f"{self._pick('npc.regret')} {line}"
        elif req.decision.branch_source_npc and self.rng.random() < 0.4:
            line = f"{self._pick('npc.rumor')} {line}"
        return line.strip()
