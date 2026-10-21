"""Panic: when Stress gets loud enough, fear takes the wheel.

A panic check is a passive event: d6 + Resolve + 3. Beat your Stress (+1 with a phobia) and you hold steady.
Otherwise you panic; the row is ``d6 + Stress // 2`` (clamped), so higher Stress pulls worse results. Panicking
discharges 2 Stress before the row's effect lands. The player never sees either die.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from . import checks
from .character import Character
from .checks import CheckLog


@dataclass(frozen=True)
class PanicEffect:
    name: str
    text: str
    effects: tuple[str, ...]  # machine-readable: "stress:+1", "condition:shaken", "lumen:-1", ...


PANIC_TABLE: tuple[PanicEffect, ...] = (
    PanicEffect("Adrenal Spike", "Fear sharpens you to a needle point.", ("stress:-1",)),
    PanicEffect("Cold Sweat", "Your hands won't stop shaking.", ("condition:shaken",)),
    PanicEffect("Ringing Silence", "Sound drops out of the world. You lose a beat.", ("lose_turn",)),
    PanicEffect("Fumble", "Something slips from your grip and clatters into the dark.", ("drop_item",)),
    PanicEffect("Whispering Static", "Your comm hisses words nobody said.", ("stress:+1",)),
    PanicEffect("Snap", "You turn on whoever is nearest, all teeth and accusation.", ("rapport:-1",)),
    PanicEffect("Bolt", "You run before you decide to, and slam into a bulkhead.", ("harm:1", "condition:shaken")),
    PanicEffect("Kill the Light", "You smother a lamp, convinced it gives you away.", ("lumen:-1",)),
    PanicEffect("Freeze", "Your body locks. The moment passes without you.", ("lose_turn", "condition:shaken")),
    PanicEffect("Fixation", "Something in this place hooks into you and stays.", ("condition:phobia",)),
    PanicEffect("Heart Stutter", "Your chest seizes. The world greys at the edges.", ("harm:2",)),
    PanicEffect("The Dark Answers", "You scream, and something in the black screams back.",
                ("lumen:-1", "stress:+2", "condition:shaken")),
)


@dataclass(frozen=True)
class PanicCheck:
    die: int
    stress: int
    held: bool
    row: int | None
    effect: PanicEffect | None

    def to_dict(self) -> dict:
        return {
            "die": self.die, "stress": self.stress, "held": self.held,
            "effect": None if self.effect is None else {"name": self.effect.name, "text": self.effect.text},
        }


def panic_check(char: Character, rng: random.Random, log: CheckLog | None = None) -> PanicCheck:
    effective = char.stress + (1 if "phobia" in char.conditions else 0)
    rec = checks.d6(rng, log, "player", "passive.panic", "resolve", char.resolve + 3, effective, strict=True)
    if rec.degree.succeeded:
        return PanicCheck(rec.natural, effective, True, None, None)
    row = max(0, min(len(PANIC_TABLE) - 1, rng.randint(1, 6) + effective // 2 - 1))
    return PanicCheck(rec.natural, effective, False, row, PANIC_TABLE[row])
