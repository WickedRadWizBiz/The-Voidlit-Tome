"""Characters: four attributes, a Hull track, Stress and a small kit.

Grit, Guile and Insight are also the three *approaches* a player can take to an obstacle. Resolve is will: it is
never an approach, but it holds against panic, intimidation and corruption.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

from . import checks
from .attributes import Attribute, mod
from .checks import CheckLog, CheckRecord


class Approach(str, Enum):
    GRIT = "grit"        # force, endurance, violence
    GUILE = "guile"      # stealth, trickery, speed, precision
    INSIGHT = "insight"  # analysis, systems, perception
    NONE = "none"

    @property
    def attribute(self) -> Attribute:
        return Attribute(self.value) if self is not Approach.NONE else Attribute.GRIT


ATTRIBUTES = (Approach.GRIT, Approach.GUILE, Approach.INSIGHT)  # the approaches
MAX_ATTRIBUTE = 4
MAX_STRESS = 10
PANIC_THRESHOLD = 4  # gaining stress at or above this triggers a panic check


@dataclass
class Item:
    name: str
    tags: tuple[str, ...] = ()
    bonus: dict[str, int] = field(default_factory=dict)  # approach or intent -> steps (+2 each on a d20)
    uses: int | None = None  # resource pool; None = never runs out
    light: int = 0  # Lumen dice restored when spent as a light source

    def bonus_for(self, approach: Approach, intent: str) -> int:
        return self.bonus.get(approach.value, 0) + self.bonus.get(intent, 0)

    def to_dict(self) -> dict:
        return {"name": self.name, "tags": list(self.tags), "bonus": dict(self.bonus),
                "uses": self.uses, "light": self.light}

    @classmethod
    def from_dict(cls, d: dict) -> "Item":
        return cls(name=d["name"], tags=tuple(d.get("tags", ())), bonus=dict(d.get("bonus", {})),
                   uses=d.get("uses"), light=d.get("light", 0))


@dataclass
class HarmResult:
    hull_lost: int
    attribute_lost: int
    last_light: CheckRecord | None
    dead: bool


@dataclass
class Character:
    name: str
    background: str
    grit: int
    guile: int
    insight: int
    hull_max: int
    resolve: int = 2
    hull: int = -1
    stress: int = 0
    conditions: set[str] = field(default_factory=set)
    inventory: list[Item] = field(default_factory=list)
    dead: bool = False

    def __post_init__(self) -> None:
        if self.hull < 0:
            self.hull = self.hull_max

    # -- checks ---------------------------------------------------------
    def attribute(self, attr: Attribute | Approach) -> int:
        if attr is Approach.NONE:
            return 0
        return getattr(self, attr.value)

    def modifier_for(self, attr: Attribute | Approach, intent: str = "") -> tuple[int, list[str]]:
        """d20 modifier for an action plus the reasons for each situational change (internal only)."""
        value = mod(self.attribute(attr))
        notes: list[str] = []
        approach = attr if isinstance(attr, Approach) else (Approach(attr.value) if attr is not Attribute.RESOLVE
                                                             else Approach.NONE)
        best_item, best_bonus = None, 0
        for item in self.inventory:
            if item.uses == 0:
                continue
            b = item.bonus_for(approach, intent)
            if b > best_bonus:
                best_item, best_bonus = item, b
        if best_item is not None:
            value += 2 * best_bonus
            notes.append(f"+{2 * best_bonus} {best_item.name}")
        if "shaken" in self.conditions:
            value -= 2
            notes.append("-2 shaken")
        if "bleeding" in self.conditions and attr in (Approach.GRIT, Attribute.GRIT):
            value -= 2
            notes.append("-2 bleeding")
        return value, notes

    def check(self, action: str, attr: Attribute | Approach, dc: int, rng: random.Random, log: CheckLog | None,
              intent: str = "", bonus: int = 0, upper_hand: bool = False) -> CheckRecord:
        """A direct action: d20 + modifier (+ situational ``bonus``) against ``dc``."""
        value, _ = self.modifier_for(attr, intent)
        name = attr.attribute.value if isinstance(attr, Approach) else attr.value
        return checks.d20(rng, log, "player", action, name, value + bonus, dc, upper_hand=upper_hand)

    def spend_item(self, item: Item) -> int:
        """Spend one use of a limited item. Returns uses left (-1 = unlimited)."""
        if item.uses is None:
            return -1
        item.uses = max(0, item.uses - 1)
        return item.uses

    def find_item(self, text: str, tag: str | None = None) -> Item | None:
        low = text.lower()
        usable = [i for i in self.inventory if i.uses != 0]
        for item in usable:
            words = [w for w in item.name.lower().split() if len(w) > 2]
            if item.name.lower() in low or any(w in low for w in words):
                return item
        if tag:
            for item in usable:
                if tag in item.tags:
                    return item
        return None

    # -- harm & stress --------------------------------------------------
    def take_harm(self, amount: int, rng: random.Random, log: CheckLog | None = None) -> HarmResult:
        """Hull soaks harm first. Past zero Hull, Grit is the backup, and a Last Light save (a passive
        d6 + Grit against 5 + the overflow) decides if you stay standing."""
        if amount <= 0 or self.dead:
            return HarmResult(0, 0, None, self.dead)
        soaked = min(self.hull, amount)
        self.hull -= soaked
        overflow = amount - soaked
        if overflow <= 0:
            return HarmResult(soaked, 0, None, False)
        save = checks.d6(rng, log, "player", "passive.last_light", "grit", self.grit, 5 + overflow)
        if not save.degree.succeeded:
            self.dead = True
            return HarmResult(soaked, 0, save, True)
        self.grit = max(0, self.grit - 1)
        self.conditions.add("bleeding")
        return HarmResult(soaked, 1, save, False)

    def heal(self, amount: int) -> int:
        before = self.hull
        self.hull = min(self.hull_max, self.hull + amount)
        if self.hull == self.hull_max:
            self.conditions.discard("bleeding")
        return self.hull - before

    def add_stress(self, amount: int) -> bool:
        """Add stress. Returns True if a panic check is now due."""
        if amount <= 0:
            return False
        self.stress = min(MAX_STRESS, self.stress + amount)
        return self.stress >= PANIC_THRESHOLD

    def relieve_stress(self, amount: int) -> int:
        before = self.stress
        self.stress = max(0, self.stress - amount)
        if self.stress < PANIC_THRESHOLD:
            self.conditions.discard("shaken")
        return before - self.stress

    # -- serialization ----------------------------------------------------
    def sheet(self) -> dict:
        """The character sheet a player may see at the start: scores, but no dice, counts or conditions."""
        return {"name": self.name, "background": self.background, "grit": self.grit, "guile": self.guile,
                "insight": self.insight, "resolve": self.resolve, "hull_max": self.hull_max,
                "inventory": [i.name for i in self.inventory]}

    def to_dict(self) -> dict:
        return {
            "name": self.name, "background": self.background,
            "grit": self.grit, "guile": self.guile, "insight": self.insight, "resolve": self.resolve,
            "hull_max": self.hull_max, "hull": self.hull, "stress": self.stress,
            "conditions": sorted(self.conditions),
            "inventory": [i.to_dict() for i in self.inventory], "dead": self.dead,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Character":
        return cls(
            name=d["name"], background=d["background"], grit=d["grit"], guile=d["guile"],
            insight=d["insight"], resolve=d.get("resolve", 2), hull_max=d["hull_max"], hull=d["hull"], stress=d["stress"],
            conditions=set(d.get("conditions", ())),
            inventory=[Item.from_dict(i) for i in d.get("inventory", ())], dead=d.get("dead", False),
        )
