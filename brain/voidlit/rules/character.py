"""Characters: three attributes, a Hull track, Stress and a small kit."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

from .dice import MAX_POOL, PoolRoll, roll_pool


class Approach(str, Enum):
    GRIT = "grit"        # force, endurance, violence
    GUILE = "guile"      # stealth, trickery, speed, precision
    INSIGHT = "insight"  # analysis, systems, perception, will
    NONE = "none"


ATTRIBUTES = (Approach.GRIT, Approach.GUILE, Approach.INSIGHT)
MAX_ATTRIBUTE = 4
MAX_STRESS = 10
PANIC_THRESHOLD = 4  # gaining stress at or above this triggers a panic check


@dataclass
class Item:
    name: str
    tags: tuple[str, ...] = ()
    bonus: dict[str, int] = field(default_factory=dict)  # approach or intent -> extra dice
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
    last_light: PoolRoll | None
    dead: bool


@dataclass
class Character:
    name: str
    background: str
    grit: int
    guile: int
    insight: int
    hull_max: int
    hull: int = -1
    stress: int = 0
    conditions: set[str] = field(default_factory=set)
    inventory: list[Item] = field(default_factory=list)
    dead: bool = False

    def __post_init__(self) -> None:
        if self.hull < 0:
            self.hull = self.hull_max

    # -- dice -----------------------------------------------------------
    def attribute(self, approach: Approach) -> int:
        if approach is Approach.NONE:
            return 0
        return getattr(self, approach.value)

    def pool_for(self, approach: Approach, intent: str = "") -> tuple[int, list[str]]:
        """Pool size for an action plus a list of the reasons for each modifier."""
        pool = self.attribute(approach)
        notes: list[str] = []
        best_item = None
        best_bonus = 0
        for item in self.inventory:
            if item.uses == 0:
                continue
            b = item.bonus_for(approach, intent)
            if b > best_bonus:
                best_item, best_bonus = item, b
        if best_item is not None:
            pool += best_bonus
            notes.append(f"+{best_bonus} {best_item.name}")
        if "shaken" in self.conditions:
            pool -= 1
            notes.append("-1 shaken")
        if "bleeding" in self.conditions and approach is Approach.GRIT:
            pool -= 1
            notes.append("-1 bleeding")
        return max(0, min(MAX_POOL, pool)), notes

    def roll(self, approach: Approach, rng: random.Random, intent: str = "",
             bonus: int = 0) -> tuple[PoolRoll, list[str]]:
        pool, notes = self.pool_for(approach, intent)
        return roll_pool(pool + bonus, rng), notes

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
    def take_harm(self, amount: int, rng: random.Random) -> HarmResult:
        """Hull soaks harm first. Past zero Hull, Grit is the backup and a
        Last Light save decides if you stay standing."""
        if amount <= 0 or self.dead:
            return HarmResult(0, 0, None, self.dead)
        soaked = min(self.hull, amount)
        self.hull -= soaked
        overflow = amount - soaked
        if overflow <= 0:
            return HarmResult(soaked, 0, None, False)
        save = roll_pool(self.grit, rng)
        if save.hits == 0:
            self.dead = True
            return HarmResult(soaked, 0, save, True)
        lost = max(1, min(2, len(save.dice) - save.hits))
        self.grit = max(0, self.grit - lost)
        self.conditions.add("bleeding")
        return HarmResult(soaked, lost, save, False)

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
    def to_dict(self) -> dict:
        return {
            "name": self.name, "background": self.background,
            "grit": self.grit, "guile": self.guile, "insight": self.insight,
            "hull_max": self.hull_max, "hull": self.hull, "stress": self.stress,
            "conditions": sorted(self.conditions),
            "inventory": [i.to_dict() for i in self.inventory], "dead": self.dead,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Character":
        return cls(
            name=d["name"], background=d["background"], grit=d["grit"], guile=d["guile"],
            insight=d["insight"], hull_max=d["hull_max"], hull=d["hull"], stress=d["stress"],
            conditions=set(d.get("conditions", ())),
            inventory=[Item.from_dict(i) for i in d.get("inventory", ())], dead=d.get("dead", False),
        )
