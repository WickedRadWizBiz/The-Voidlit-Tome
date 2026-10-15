"""The Bearer: four stats, Grit, Stress, Wounds, Brands and a kit (TOME: Core Rules 0.2).

Classes, Bonds, Openings and Origins are content (the TOME Forge's JSON). Two things the Core Rules state in
prose and the app needs as data live here, taken from the playtest simulator's final configuration:

* which chapter kinds each class's Talent covers (``TALENT_KINDS``), and
* which chapter kinds each kit item gives a die to (``ITEM_RULES``).

Both are flagged for the Forge's schema so they can move into content later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .dice import MAX_POOL

GRIT = 5
MAX_STAT = 4
MAX_STRESS = 10
PANIC_AT = 5          # gaining Stress at this level or above calls for a Panic Check
WOUNDS_FOR_RITES = 3  # three Wounds also means Last Rites
BRANDS_TO_BELONG = 2  # with two Brands you belong to it


class Stat(str, Enum):
    MIGHT = "might"
    GRACE = "grace"
    WITS = "wits"
    NERVE = "nerve"
    NONE = "none"


STATS: tuple[Stat, ...] = (Stat.MIGHT, Stat.GRACE, Stat.WITS, Stat.NERVE)


class Kind(str, Enum):
    HOSTILE = "hostile"
    HAZARD = "hazard"
    RELIC = "relic"
    WITNESS = "witness"


KINDS: tuple[Kind, ...] = (Kind.HOSTILE, Kind.HAZARD, Kind.RELIC, Kind.WITNESS)

# Hostile chapters can be met three ways; the mode decides whether a weapon's Damage counts.
FIGHT_MODES = ("melee", "ranged")

# Class Talents (+1 die), per the playtest's final_cfg.json. Hostile Talents only cover some modes.
TALENT_KINDS: dict[str, dict[Kind, tuple[str, ...] | None]] = {
    "breacher": {Kind.HOSTILE: FIGHT_MODES},
    "archivist": {Kind.RELIC: None, Kind.HAZARD: None},
    "vessel": {Kind.RELIC: None, Kind.HAZARD: None},
    "hauler": {Kind.HAZARD: None},
    "pilgrim": {Kind.WITNESS: None, Kind.HAZARD: None},
    "fence": {Kind.HOSTILE: ("hide",), Kind.WITNESS: None},
}

# What kit items do, by a keyword in their name. Damage/Guard/Uses come from the kit text itself.
ITEM_RULES: tuple[tuple[str, dict], ...] = (
    ("rifle", {"mode": "ranged"}),
    ("pistol", {"mode": "ranged"}),
    ("torch", {"mode": "melee"}),
    ("tool-arm", {"mode": "melee"}),
    ("blade", {"mode": "melee"}),
    ("armor", {}),
    ("scanner", {"helps": ("relic", "hazard")}),
    ("battery", {"helps": ("relic",)}),
    ("toolbelt", {"helps": ("hazard", "relic")}),
    ("lockpicks", {"helps": ("relic",)}),
    ("prayer cords", {"helps": ("witness",)}),
    ("medkit", {"tags": ("heal",)}),
)
COMMON_KIT = "a light, rations (Uses 3), medkit (Uses 2)"


@dataclass
class Item:
    name: str
    uses: int | None = None       # None: never runs out
    max_uses: int | None = None
    damage: int = 0
    mode: str = ""                # melee | ranged for weapons
    guard: int = 0
    helps: tuple[str, ...] = ()   # chapter kinds it adds a die to
    tags: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return self.uses is None or self.uses > 0

    def spend(self) -> None:
        if self.uses is not None:
            self.uses = max(0, self.uses - 1)

    def to_dict(self) -> dict:
        return {"name": self.name, "uses": self.uses, "max_uses": self.max_uses, "damage": self.damage,
                "mode": self.mode, "guard": self.guard, "helps": list(self.helps), "tags": list(self.tags)}

    @classmethod
    def from_dict(cls, d: dict) -> "Item":
        return cls(name=d["name"], uses=d.get("uses"), max_uses=d.get("max_uses"), damage=d.get("damage", 0),
                   mode=d.get("mode", ""), guard=d.get("guard", 0), helps=tuple(d.get("helps", ())),
                   tags=tuple(d.get("tags", ())))


_ITEM = re.compile(r"([^,(]+?)\s*(?:\(([^)]*)\))?\s*(?:,|$)")


def parse_kit(text: str) -> list[Item]:
    """'Rifle (Damage 2, Uses 3), armor (Guard 1, Uses 2)' -> items."""
    items = []
    for m in _ITEM.finditer(text.replace(" and ", ", ")):
        name = m.group(1).strip()
        if not name:
            continue
        name = re.sub(r"^(a|an|one)\s+", "", name, flags=re.I)
        stats = {k.lower(): int(v) for k, v in re.findall(r"(Damage|Uses|Guard)\s+(\d+)", m.group(2) or "")}
        item = Item(name=name, uses=stats.get("uses"), max_uses=stats.get("uses"),
                    damage=stats.get("damage", 0), guard=stats.get("guard", 0))
        low = name.lower()
        for key, rule in ITEM_RULES:
            if key in low:
                item.mode = rule.get("mode", "")
                item.helps = tuple(rule.get("helps", ()))
                item.tags = tuple(rule.get("tags", ()))
                break
        items.append(item)
    return items


@dataclass
class Wound:
    name: str
    roll: int
    stat: str = ""         # the stat it shrinks, if any
    bleeding: bool = False
    stabilized: bool = False

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "Wound":
        return cls(**d)


@dataclass
class Bearer:
    name: str
    class_id: str
    class_name: str
    might: int
    grace: int
    wits: int
    nerve: int
    min_stress: int
    stress: int = -1
    grit: int = GRIT
    grit_max: int = GRIT
    wounds: list[Wound] = field(default_factory=list)
    brands: list[str] = field(default_factory=list)
    phobias: list[str] = field(default_factory=list)    # chapter kinds that cost 1 Stress to face
    items: list[Item] = field(default_factory=list)
    talents: list[str] = field(default_factory=list)    # Talents bought with Insight: chapter kinds
    signature: str = "might"
    bond: str = ""
    bond_question: str = ""
    bond_answer: str = ""
    origin: str = ""
    debt: str = ""                  # an origins.json "What It Took" id
    locked_stat: str = ""           # a Debt of memory: this stat cannot be raised
    insight: int = 0
    trials: int = 0
    dead: bool = False

    def __post_init__(self) -> None:
        if self.stress < 0:
            self.stress = self.min_stress

    # -- stats ------------------------------------------------------------
    def base(self, stat: Stat) -> int:
        return 0 if stat is Stat.NONE else getattr(self, stat.value)

    def penalty(self, stat: Stat) -> int:
        """Dice lost to Wounds: a broken limb on one stat, concussion on Wits, gutted on all but Nerve."""
        p = 0
        for w in self.wounds:
            if w.stat == stat.value or (w.stat == "all_but_nerve" and stat is not Stat.NERVE):
                p += 1
        return p

    def stat(self, stat: Stat) -> int:
        return max(0, self.base(stat) - self.penalty(stat))

    def stats(self) -> dict[str, int]:
        return {s.value: self.base(s) for s in STATS}

    def best_stat(self, allowed: tuple[Stat, ...]) -> Stat:
        return max(allowed, key=lambda s: (self.stat(s), -allowed.index(s)))

    def talent_applies(self, kind: Kind, mode: str = "") -> bool:
        if kind.value in self.talents:
            return True
        cover = TALENT_KINDS.get(self.class_id, {})
        if kind not in cover:
            return False
        modes = cover[kind]
        return modes is None or mode in modes

    @property
    def can_push(self) -> bool:
        return not any(w.name == "Concussed" for w in self.wounds)

    # -- pools ------------------------------------------------------------
    def gear_for(self, kind: Kind) -> Item | None:
        for item in self.items:
            if kind.value in item.helps and item.usable:
                return item
        return None

    def weapon_for(self, mode: str) -> Item | None:
        best = None
        for item in self.items:
            if item.mode == mode and item.damage and item.usable and (best is None or item.damage > best.damage):
                best = item
        return best

    def armor(self) -> Item | None:
        for item in self.items:
            if item.guard and item.usable:
                return item
        return None

    def find_item(self, text: str, tag: str = "") -> Item | None:
        low = text.lower()
        for item in self.items:
            words = [w for w in re.split(r"[\s-]+", item.name.lower()) if len(w) > 2]
            if item.usable and (item.name.lower() in low or any(w in low for w in words)):
                return item
        if tag:
            for item in self.items:
                if tag in item.tags and item.usable:
                    return item
        return None

    # -- wounds -----------------------------------------------------------
    @property
    def open_wounds(self) -> int:
        """Wounds that count toward death (a stabilized Wound stops counting until the next Haven)."""
        return sum(1 for w in self.wounds if not w.stabilized)

    @property
    def bleeding(self) -> int:
        return sum(1 for w in self.wounds if w.bleeding and not w.stabilized)

    def heal_wounds(self) -> None:
        """A Haven heals all Wounds except permanent ones (a Maimed stat loss has already been applied)."""
        self.wounds = []

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["wounds"] = [w.to_dict() for w in self.wounds]
        d["items"] = [i.to_dict() for i in self.items]
        d["brands"] = list(self.brands)
        d["phobias"] = list(self.phobias)
        d["talents"] = list(self.talents)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Bearer":
        d = dict(d)
        d["wounds"] = [Wound.from_dict(w) for w in d.get("wounds", [])]
        d["items"] = [Item.from_dict(i) for i in d.get("items", [])]
        return cls(**d)


def pool_size(n: int) -> int:
    return max(0, min(MAX_POOL, n))
