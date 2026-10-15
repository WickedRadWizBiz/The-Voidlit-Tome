"""A campaign: the story of the TOME, not of any one Bearer (Core Rules 0.2, "Progression and the TOME's Hunger").

Creation (class, Bond, Origin, the Opening roll), Insight, Pages, Hunger, the Haven, Legacy, the Atlas and the
TOME's secret Agenda. Content (classes, origins, agendas) arrives as plain dicts from the TOME Forge's JSON.

The Opening roll is Steven's 2026-10-14 decision (one d6: 1-4 pick a stat, 5-6 the class's signature stat;
that stat +1 unless it is already 3, another stat -1, min 1; the class Opening for that stat is the first
Threshold). It simulated neutral and is final in Core Rules (2026-10-15); by default the app lowers the
weakest other stat above 1, the same policy as the simulator.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .bearer import COMMON_KIT, KINDS, MAX_STAT, STATS, Bearer, Stat, parse_kit
from .dice import d6
from .pages import OPEN_COST
from .trial import PAGE_IDS

FRAGMENTS_TO_NAME = 3
FRAGMENTS_TO_END = 6
OPENING_STATS = {1: Stat.MIGHT, 2: Stat.GRACE, 3: Stat.WITS, 4: Stat.NERVE}
# Words that recognise a guess at each Agenda (agendas.json, by roll).
AGENDA_WORDS = {
    "agenda-1": ("home", "door", "return", "back where"),
    "agenda-2": ("maker", "made it", "creator", "builder"),
    "agenda-3": ("body", "brand", "flesh", "become"),
    "agenda-4": ("out", "prison", "escape", "warden", "free"),
    "agenda-5": ("map", "atlas", "chart", "coming"),
    "agenda-6": ("heir", "successor", "replace", "next tome", "test"),
}
GLYPHS = "⟊⟒⟟⟡⟣⟤⟥⟦⟧⧖⧗⧫⧬⧭"


@dataclass(frozen=True)
class Opening:
    roll: int
    stat: Stat
    raised: bool
    lowered: Stat | None

    def to_dict(self) -> dict:
        return {"roll": self.roll, "stat": self.stat.value, "raised": self.raised,
                "lowered": self.lowered.value if self.lowered else None}


def opening_roll(b: Bearer, rng: random.Random, lower: Stat | None = None) -> Opening:
    roll = d6(rng)
    stat = OPENING_STATS.get(roll) or Stat(b.signature)
    if b.base(stat) >= 3:
        return Opening(roll, stat, False, None)
    setattr(b, stat.value, b.base(stat) + 1)
    options = [s for s in STATS if s is not stat and b.base(s) > 1]
    if lower not in options:
        # The player's choice; by default the least harmful: the weakest stat that can still drop.
        lower = min(options, key=lambda s: (b.base(s), STATS.index(s))) if options else None
    if lower is not None:
        setattr(b, lower.value, b.base(lower) - 1)
    return Opening(roll, stat, True, lower)


def make_bearer(cls: dict, name: str, rng: random.Random, origins: list[dict] | None = None) -> Bearer:
    """Copy a class's stats, Grit, minimum Stress, Talent and Kit, then roll both Origin tables."""
    b = Bearer(name=name, class_id=cls["id"], class_name=cls["name"], might=int(cls["might"]),
               grace=int(cls["grace"]), wits=int(cls["wits"]), nerve=int(cls["nerve"]),
               min_stress=int(cls.get("min_stress", 2)), signature=cls.get("signature", "might"),
               bond=cls.get("bond", ""), bond_question=cls.get("bond_question", ""),
               items=parse_kit(f"{cls.get('kit', '')}, {COMMON_KIT}"))
    if origins:
        night = [o for o in origins if o.get("table") == "The Night It Woke"]
        took = [o for o in origins if o.get("table") == "What It Took"]
        if night:
            b.origin = rng.choice(night)["id"]
        if took:
            b.debt = took[(d6(rng) - 1) % len(took)]["id"]
            if b.debt == "took-1":
                b.locked_stat = rng.choice(STATS).value
    return b


@dataclass
class Campaign:
    bearer: Bearer
    agenda: str                        # agendas.json id, hidden from the player
    hunger: int = 0
    pages_open: list[str] = field(default_factory=list)
    fragments: int = 0
    named: bool = False
    atlas: list[str] = field(default_factory=list)       # places survived
    rumors: list[str] = field(default_factory=list)      # places heard of, never seen
    relics: int = 0                                      # relics carried out of Relic chapters (food for the TOME)
    fallen: list[str] = field(default_factory=list)      # Bearers the TOME has outlived
    last_crucible: str = ""
    trials: int = 0

    @classmethod
    def begin(cls, bearer: Bearer, agendas: list[dict], rng: random.Random) -> "Campaign":
        agenda = rng.choice(agendas)["id"] if agendas else "agenda-1"
        return cls(bearer=bearer, agenda=agenda)

    # -- the trial's results ------------------------------------------------
    def absorb(self, trial) -> None:
        """Carry a finished (or lost) trial's gains into the campaign."""
        self.hunger = max(0, self.hunger + trial.hunger_gain)
        self.relics += trial.relics_found
        for p in trial.pages_opened:
            if p not in self.pages_open:
                self.pages_open.append(p)
        self.fragments = min(FRAGMENTS_TO_END, self.fragments + trial.fragments_gain)
        self.trials += 1
        if trial.survived:
            self.bearer.insight += trial.insight
            self.bearer.trials += 1

    @property
    def last_threshold(self) -> bool:
        return self.fragments >= FRAGMENTS_TO_END

    # -- the Haven ------------------------------------------------------------
    def haven(self) -> list[str]:
        """Stress to the minimum, Grit and temporary Wounds healed, two items restocked to full Uses."""
        b = self.bearer
        notes = []
        b.stress = b.min_stress + (1 if b.debt == "took-4" else 0)
        b.grit = b.grit_max
        b.heal_wounds()
        spent = sorted((i for i in b.items if i.max_uses and i.uses is not None and i.uses < i.max_uses),
                       key=lambda i: i.uses - i.max_uses)
        for item in spent[:2]:
            item.uses = item.max_uses
            notes.append(item.name)
        return notes

    def raise_cost(self, stat: Stat) -> int:
        return 3 * (self.bearer.base(stat) + 1)

    def raise_stat(self, stat: Stat) -> tuple[bool, str]:
        b = self.bearer
        if stat not in STATS:
            return False, "Name a stat."
        if b.base(stat) >= MAX_STAT:
            return False, f"{stat.value.title()} is already at its limit."
        if b.locked_stat == stat.value:
            return False, f"The TOME took a memory; {stat.value} cannot rise until it is recovered."
        cost = self.raise_cost(stat)
        if b.insight < cost:
            return False, f"Raising {stat.value} costs {cost} Insight."
        b.insight -= cost
        setattr(b, stat.value, b.base(stat) + 1)
        return True, f"{stat.value.title()} rises to {b.base(stat)}."

    def buy_talent(self, kind: str) -> tuple[bool, str]:
        b = self.bearer
        if kind not in {k.value for k in KINDS} or kind in b.talents:
            return False, "Name a kind of chapter you do not already have a Talent for."
        if b.insight < 5:
            return False, "A Talent costs 5 Insight."
        b.insight -= 5
        b.talents.append(kind)
        return True, f"A new Talent: +1 die against {kind} chapters."

    def lower_min_stress(self) -> tuple[bool, str]:
        b = self.bearer
        if b.min_stress <= 0 or b.insight < 4:
            return False, "Lowering minimum Stress costs 4 Insight."
        b.insight -= 4
        b.min_stress -= 1
        return True, f"Minimum Stress falls to {b.min_stress}."

    def open_page(self, page: str) -> tuple[bool, str]:
        b = self.bearer
        if page not in PAGE_IDS or page in self.pages_open:
            return False, "That Page is not one the TOME can open for you."
        if b.insight < OPEN_COST:
            return False, f"Opening a Page costs {OPEN_COST} Insight."
        b.insight -= OPEN_COST
        self.pages_open.append(page)
        self.hunger += 1
        return True, f"The {page.title()} Page opens. The TOME grows hungrier."

    def feed_relic(self) -> tuple[bool, str]:
        if self.relics <= 0 or self.hunger <= 0:
            return False, "There is nothing to feed it, or it is not hungry."
        self.relics -= 1
        self.hunger -= 1
        return True, "The TOME takes the relic. Its Hunger eases."

    # -- the Agenda -----------------------------------------------------------
    def name_agenda(self, guess: str) -> tuple[bool, str]:
        if self.fragments < FRAGMENTS_TO_NAME:
            return False, f"The TOME can be Named once you hold {FRAGMENTS_TO_NAME} Fragments."
        if self.named:
            return False, "The Agenda has already been Named."
        low = guess.lower()
        right = any(w in low for w in AGENDA_WORDS.get(self.agenda, ()))
        b = self.bearer
        if right:
            self.named = True
            b.insight += self.fragments
            self.hunger = max(0, self.hunger - 1)
            return True, "The glyphs settle into words you can read. You have Named what it wants."
        b.stress = min(10, b.stress + 2)
        self.hunger += 1
        return False, "The TOME laughs."

    def fragment_text(self, wants: str, seed: int) -> str:
        """The Agenda in violet glyphs; each Fragment translates a few more of them."""
        if self.named or self.last_threshold:
            return wants
        rng = random.Random(seed)
        order = list(range(len(wants)))
        rng.shuffle(order)
        shown = set(order[: int(len(wants) * self.fragments / FRAGMENTS_TO_END)])
        return "".join(c if (i in shown or not c.isalpha()) else GLYPHS[i % len(GLYPHS)]
                       for i, c in enumerate(wants))

    # -- Legacy ---------------------------------------------------------------
    def legacy(self, heir: Bearer) -> None:
        """The TOME outlives its Bearers: the heir inherits its Pages, Hunger and Agenda, nothing else."""
        self.fallen.append(self.bearer.name)
        self.bearer = heir

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["bearer"] = self.bearer.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Campaign":
        d = dict(d)
        d["bearer"] = Bearer.from_dict(d["bearer"])
        return cls(**d)
