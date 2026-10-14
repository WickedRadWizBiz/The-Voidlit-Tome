"""A Trial: the TOME's hidden deck, the Lumen and the Crucible (TOME: Core Rules 0.2, "The TOME engine").

This module is pure rules. It never reads player text: the engine turns words and touches into calls
(`draw`, `cast`, `decide_push`, ...), and every call appends what happened to ``log``.

Decisions only the player can make are *pending*: a Push, a Breather's gift, a Last Rites Bargain, an Echo's
offer. While one is pending the Trial waits; ``pending["type"]`` names it.

The numbers follow the playtest simulator's final configuration (``/tome/playtest/final_cfg.json``): Grit 5,
one Wound roll per hit, Abyssal chapters and the Crucible at Difficulty 2, Crucible Depth 8 (Damage 2), Stress over 10
forcing a Panic, Relics also taking Might and Witnesses also taking Wits, and the Lumen's death raising the
Crucible at once with every Bite 1 worse.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .bearer import (BRANDS_TO_BELONG, KINDS, MAX_STRESS, PANIC_AT, STATS, WOUNDS_FOR_RITES, Bearer, Item, Kind,
                     Stat, Wound, pool_size)
from .dice import PoolRoll, Result, better, d6, push, roll_2d6, roll_pool
from .events import Log

LUMEN_MAX = 6
LUMEN_MIN_START = 2
SHORT, LONG = 3, 5

RANKS = ("shallow", "deep", "abyssal")
DEPTH = {"shallow": 2, "deep": 4, "abyssal": 6}
DIFFICULTY = {"shallow": 1, "deep": 1, "abyssal": 2}
DAMAGE = {"shallow": 1, "deep": 2, "abyssal": 3}
# Retuned 2026-10-14 when Panic 7 in the Crucible became a lost action (rules thread, rules_changes.md).
CRUCIBLE_DEPTH, CRUCIBLE_DIFFICULTY, CRUCIBLE_DAMAGE = 8, 2, 2
PER_KIND = {"shallow": 4, "deep": 4, "abyssal": 1}   # x4 kinds = 16 / 16 / 4
ECHOES, BREATHERS, INTERVENTIONS = 12, 4, 2

ACCEPTS: dict[Kind, tuple[Stat, ...]] = {
    Kind.HOSTILE: (Stat.MIGHT, Stat.GRACE, Stat.NERVE),
    Kind.HAZARD: (Stat.GRACE, Stat.MIGHT),
    Kind.RELIC: (Stat.WITS, Stat.MIGHT),
    Kind.WITNESS: (Stat.NERVE, Stat.WITS),
}
# The forms the shapes take when the TOME offers a choice; they hint at the kind beneath.
SHAPES = {"hostile": "spines", "hazard": "fracture", "relic": "facets", "witness": "ring",
          "echo": "smooth", "breather": "smooth", "intervention": "smooth"}

BRANDS = ("glowing veins", "a second voice", "no heartbeat", "black eyes", "skin like the slab",
          "no reflection")
WOUND_TABLE = (  # (highest total, name)
    (4, "Bruised"), (6, "Bleeding"), (7, "Broken limb"), (8, "Concussed"), (9, "Maimed"), (10, "Gutted"),
)


@dataclass
class Leaf:
    kind: str          # hostile | hazard | relic | witness | echo | breather | intervention
    rank: str = ""     # shallow | deep | abyssal for chapters

    @property
    def chapter(self) -> bool:
        return self.kind in {k.value for k in KINDS}

    @property
    def shape(self) -> str:
        return SHAPES[self.kind]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "rank": self.rank}

    @classmethod
    def from_dict(cls, d: dict) -> "Leaf":
        return cls(**d)


def build_deck(hunger: int, rng: random.Random) -> list[Leaf]:
    deck = [Leaf(k.value, rank) for k in KINDS for rank, n in PER_KIND.items() for _ in range(n)]
    deck += [Leaf("echo")] * ECHOES + [Leaf("breather")] * BREATHERS
    deck += [Leaf("intervention")] * (INTERVENTIONS + (1 if hunger >= 4 else 0))
    deck = [Leaf(x.kind, x.rank) for x in deck]
    rng.shuffle(deck)
    return deck


@dataclass
class Chapter:
    kind: str
    rank: str
    depth: int
    difficulty: int
    damage: int
    progress: int = 0
    casts: int = 0
    crucible: bool = False
    demand: str = ""         # the Crucible's demanded kind for the next cast
    rival: bool = False      # a rival Bearer (Intervention 4)

    @property
    def active_kind(self) -> Kind:
        return Kind(self.demand if self.crucible else self.kind)

    @property
    def accepts(self) -> tuple[Stat, ...]:
        return ACCEPTS[self.active_kind]

    @property
    def remaining(self) -> int:
        return max(0, self.depth - self.progress)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "Chapter":
        return cls(**d)


@dataclass
class Trial:
    """One trial. ``campaign`` supplies Hunger, open Pages, Fragments and Debts; see campaign.py."""

    bearer: Bearer
    hunger: int = 0
    length: str = "short"
    need: int = SHORT
    lumen: int = LUMEN_MAX
    lumen_out: bool = False
    deck: list[Leaf] = field(default_factory=list)
    lost: int = 0                      # Leaves lost to the deck (unchosen shapes, Interventions)
    peek: list[str] = field(default_factory=list)   # shapes the TOME has shown of the next Leaves
    phase: str = "draw"                # draw | chapter | crucible | verdict | dead
    chapter: Chapter | None = None
    cleared: int = 0
    chapters_seen: int = 0
    next_depth_bonus: int = 0
    lost_actions: int = 0
    shaky: int = 0
    dropped: bool = False
    tunnel: bool = False
    possessed: bool = False
    become: bool = False
    ward: bool = False
    rival_next: bool = False
    crucible_now: bool = False
    dying: bool = False
    used: list[str] = field(default_factory=list)   # once-per-trial abilities and Pages already spent
    pending: dict | None = None
    insight: int = 0
    hunger_gain: int = 0
    fragments_gain: int = 0
    relics_found: int = 0
    pages_opened: list[str] = field(default_factory=list)
    pages_open: list[str] = field(default_factory=list)
    log: Log = field(default_factory=Log)

    # ================================================================ setup
    @classmethod
    def begin(cls, bearer: Bearer, rng: random.Random, hunger: int = 0, length: str = "short",
              pages_open: list[str] | None = None) -> "Trial":
        t = cls(bearer=bearer, hunger=hunger, length=length, need=LONG if length == "long" else SHORT,
                pages_open=list(pages_open or []))
        t.deck = build_deck(hunger, rng)
        t.lumen = max(LUMEN_MIN_START, LUMEN_MAX - hunger)
        t.log.add("kindle", f"LUMEN {t.lumen}", "lumen",
                  f"The Lumen kindles: {t.lumen} amber sigils light on the spine.", dice=t.lumen)
        return t

    @property
    def over(self) -> bool:
        return self.phase in ("verdict", "dead")

    @property
    def survived(self) -> bool:
        return self.phase == "verdict"

    # ================================================================ the Draw
    def offer(self, mode: str, rng: random.Random) -> list[str]:
        """The TOME asks to be touched. Returns the shapes offered (empty for the dot and the dark)."""
        if self.phase != "draw" or self.pending:
            raise RuntimeError("the TOME is not offering a Leaf")
        if not self.deck:
            self._raise_crucible("The deck is spent.")
            return []
        shapes = []
        if mode == "shapes":
            n = min(len(self.deck), rng.choice((2, 3)))
            shapes = [leaf.shape for leaf in reversed(self.deck[-n:])]
            if n < 2:
                mode = "dot"
                shapes = []
        self.pending = {"type": "draw", "mode": mode, "shapes": shapes}
        return shapes

    def draw(self, rng: random.Random, choice: int = 0) -> Leaf | None:
        """Touch the TOME (``choice`` picks a shape). Resolves non-chapter Leaves; a chapter Leaf begins."""
        p = self.pending
        if not p or p["type"] != "draw":
            raise RuntimeError("nothing to draw")
        self.pending = None
        if p["mode"] == "shapes" and p["shapes"]:
            n = len(p["shapes"])
            choice = max(0, min(n - 1, choice))
            offered = [self.deck.pop() for _ in range(n)]
            leaf = offered[choice]
            self.lost += n - 1
            self.log.add("lost", f"{n - 1} SHAPES SINK", "absence",
                         f"The other {'shape sinks' if n == 2 else 'shapes sink'} into the stone, lost.")
        else:
            leaf = self.deck.pop()
        if self.peek:
            self.peek.pop(0)
        if leaf.kind == "intervention":
            self._intervention(rng)
        elif leaf.kind == "echo":
            self._echo(rng)
        elif leaf.kind == "breather":
            self.pending = {"type": "breather"}
            self.log.add("breather", "BREATHER", "lumen",
                         "A safe pocket opens. The TOME offers one mercy: ease Stress, restore Grit, "
                         "or restore a Use.")
        else:
            self._begin_chapter(leaf)
        if self.phase == "draw" and not self.pending and (self.crucible_now or self.cleared >= self.need):
            self._raise_crucible()
        return leaf

    # ================================================================ chapters
    def _begin_chapter(self, leaf: Leaf) -> None:
        b = self.bearer
        depth = DEPTH[leaf.rank] + self.next_depth_bonus
        self.next_depth_bonus = 0
        if b.class_id == "archivist":
            depth = max(1, depth - 1)      # Study
        rival = leaf.kind == Kind.HOSTILE.value and self.rival_next
        if rival:
            self.rival_next = False
        self.chapter = Chapter(kind=leaf.kind, rank=leaf.rank, depth=depth, difficulty=DIFFICULTY[leaf.rank],
                               damage=DAMAGE[leaf.rank], rival=rival)
        self.phase = "chapter"
        self.chapters_seen += 1
        self.dropped = self.tunnel = False
        self.lost_actions = 0      # a panic's lost moments belong to the scene they happened in
        self.log.add("leaf", f"{leaf.kind.upper()} · DEPTH {depth}", "stone", "",
                     kind=leaf.kind, rank=leaf.rank, depth=depth, difficulty=self.chapter.difficulty)
        if leaf.kind in b.phobias:
            self.log.add("phobia", "PHOBIA", "stress", "This is the thing you fear. Facing it costs you.")
            self.add_stress(1, rng=None)

    def _raise_crucible(self, why: str = "") -> None:
        b = self.bearer
        depth = CRUCIBLE_DEPTH - (1 if b.class_id == "archivist" else 0)
        self.chapter = Chapter(kind="crucible", rank="crucible", depth=depth, difficulty=CRUCIBLE_DIFFICULTY,
                               damage=CRUCIBLE_DAMAGE, crucible=True, demand=KINDS[0].value)
        self.phase = "crucible"
        self.crucible_now = False
        self.dropped = self.tunnel = False
        self.lost_actions = 0
        self.log.add("crucible", "THE CRUCIBLE RISES", "tome",
                     f"{why + ' ' if why else ''}The Crucible rises: the trial's last chapter.",
                     depth=depth, difficulty=CRUCIBLE_DIFFICULTY)

    def demand(self, rng: random.Random) -> Kind:
        """Each cast, the Crucible demands a different kind. It opens on Hostile and turns after every cast."""
        ch = self.chapter
        if ch and ch.crucible:
            ch.demand = rng.choice([k.value for k in KINDS if k.value != ch.demand] if ch.casts else
                                   [k.value for k in KINDS])
        return ch.active_kind

    def pool(self, stat: Stat, mode: str = "", ally: bool = False, use_gear: bool = True,
             extra: int = 0) -> tuple[int, list[str], Item | None]:
        """Dice for a cast, the reasons for each, and the gear that would spend a Use."""
        b, ch = self.bearer, self.chapter
        kind = ch.active_kind
        n = b.stat(stat)
        notes = [f"{b.stat(stat)} {stat.value}"]
        if b.talent_applies(kind, mode):
            n += 1
            notes.append("+1 talent")
        gear = b.gear_for(kind) if use_gear else None
        if gear is not None:
            n += 1
            notes.append(f"+1 {gear.name}")
        if ally and not self.tunnel:
            n += 1
            notes.append("+1 ally")
        if self.become:
            n += 2
            notes.append("+2 Become")
        if extra:
            n += extra
            notes.append(f"+{extra}")
        if self.shaky:
            n -= self.shaky
            notes.append(f"-{self.shaky} shaking")
        return pool_size(n), notes, gear

    def cast(self, stat: Stat, rng: random.Random, mode: str = "", ally: bool = False, extra: int = 0,
             allow_push: bool = True) -> PoolRoll | None:
        """Cast the pool against the chapter in play. Returns the roll, or None if the action was lost."""
        ch = self.chapter
        if self.pending or ch is None or self.phase not in ("chapter", "crucible"):
            raise RuntimeError("no chapter to cast against")
        kind = ch.active_kind
        if stat not in ch.accepts:
            stat = self.bearer.best_stat(ch.accepts)
        if self.possessed:
            stat = self.bearer.best_stat(ch.accepts)
        if kind is Kind.HOSTILE and mode not in ("melee", "ranged", "hide", "talk"):
            mode = {Stat.MIGHT: "melee", Stat.GRACE: "ranged", Stat.NERVE: "talk"}[stat]
        ch.casts += 1
        if self.lost_actions > 0:
            self.lost_actions -= 1
            self.log.add("lost_action", "LOST", "absence", "Panic holds you. The moment passes without you, "
                         "and the chapter strikes.")
            self._bite(kind, rng)
            self._settle(rng)
            return None
        n, notes, gear = self.pool(stat, mode, ally, extra=extra)
        if gear is not None:
            gear.spend()
        self.shaky = 0
        roll = roll_pool(n, rng, ch.difficulty)
        self.log.add("cast", f"{stat.value.upper()} {roll.hits}/{ch.difficulty}", "hit" if roll.hits else "absence",
                     "", roll=roll.to_dict(), stat=stat.value, mode=mode, notes=notes)
        if roll.missed and allow_push and self.bearer.can_push and not self.possessed:
            self.pending = {"type": "push", "roll": roll.to_dict(), "stat": stat.value, "mode": mode,
                            "kind": kind.value}
            return roll
        self._resolve(roll, stat, mode, kind, rng)
        return roll

    def decide_push(self, rng: random.Random, do_push: bool, borrow: bool = False) -> PoolRoll:
        p = self.pending
        if not p or p["type"] != "push":
            raise RuntimeError("no roll to push")
        self.pending = None
        roll = PoolRoll.from_dict(p["roll"])
        if borrow:
            again = roll_pool(roll.pool, rng, roll.difficulty)
            roll = better(roll, again)
            self.log.add("borrow", f"BORROW {roll.hits}", "page", "The TOME lends a second cast; the better stands.",
                         roll=roll.to_dict())
        elif do_push:
            roll = push(roll, rng)
            self.log.add("push", f"PUSH {roll.hits}/{roll.difficulty}", "hit" if roll.hits else "absence",
                         "You drag the dark dice back up and cast them again.", roll=roll.to_dict())
            if roll.push_ones:
                self.log.add("push_cost", f"+{roll.push_ones} STRESS", "stress",
                             f"{roll.push_ones} of the pushed dice bleed into the Stress stain.")
                self.add_stress(roll.push_ones, rng)
        if not self.dying and self.phase != "dead" and self.chapter is not None:
            self._resolve(roll, Stat(p["stat"]), p["mode"], Kind(p["kind"]), rng)
        return roll

    def _resolve(self, roll: PoolRoll, stat: Stat, mode: str, kind: Kind, rng: random.Random) -> None:
        ch = self.chapter
        b = self.bearer
        if ch is None:
            return
        if roll.result is Result.FAIL:
            self.log.add("miss", "NO HITS", "absence", "Nothing lands. It gets worse.")
            self.add_stress(1, rng)
            self._bite(kind, rng)
        elif ch is self.chapter:
            gain = roll.hits
            if kind is Kind.HOSTILE and mode in ("melee", "ranged") and not self.dropped:
                weapon = b.weapon_for(mode)
                if weapon is not None:
                    gain += weapon.damage
                    weapon.spend()
                    self.log.add("weapon", f"+{weapon.damage} {weapon.name.upper()}", "hit",
                                 f"The {weapon.name} adds its weight.", item=weapon.name)
            ch.progress = min(ch.depth, ch.progress + gain)
            self.log.add("progress", f"DEPTH {ch.progress}/{ch.depth}", "hit", "", progress=ch.progress,
                         depth=ch.depth)
            if roll.result is Result.SHORT:
                self.log.add("short", "SHORT", "stress", "Not enough. It counts, but the chapter bites.")
                self._bite(kind, rng)
        self._settle(rng)

    def _settle(self, rng: random.Random) -> None:
        """After every cast: a cleared ring ends the chapter; the Lumen's death collapses it."""
        ch = self.chapter
        if self.phase == "dead" or self.dying or ch is None:
            return
        if ch.progress >= ch.depth:
            self._end_chapter(rng, cleared=True)
        elif self.crucible_now and not ch.crucible:
            self._end_chapter(rng, cleared=False, collapsed=True)
        elif ch.crucible:
            self.demand(rng)       # the next demand shows before the player chooses

    def _end_chapter(self, rng: random.Random, cleared: bool, collapsed: bool = False, fled: bool = False) -> None:
        ch = self.chapter
        b = self.bearer
        if ch.crucible:
            if cleared:
                self.insight += self.cleared + 2
                self.phase = "verdict"
                self.chapter = None
                self.log.add("verdict", "THE VERDICT", "tome",
                             f"The Crucible sinks into the stone. You survive the trial and earn "
                             f"{self.insight} Insight.", insight=self.insight)
            return
        if cleared:
            self.cleared += 1
            if ch.kind == Kind.RELIC.value:
                self.relics_found += 1
            self.log.add("cleared", f"CLEARED {self.cleared}/{self.need}", "hit",
                         "The ring is full. The Leaf collapses back into the stone.")
        elif collapsed:
            self.log.add("collapsed", "COLLAPSED", "absence", "In the dark, the chapter collapses unfinished.")
        elif fled:
            self.log.add("fled", "FLED", "absence", "You flee the chapter. It does not count.")
        self.chapter = None
        self.phase = "draw"
        if b.bleeding:
            self.log.add("bleeding", "BLEEDING", "harm", "You bleed as you move on.")
            self.harm(b.bleeding, rng, guardable=False, source="bleeding")
        if self.become:
            self.become = False
            self.panic(sum(roll_2d6(rng)), rng)
        self.possessed = False
        if not self.lumen_out and not self.dying and self.phase != "dead":
            self.dim(rng)
        if self.phase == "draw" and not self.pending and not self.dying and (
                self.crucible_now or self.cleared >= self.need):
            self._raise_crucible()

    def use_medkit(self, stabilize: bool | None = None) -> bool:
        """A medkit Use restores 2 Grit or stabilizes one Wound. Medkits do not work on a Vessel."""
        b = self.bearer
        kit = b.find_item("medkit", tag="heal")
        if kit is None or b.class_id == "vessel" or self.pending:
            return False
        open_wounds = [w for w in b.wounds if not w.stabilized]
        if stabilize is None:
            stabilize = bool(open_wounds) and (b.grit > 1 or any(w.bleeding for w in open_wounds))
        if stabilize and open_wounds:
            w = next((w for w in open_wounds if w.bleeding), open_wounds[-1])
            w.stabilized = True
            self.log.add("medkit", f"STABILIZED {w.name.upper()}", "stone", f"The medkit stabilizes the "
                         f"{w.name.lower()}.")
        elif b.grit < b.grit_max:
            b.grit = min(b.grit_max, b.grit + 2)
            self.log.add("medkit", "+2 GRIT", "stone", "The medkit buys some Grit back.")
        else:
            return False
        kit.spend()
        return True

    def abandon_witness(self, rng: random.Random) -> bool:
        """Feed the TOME: abandon a Witness to it (+2 Stress, Hunger -1). The chapter ends."""
        ch = self.chapter
        if ch is None or ch.crucible or ch.kind != Kind.WITNESS.value or self.pending:
            return False
        self.hunger_gain -= 1
        self.log.add("abandoned", "ABANDONED", "tome", "You leave the Witness to the TOME. It feeds, "
                     "and its Hunger eases.")
        self.add_stress(2, rng)
        if self.phase != "dead" and not self.dying and self.chapter is ch:
            ch.progress = ch.depth
            self._settle(rng)
        return True

    def flee(self, rng: random.Random) -> None:
        if self.chapter is None or self.chapter.crucible:
            return
        self._end_chapter(rng, cleared=False, fled=True)

    # ================================================================ Bites, harm, Wounds, Last Rites
    def _bite(self, kind: Kind, rng: random.Random) -> None:
        if self.chapter is None or self.dying or self.phase == "dead":
            return      # the Bearer fled, or is already dying
        dark = self.lumen_out
        if kind is Kind.HOSTILE:
            dmg = self.chapter.damage + (1 if dark else 0)
            self.log.add("bite", f"STRIKE {dmg}", "harm", "It strikes you.", kind=kind.value, amount=dmg)
            self.harm(dmg, rng, source="enemy")
        elif kind is Kind.HAZARD:
            self.log.add("bite", "HAZARD", "harm", "The place itself hurts you.", kind=kind.value)
            self.harm(2 if dark else 1, rng, source="hazard")
        elif kind is Kind.RELIC:
            if dark:
                self.log.add("bite", "RELIC", "harm", "The relic lashes out in the dark.", kind=kind.value)
                self.harm(2, rng, source="relic")
                return
            items = [i for i in self.bearer.items if i.uses]
            # The TOME drains what helps first: gear, then weapons, then anything but the medkit.
            item = next((i for i in items if i.helps), None) or next((i for i in items if i.damage), None) or \
                next((i for i in items if "heal" not in i.tags), None)
            if item is not None:
                item.spend()
                self.log.add("bite", f"-1 USE {item.name.upper()}", "absence",
                             f"The relic drains the {item.name}.", kind=kind.value, item=item.name)
            else:
                self.log.add("bite", "THE LUMEN DIMS", "lumen", "The relic drinks the TOME's light.",
                             kind=kind.value)
                self.dim_by(1)
        elif kind is Kind.WITNESS:
            self.log.add("bite", "WITNESS", "stress", "What you witness stays with you.", kind=kind.value)
            self.add_stress(2 if dark else 1, rng)

    def harm(self, n: int, rng: random.Random, guardable: bool = True, source: str = "") -> None:
        b = self.bearer
        if n <= 0 or b.dead or self.dying:
            return
        if self.ward and source not in ("bleeding",):
            self.ward = False
            self.log.add("ward", "WARD", "page", "The Ward Page turns the harm aside.")
            return
        if guardable:
            armor = b.armor()
            if armor is not None:
                armor.spend()
                n -= armor.guard
                self.log.add("guard", f"GUARD {armor.guard}", "stone", f"The {armor.name} takes some of it.")
                if n <= 0:
                    return
        absorbed = min(b.grit, n)
        b.grit -= absorbed
        n -= absorbed
        if absorbed:
            self.log.add("grit", f"-{absorbed} GRIT", "harm", "", grit=b.grit)
        if n > 0:
            self._wound(n - 1, rng)

    def _wound(self, extra: int, rng: random.Random) -> None:
        b = self.bearer
        dice = roll_2d6(rng)
        total = sum(dice) + b.open_wounds + extra
        name = next((nm for top, nm in WOUND_TABLE if total <= top), "Last Rites")
        self.log.add("wound", name.upper(), "harm", f"A hit gets past your Grit: {name.lower()}.",
                     dice=list(dice), total=total, wound=name)
        if name == "Last Rites":
            self.last_rites(rng)
            return
        if name == "Bruised":
            self.add_stress(1, rng)
            return
        w = Wound(name=name, roll=total)
        if name == "Bleeding":
            w.bleeding = True
        elif name == "Broken limb":
            w.stat = rng.choice(STATS).value     # the TOME's choice
        elif name == "Concussed":
            w.stat = Stat.WITS.value
        elif name == "Maimed":
            s = rng.choice(STATS)
            setattr(b, s.value, max(0, b.base(s) - 1))
            self.log.add("maimed", f"-1 {s.value.upper()} FOREVER", "harm", f"You lose {s.value} for good.")
        elif name == "Gutted":
            w.stat = "all_but_nerve"
        b.wounds.append(w)
        if b.open_wounds >= WOUNDS_FOR_RITES:
            self.last_rites(rng)

    def last_rites(self, rng: random.Random) -> None:
        b = self.bearer
        if self.dying or b.dead:
            return
        if len(b.brands) >= BRANDS_TO_BELONG:
            self.log.add("taken", "IT TAKES YOU", "tome", "With two Brands you already belong to it. "
                         "The Bargain cannot be refused, and the TOME takes you.")
            self._die("taken")
            return
        self.dying = True
        self.pending = {"type": "bargain"}
        self.log.add("last_rites", "LAST RITES", "tome", "You are dying. The TOME offers a Bargain: a stat, "
                     "Hunger and a Brand, for one more breath.")

    def decide_bargain(self, rng: random.Random, accept: bool, stat: Stat | None = None) -> None:
        p = self.pending
        if not p or p["type"] != "bargain":
            raise RuntimeError("no Bargain offered")
        self.pending = None
        self.dying = False
        b = self.bearer
        if not accept:
            self.log.add("refused", "REFUSED", "tome", "You refuse the Bargain, and you die.")
            self._die("refused")
            return
        lowered = stat if stat in STATS else min(STATS, key=lambda s: (b.base(s), STATS.index(s)))
        setattr(b, lowered.value, max(0, b.base(lowered) - 1))
        brand = BRANDS[d6(rng) - 1]
        b.brands.append(brand)
        self.hunger_gain += 1
        b.grit = 1
        for w in b.wounds:
            w.stabilized = True
        self.log.add("bargain", "BARGAIN STRUCK", "tome",
                     f"You accept. You lose {lowered.value} for good, the TOME grows hungrier, and you "
                     f"carry a Brand: {brand}.", stat=lowered.value, brand=brand)
        if self.chapter is not None:
            self._settle(rng)

    def _die(self, how: str) -> None:
        self.bearer.dead = True
        self.dying = False
        self.pending = None
        self.phase = "dead"
        self.log.add("death", "THE BEARER DIES", "tome", "", how=how)

    # ================================================================ Stress and Panic
    def add_stress(self, n: int, rng: random.Random | None) -> None:
        b = self.bearer
        for _ in range(n):
            if b.dead or self.dying:
                return
            if b.stress >= MAX_STRESS:
                if rng is None:
                    return
                self.log.add("overflow", "STRESS 10", "stress", "Stress cannot rise further; it breaks into Panic.")
                self.panic(sum(roll_2d6(rng)), rng)
                continue
            b.stress += 1
            if b.stress >= PANIC_AT and rng is not None:
                total = sum(roll_2d6(rng))
                if total <= b.stress:
                    self.panic(total, rng)
                else:
                    self.log.add("held", f"HOLD {total}>{b.stress}", "stone", "", total=total, stress=b.stress)

    def panic(self, total: int, rng: random.Random) -> None:
        b = self.bearer
        if b.class_id == "vessel" and total not in (9, 11):
            if self.lumen_out or (self.chapter is not None and self.chapter.crucible):
                # In the Crucible, or with the Lumen dead, the panic costs the Vessel's next action instead.
                self.lost_actions += 1
                self.log.add("vessel", "THE TOME RUNS THROUGH YOU", "tome",
                             "The panic runs through your code and locks you up for a moment.")
                return
            self.log.add("vessel", "THE TOME RUNS THROUGH YOU", "tome",
                         "The panic runs through your code and out into the TOME's light.")
            self.dim_by(1)
            return
        if b.class_id == "breacher" and "breacher_ignore" not in self.used:
            self.used.append("breacher_ignore")
            self.log.add("steady", "STEADY", "stone", "Your training holds, once.")
            return
        if b.class_id == "pilgrim" and b.grit >= 2:
            self.log.add("numb", "YOU FEEL NOTHING", "harm", "You take the hurt instead of the fear.")
            self.harm(1, rng, guardable=False, source="faith")
            return
        self.log.add("panic", f"PANIC {total}", "stress", "", total=total)
        ch = self.chapter
        if total == 2:
            self.lost_actions += 1
        elif total == 3:
            self.shaky += 1
        elif total == 4:
            self.tunnel = True
        elif total == 5:
            self.dropped = True
        elif total == 6:
            self.need += 1
        elif total == 7:
            if ch is not None and not ch.crucible:
                self._end_chapter(rng, cleared=False, fled=True)
                self.harm(1, rng, guardable=False, source="flight")
            else:
                self.lost_actions += 1
        elif total == 8:
            self.lost_actions += 1
        elif total == 9:
            b.min_stress = min(MAX_STRESS, b.min_stress + 1)
        elif total == 10:
            self.harm(1, rng, guardable=False, source="freeze")
            self.lost_actions += 2
        elif total == 11:
            kind = ch.active_kind.value if ch is not None else rng.choice([k.value for k in KINDS])
            if kind not in b.phobias:
                b.phobias.append(kind)
        elif total == 12:
            self.possessed = True
            self.dim_by(2)

    # ================================================================ the Lumen
    def dim(self, rng: random.Random) -> int:
        if self.lumen <= 0:
            return 0
        dice = [rng.randint(1, 6) for _ in range(self.lumen)]
        lost = dice.count(1)
        self.log.add("dim", f"LUMEN {self.lumen - lost}" if lost else "THE LUMEN HOLDS", "lumen",
                     "", dice=dice, lost=lost)
        self.dim_by(lost, cast=False)
        return lost

    def dim_by(self, n: int, cast: bool = True) -> None:
        if n <= 0 or self.lumen <= 0:
            return
        self.lumen = max(0, self.lumen - n)
        if cast:
            self.log.add("dark", f"-{n} LUMEN", "lumen", "", lost=n)
        if self.lumen == 0 and not self.lumen_out:
            self.lumen_out = True
            self.crucible_now = True
            self.log.add("lumen_out", "THE LUMEN DIES", "lumen",
                         "The Lumen goes out. In the dark, every Bite is worse, and the Crucible rises.")

    def kindle(self, n: int) -> None:
        self.lumen += n
        self.log.add("kindle", f"+{n} LUMEN", "lumen", "", dice=self.lumen)

    # ================================================================ Echoes, Breathers, Interventions
    def _echo(self, rng: random.Random) -> None:
        roll = d6(rng)
        names = {1: "Lore", 2: "Warning", 3: "Bargain", 4: "Temptation", 5: "Judgment", 6: "Silence"}
        self.log.add("echo", f"ECHO: {names[roll].upper()}", "tome", "", roll=roll, name=names[roll])
        if roll == 1:
            self.fragments_gain += 1
            self.log.add("fragment", "A FRAGMENT", "tome", "A line of violet glyphs rises: a Fragment of the "
                         "TOME's Agenda, in a script you cannot read.")
        elif roll == 2:
            nxt = [leaf.shape for leaf in reversed(self.deck[-2:])]
            if len(nxt) == 2:
                self.pending = {"type": "warning", "shapes": nxt}
                self.log.add("warning", "WARNING", "tome", "The TOME shows the shapes of the next two Leaves and "
                             "lets you choose their order.", shapes=nxt)
        elif roll == 3:
            self.kindle(2)
            self.next_depth_bonus += 2
            self.log.add("bargain_echo", "+2 LUMEN · DEPTH +2", "tome",
                         "The TOME gives two Lumen dice now; the next chapter will run 2 deeper.")
        elif roll == 4:
            closed = [p for p in PAGE_IDS if p not in self.pages_open]
            if closed:
                page = closed[(d6(rng) * 2 + d6(rng)) % len(closed)]
                self.pending = {"type": "temptation", "page": page}
                self.log.add("temptation", "TEMPTATION", "page",
                             "The TOME offers to open one of its Pages for free. It will grow hungrier.",
                             page=page)
        elif roll == 5:
            self.pending = {"type": "judgment", "question": self.bearer.bond_question}
            self.log.add("judgment", "JUDGMENT", "tome", "The TOME asks your Bond Question again.")
        elif roll == 6:
            self.log.add("silence", "", "absence", "The TOME says nothing at all.")
            self.add_stress(1, rng)

    def decide_breather(self, choice: str, item: str = "") -> None:
        if not self.pending or self.pending["type"] != "breather":
            raise RuntimeError("no Breather")
        self.pending = None
        b = self.bearer
        done = False
        if choice == "stress" and b.stress > b.min_stress:
            b.stress -= 1
            done = True
            self.log.add("eased", "-1 STRESS", "stress", "Your breathing slows. 1 Stress fades.")
        elif choice == "use":
            target = b.find_item(item) if item else None
            low = [i for i in b.items if i.max_uses and i.uses is not None and i.uses < i.max_uses]
            target = target if target in low else (low[0] if low else None)
            if target is not None:
                target.uses += 1
                done = True
                self.log.add("restored", f"+1 USE {target.name.upper()}", "stone", f"The {target.name} is restored "
                             "a little.")
        if not done:
            b.grit = b.grit_max
            self.log.add("grit_back", "GRIT RESTORED", "stone", "You get your footing back. Grit restored.")
        self._after_pending()

    def decide_warning(self, swap: bool) -> None:
        if not self.pending or self.pending["type"] != "warning":
            raise RuntimeError("no Warning")
        shapes = self.pending["shapes"]
        self.pending = None
        if swap and len(self.deck) >= 2:
            self.deck[-1], self.deck[-2] = self.deck[-2], self.deck[-1]
            shapes = shapes[::-1]
        self.peek = list(shapes)
        self._after_pending()

    def decide_temptation(self, accept: bool) -> str | None:
        if not self.pending or self.pending["type"] != "temptation":
            raise RuntimeError("no Temptation")
        page = self.pending["page"]
        self.pending = None
        opened = None
        if accept:
            self.pages_open.append(page)
            self.pages_opened.append(page)
            self.hunger_gain += 1
            opened = page
            self.log.add("page_opened", f"PAGE: {page.upper()}", "page",
                         f"A Page opens: {page.title()}. The TOME grows hungrier.", page=page)
        self._after_pending()
        return opened

    def decide_judgment(self, answer: str) -> bool:
        if not self.pending or self.pending["type"] != "judgment":
            raise RuntimeError("no Judgment")
        self.pending = None
        b = self.bearer
        changed = bool(answer.strip()) and _norm(answer) != _norm(b.bond_answer)
        if b.bond_answer and changed:
            b.stress = max(b.min_stress, b.stress - 2)
            self.log.add("judged", "-2 STRESS", "tome", "The answer has changed, and something in you "
                         "loosens.")
        else:
            self.log.add("judged", "UNCHANGED", "tome", "The answer is the same as before. The TOME notes it.")
        if answer.strip():
            b.bond_answer = answer.strip()
        self._after_pending()
        return changed

    def _intervention(self, rng: random.Random) -> None:
        roll = d6(rng)
        b = self.bearer
        self.log.add("intervention", "INTERVENTION", "tome", "", roll=roll)
        if roll == 1:
            if b.debt in ("", "took-6"):
                b.debt = f"took-{rng.randint(1, 5)}"
                self.log.add("debt", "A DEBT IS CLAIMED", "tome", "The TOME claims what it was saving.", debt=b.debt)
            else:
                self.log.add("debt", "THE DEBT DEEPENS", "tome", "The TOME deepens the Debt it holds.", debt=b.debt)
        elif roll == 2:
            self.log.add("darkness", "EVERYTHING GOES DARK", "lumen", "Everything goes dark for a moment.")
            self.dim_by(2)
        elif roll == 3:
            if self.deck:
                self.deck.pop()
                self.lost += 1
                if self.peek:
                    self.peek.pop(0)
            self.log.add("moved", "THE TOME MOVES YOU", "tome", "The TOME moves you. A Leaf is lost unseen.")
        elif roll == 4:
            self.rival_next = True
            self.log.add("rival", "ANOTHER BEARER", "tome", "Another Bearer's trial bleeds into this one.")
        elif roll == 5:
            b.grit = b.grit_max
            b.wounds = []
            self.log.add("healed", "HEALED", "tome", "The TOME heals you fully. It wants you alive for "
                         "something.")
        elif roll == 6:
            self.crucible_now = True
            self.log.add("summoned", "NOW", "tome", "The TOME will not wait any longer.")

    def _after_pending(self) -> None:
        if self.phase == "draw" and not self.pending and (self.crucible_now or self.cleared >= self.need):
            self._raise_crucible()

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("bearer", "log")}
        d["deck"] = [x.to_dict() for x in self.deck]
        d["chapter"] = self.chapter.to_dict() if self.chapter else None
        return d

    @classmethod
    def from_dict(cls, d: dict, bearer: Bearer) -> "Trial":
        d = dict(d)
        d["deck"] = [Leaf.from_dict(x) for x in d["deck"]]
        d["chapter"] = Chapter.from_dict(d["chapter"]) if d.get("chapter") else None
        return cls(bearer=bearer, **d)


PAGE_IDS = ("ward", "unlock", "read", "mend", "unmake", "quiet", "kindle", "command", "step", "borrow", "speak",
            "become")


def _norm(s: str) -> str:
    return " ".join("".join(c for c in s.lower() if c.isalnum() or c.isspace()).split())
