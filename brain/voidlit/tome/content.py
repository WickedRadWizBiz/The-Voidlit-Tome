"""Scenario loading: TOML on disk -> typed rules and brain specs."""

from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ..limbic.npc_brain import BranchSeed, NPCSpec, PlotSpec, TacticSpec
from ..rules.character import Character, Item
from ..rules.clocks import Clock
from ..rules.obstacles import Obstacle
from ..rules.sector import Chapter

CONTENT_DIR = Path(__file__).resolve().parent.parent / "content"


@dataclass
class Background:
    id: str
    name: str
    grit: int
    guile: int
    insight: int
    hull: int
    items: list[Item]
    resolve: int = 2

    def make_character(self, name: str) -> Character:
        return Character(name=name, background=self.name, grit=self.grit, guile=self.guile,
                         insight=self.insight, resolve=self.resolve, hull_max=self.hull,
                         inventory=copy.deepcopy(self.items))


@dataclass(frozen=True)
class Attributes:
    """An NPC's check scores (kept beside its brain spec; the brain itself never rolls)."""

    grit: int = 1
    guile: int = 1
    insight: int = 1
    resolve: int = 1

    def score(self, attr: str) -> int:
        return getattr(self, attr)


@dataclass
class Scenario:
    id: str
    title: str
    intro: str
    per_act: int
    lumen_interval: int
    lore: list[str]
    backgrounds: dict[str, Background]
    obstacles: dict[str, Obstacle]
    hazards: dict[int, list[str]]
    chapters: dict[str, Chapter]
    npcs: dict[str, NPCSpec]
    clocks: list[Clock]
    narration: dict[str, list[str]] = field(default_factory=dict)
    dc_base: int = 12      # obstacle DC = base + per_act * act + offset (+1 in the dark)
    dc_per_act: int = 2
    dc_offset: int = 0     # the one global balance knob (fitted by the balance gate)
    npc_attributes: dict[str, Attributes] = field(default_factory=dict)
    ledger: dict = field(default_factory=dict)   # raw [topics], [[facts]], [[suggestions]], [[probes]] (tome.ledger)

    def new_obstacle(self, obstacle_id: str) -> Obstacle:
        return copy.deepcopy(self.obstacles[obstacle_id])

    def new_clocks(self) -> list[Clock]:
        return copy.deepcopy(self.clocks)

    def validate(self) -> None:
        for ch in self.chapters.values():
            for oid in ch.obstacles:
                if oid not in self.obstacles:
                    raise ValueError(f"chapter {ch.id}: unknown obstacle {oid!r}")
            for nid in ch.npcs:
                if nid not in self.npcs:
                    raise ValueError(f"chapter {ch.id}: unknown npc {nid!r}")
        for ids in self.hazards.values():
            for oid in ids:
                if oid not in self.obstacles:
                    raise ValueError(f"hazards: unknown obstacle {oid!r}")
        for c in self.clocks:
            if c.consequence not in ("", "lumen") and c.consequence not in self.obstacles:
                raise ValueError(f"clock {c.id}: unknown consequence {c.consequence!r}")
        ids = [n.numeric_id for n in self.npcs.values()]
        if len(set(ids)) != len(ids) or 0 in ids:
            raise ValueError("npc numeric_id values must be unique and non-zero")
        from .ledger import FactLedger

        led = FactLedger.from_content(self.ledger)
        for f in led.facts.values():
            if f.scope != "any" and f.scope not in self.chapters:
                raise ValueError(f"fact {f.id}: unknown scope {f.scope!r}")


def _item(d: dict) -> Item:
    return Item(name=d["name"], tags=tuple(d.get("tags", ())), bonus=dict(d.get("bonus", {})),
                uses=d.get("uses"), light=d.get("light", 0))


def _npc(d: dict) -> NPCSpec:
    return NPCSpec(
        id=d["id"], name=d["name"], numeric_id=d["numeric_id"], persona=d.get("persona", ""),
        color=d.get("color", "#9fb4ff"), aliases=list(d.get("aliases", [])),
        plots=[PlotSpec(id=p["id"], name=p["name"], role=p.get("role", "other"), momentum=p.get("momentum", 0.2))
               for p in d["plots"]],
        tactics=[TacticSpec(id=t["id"], directive=t["directive"], plot=t.get("plot", ""), lines=list(t.get("lines", [])))
                 for t in d["tactics"]],
        branches=[BranchSeed(plot=b["plot"], channels=list(b["channels"]), weight=b.get("weight", 0.55))
                  for b in d.get("branches", [])],
        reward=dict(d.get("reward", {"rapport": 0.6, "deception": -0.5, "exposure": -0.4})),
        trust=dict(d.get("trust", {})),
    )


def load_scenario(name_or_path: str | Path = "hollow_lantern") -> Scenario:
    path = Path(name_or_path)
    if not path.suffix:
        path = CONTENT_DIR / f"{name_or_path}.toml"
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    meta = raw["scenario"]
    obstacles = {
        oid: Obstacle(id=oid, name=o["name"], kind=o["kind"], pips=list(o["pips"]), harm=o.get("harm", 1),
                      stress=o.get("stress", 0), text=o.get("text", ""), defeat_text=o.get("defeat_text", ""),
                      blocking=o.get("blocking", True))
        for oid, o in raw.get("obstacles", {}).items()
    }
    chapters = {
        c["id"]: Chapter(id=c["id"], act=c["act"], kind=c["kind"], title=c["title"], text=c["text"],
                         obstacles=list(c.get("obstacles", [])), npcs=list(c.get("npcs", [])),
                         clues=list(c.get("clues", [])), respite=c.get("respite", False),
                         light=c.get("light", 0), exit_text=c.get("exit_text", ""))
        for c in raw["chapters"]
    }
    scenario = Scenario(
        id=meta["id"], title=meta["title"], intro=meta["intro"], per_act=meta.get("per_act", 2),
        lumen_interval=meta.get("lumen_interval", 3), lore=list(meta.get("lore", [])),
        backgrounds={
            b["id"]: Background(id=b["id"], name=b["name"], grit=b["grit"], guile=b["guile"],
                                insight=b["insight"], hull=b["hull"], items=[_item(i) for i in b.get("items", [])],
                                resolve=b.get("resolve", 2))
            for b in raw["backgrounds"]
        },
        obstacles=obstacles,
        hazards={int(k.removeprefix("act")): list(v) for k, v in raw.get("hazards", {}).items()},
        chapters=chapters,
        npcs={n["id"]: _npc(n) for n in raw.get("npcs", [])},
        clocks=[Clock(id=c["id"], name=c["name"], segments=c["segments"], consequence=c.get("consequence", ""))
                for c in raw.get("clocks", [])],
        narration={k: list(v) for k, v in raw.get("narration", {}).items()},
        dc_base=meta.get("dc_base", 12), dc_per_act=meta.get("dc_per_act", 2), dc_offset=meta.get("dc_offset", 0),
        npc_attributes={n["id"]: Attributes(**n.get("attributes", {})) for n in raw.get("npcs", [])},
        ledger={k: raw[k] for k in ("topics", "facts", "suggestions", "probes") if k in raw},
    )
    scenario.validate()
    return scenario
