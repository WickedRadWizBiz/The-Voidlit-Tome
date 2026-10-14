"""Game content: the TOME Forge's JSON (the codex) plus place packs (TOML).

The codex is a snapshot of the project's content files (``/tome/content/*.json``, Forge format: one file per
collection, ``{"collection", "schemaVersion", "entries": [...]}``), shipped in ``voidlit/content/tome``.
``tools/sync_content.py`` refreshes it from that folder or from a Forge "Export JSON" bundle. The engine never
edits it: names, Bonds, Openings and tables are changed in the Forge.

A place is somewhere in the Unknown with its own authored Leaves and people (NPCs with BindsNET brains). Places
come from the Forge (``places``, ``people`` and the ``leaves`` that name a ``place``); the Hollow Lantern is the
first. A TOML pack in ``content/places/`` can add a local place the Forge does not have yet.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ..limbic.npc_brain import BranchSeed, NPCSpec, PlotSpec, TacticSpec

CONTENT_DIR = Path(__file__).resolve().parent.parent / "content"
CODEX_DIR = CONTENT_DIR / "tome"
PLACES_DIR = CONTENT_DIR / "places"
REGISTERS = ("plain", "uneasy", "eldritch", "breakthrough")


class Codex:
    """Every Forge collection, by name, as a list of entries sorted by ``order``."""

    def __init__(self, collections: dict[str, list[dict]]) -> None:
        self.c = collections

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Codex":
        p = Path(path) if path else CODEX_DIR
        if p.is_file():
            blob = json.loads(p.read_text())
            if blob.get("format") == "tome-content":
                cols = {k: list(v if isinstance(v, list) else v.get("entries", []))
                        for k, v in blob["collections"].items()}
            else:
                cols = {blob["collection"]: blob["entries"]}
        else:
            cols = {}
            for f in sorted(p.glob("*.json")):
                blob = json.loads(f.read_text())
                cols[blob.get("collection", f.stem)] = blob["entries"]
        for k in cols:
            cols[k] = sorted(cols[k], key=lambda e: (e.get("order", 1e9), e.get("id", "")))
        return cls(cols)

    def entries(self, collection: str) -> list[dict]:
        return self.c.get(collection, [])

    def by_id(self, collection: str, id_: str) -> dict | None:
        return next((e for e in self.entries(collection) if e.get("id") == id_), None)

    def by_roll(self, collection: str, roll: int) -> dict | None:
        return next((e for e in self.entries(collection) if str(e.get("roll")) == str(roll)), None)

    @property
    def classes(self) -> dict[str, dict]:
        return {e["id"]: e for e in self.entries("classes")}

    def opening(self, class_id: str, stat: str) -> dict | None:
        return next((e for e in self.entries("openings") if e.get("class") == class_id and e.get("stat") == stat),
                    None)

    def leaves(self, kind: str) -> list[dict]:
        """The generic Leaves of a kind; a place's own Leaves are dealt only in that place."""
        return [e for e in self.entries("leaves") if e.get("kind", "").lower() == kind and not e.get("place")]

    def places(self) -> dict[str, "Place"]:
        """Every playable place in the Forge (one with an arrival), with its Leaves and people."""
        out = {}
        for e in self.entries("places"):
            if not e.get("arrival"):
                continue
            pid = e["id"]
            narration = {"beat.calm": e.get("narration_calm", []), "beat.foreshadow": e.get("narration_foreshadow", []),
                         "ending.verdict": e.get("narration_verdict", [])}
            place = Place(
                id=pid, name=e["name"], threshold=e.get("threshold", ""), arrival=e["arrival"],
                crucible=e.get("crucible", ""), crucible_text=e.get("crucible_text", ""),
                lore=list(e.get("lore", [])), rumors=list(e.get("rumors", [])),
                leaves=[PlaceLeaf(id=x["id"], kind=x["kind"].lower(), title=x["title"], rank=_rank(x.get("depth", "")),
                                  text={r: x[r] for r in REGISTERS if x.get(r)}, npcs=list(x.get("people", [])),
                                  clue=x.get("clue", ""), cleared=x.get("cleared", ""))
                        for x in self.entries("leaves") if x.get("place") == pid],
                npcs={n["id"]: _npc(n) for n in self.entries("people") if n.get("place") == pid},
                narration={k: list(v) for k, v in narration.items() if v},
            )
            place.validate()
            out[pid] = place
        return out

    def register_for(self, stress: int, panicked: bool = False) -> str:
        if panicked:
            return "breakthrough"
        for e in self.entries("registers"):
            if int(e.get("stress_min", 0)) <= stress <= int(e.get("stress_max", 10)):
                return e["id"]
        return "plain"

    def register_style(self, register: str) -> str:
        e = self.by_id("registers", register)
        return e.get("style", "") if e else ""

    def thresholds(self, fragments: int = 0) -> list[dict]:
        out = []
        for e in self.entries("thresholds"):
            pool = str(e.get("pool", "start"))
            if pool == "start" or (pool.startswith("after") and fragments >= int(pool.split()[1])):
                out.append(e)
        return out

    def validate(self) -> None:
        for need in ("classes", "openings", "leaves", "panic", "echoes", "interventions", "crucible_forms",
                     "agendas", "thresholds", "registers"):
            if not self.entries(need):
                raise ValueError(f"content has no {need!r} entries")
        for c in self.entries("classes"):
            for k in ("might", "grace", "wits", "nerve"):
                if not 0 <= int(c[k]) <= 4:
                    raise ValueError(f"class {c['id']}: {k} out of range")


def _rank(depth: str) -> str:
    """Forge depth ("Shallow (2)", "any") to a rank ("shallow", "")."""
    word = depth.split()[0].lower() if depth else ""
    return word if word in ("shallow", "deep", "abyssal") else ""


def place_key(name: str) -> str:
    """Place ids are compared without caring about dashes or underscores (hollow_lantern = hollow-lantern)."""
    return name.strip().lower().replace("_", "-")


@dataclass
class PlaceLeaf:
    id: str
    kind: str                       # hostile | hazard | relic | witness
    title: str
    text: dict[str, str]            # register -> prose
    rank: str = ""                  # "" = any
    npcs: list[str] = field(default_factory=list)
    clue: str = ""
    cleared: str = ""               # what the surface says when the ring fills


@dataclass
class Place:
    id: str
    name: str
    threshold: str                  # what the TOME says on the cover
    arrival: str                    # the first thing the Bearer sees
    crucible: str
    crucible_text: str = ""
    lore: list[str] = field(default_factory=list)       # truths the place gives up to Clue beats and Speak
    rumors: list[str] = field(default_factory=list)
    leaves: list[PlaceLeaf] = field(default_factory=list)
    npcs: dict[str, NPCSpec] = field(default_factory=dict)
    narration: dict[str, list[str]] = field(default_factory=dict)

    def leaves_of(self, kind: str, rank: str) -> list[PlaceLeaf]:
        return [x for x in self.leaves if x.kind == kind and x.rank in ("", rank)]

    def validate(self) -> None:
        for leaf in self.leaves:
            if leaf.kind not in ("hostile", "hazard", "relic", "witness"):
                raise ValueError(f"place {self.id}: leaf {leaf.id} has unknown kind {leaf.kind!r}")
            for n in leaf.npcs:
                if n not in self.npcs:
                    raise ValueError(f"place {self.id}: leaf {leaf.id}: unknown npc {n!r}")
        ids = [n.numeric_id for n in self.npcs.values()]
        if len(set(ids)) != len(ids) or 0 in ids:
            raise ValueError("npc numeric_id values must be unique and non-zero")


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


def load_place(name_or_path: str | Path) -> Place:
    path = Path(name_or_path)
    if not path.suffix:
        path = PLACES_DIR / f"{name_or_path}.toml"
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    meta = raw["place"]
    place = Place(
        id=meta["id"], name=meta["name"], threshold=meta["threshold"], arrival=meta["arrival"],
        crucible=meta["crucible"], crucible_text=meta.get("crucible_text", ""), lore=list(meta.get("lore", [])),
        rumors=list(meta.get("rumors", [])),
        leaves=[PlaceLeaf(id=x["id"], kind=x["kind"], title=x["title"], rank=x.get("rank", ""),
                          text={r: x[r] for r in REGISTERS if x.get(r)}, npcs=list(x.get("npcs", [])),
                          clue=x.get("clue", ""), cleared=x.get("cleared", ""))
                for x in raw.get("leaves", [])],
        npcs={n["id"]: _npc(n) for n in raw.get("npcs", [])},
        narration={k: list(v) for k, v in raw.get("narration", {}).items()},
    )
    place.validate()
    return place


def place_ids() -> list[str]:
    """Local TOML place packs (the Forge's places come from ``Codex.places``)."""
    return sorted(p.stem for p in PLACES_DIR.glob("*.toml"))


def all_places(codex: Codex) -> dict[str, Place]:
    """The Forge's places, plus any local TOML pack the Forge does not have."""
    places = {place_key(k): v for k, v in codex.places().items()}
    for pid in place_ids():
        if place_key(pid) not in places:
            places[place_key(pid)] = load_place(pid)
    return places
