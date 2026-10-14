"""The TOME's turn loop: one player input in, one frame out.

The rules are TOME: Core Rules (Draft 0.2), implemented in ``voidlit.rules``. Content (classes, Openings,
Leaves, tables) is the TOME Forge's JSON. Around those, the app adds a mind::

    senses     transducer (llama.cpp GBNF pass, or heuristics) -> SensoryState: intent, stat, target, tone
    rules      a touch, a cast, a Push, a Bargain... -> Trial events (every die rolled by the rules)
    limbic     the people in a Leaf perceive the Bearer (BindsNET, R-STDP); the addressed one answers
    cortex     the Director (SpikingJelly) picks how the TOME asks to be touched, and the narrative beat
    voice      narration in the Stress register (llama.cpp, or the offline phrase banks)
    frame      semantic blocks for the voxel TOME (or the terminal)

The language model never decides an outcome. By the time the voice is called, every die has been rolled and
every consequence applied. It is only told what happened, in which register.

Free text stands in for touches: a player types what they do, and words like "push", "accept" or "2" answer
the TOME's offers. The voxel client can send the same words for its taps.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..cortex.director import Director
from ..cortex.features import DRAWS, NARRATIVE, Beat, DirectorFeatures
from ..limbic.contagion import gossip
from ..limbic.npc_brain import BrainDecision, NPCBrain, NPCSpec
from ..rules.bearer import KINDS, STATS, Kind, Stat
from ..rules.campaign import FRAGMENTS_TO_NAME, Campaign, make_bearer, opening_roll
from ..rules.dice import roll_2d6
from ..rules.events import Event
from ..rules.oracle import ask, likelihood_of
from ..rules.pages import USE as PAGE_USE
from ..rules.pages import find_page, use_page
from ..rules.trial import LUMEN_MAX, PAGE_IDS, SHAPES, Trial
from ..senses.heuristic import HeuristicTransducer
from ..senses.state import ENGAGING, Intent, SensoryState
from ..voice.narrator import OfflineVoice
from ..voice.prompts import compile_npc_prompt
from ..voice.requests import DialogueRequest, NarrationRequest
from .content import Codex, Place, load_place, place_ids
from .frames import COLORED, SHAPE_GLYPH, STAT_GLYPH, Block, Span, TomeFrame, dice_block, span, text_block

SAVE_VERSION = 2
YES = ("yes", "accept", "take it", "i accept", "agree", "deal", "open it", "push", "do it", "y")
NO = ("no", "refuse", "decline", "never", "let it stand", "leave it", "don't", "n")


def _says(low: str, words: tuple[str, ...]) -> bool:
    tokens = set(re.findall(r"[a-z']+", low))
    return any((w in tokens) if " " not in w else (w in low) for w in words)


@dataclass
class LeafView:
    """The Leaf in play, as the player sees it: content from the Forge or a place pack."""

    id: str
    kind: str
    title: str
    text: dict[str, str] = field(default_factory=dict)
    npcs: list[str] = field(default_factory=list)
    clue: str = ""
    cleared: str = ""

    def prose(self, register: str) -> str:
        return self.text.get(register) or ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "LeafView":
        return cls(**d)


@dataclass
class GameState:
    seed: int
    campaign: Campaign
    trial: Trial | None = None
    stage: str = "trial"                 # trial | haven | fallen | over
    place: str = ""                      # a place pack id, or "" for the Unknown at large
    threshold: str = ""
    crucible_form: str = ""
    crucible_text: str = ""
    opening: dict = field(default_factory=dict)
    leaf: LeafView | None = None
    used_leaves: list[str] = field(default_factory=list)
    npcs_present: list[str] = field(default_factory=list)
    npcs_met: list[str] = field(default_factory=list)
    clues_revealed: list[str] = field(default_factory=list)
    truths: list[str] = field(default_factory=list)      # canon secrets the TOME has told
    tension: float = 0.2
    last_outcome: str | None = None
    legacy_threshold: str = ""
    final: bool = False                  # this trial is the Last Threshold
    turn: int = 0
    history: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("campaign", "trial", "leaf")}
        d["campaign"] = self.campaign.to_dict()
        d["trial"] = self.trial.to_dict() if self.trial else None
        d["leaf"] = self.leaf.to_dict() if self.leaf else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GameState":
        d = dict(d)
        d["campaign"] = Campaign.from_dict(d["campaign"])
        d["trial"] = Trial.from_dict(d["trial"], d["campaign"].bearer) if d.get("trial") else None
        d["leaf"] = LeafView.from_dict(d["leaf"]) if d.get("leaf") else None
        return cls(**d)


@dataclass
class Turn:
    """Everything that happened in one input, gathered for the voice and the frame."""

    text: str
    sensory: SensoryState | None = None
    events: list[Event] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)     # dice and events, in order
    npc_lines: list[Block] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    decisions: dict[str, BrainDecision] = field(default_factory=dict)
    outcome: str | None = None
    stat: str = "none"
    variant: str | None = None
    npc_name: str | None = None
    entering: bool = False
    cleared_text: str | None = None
    clue: str | None = None
    beat: Beat = Beat.CALM
    beat_fact: str = ""
    beat_note: str = ""
    beat_probs: dict[str, float] = field(default_factory=dict)
    gossip: list[str] = field(default_factory=list)
    ending: str | None = None
    panicked: bool = False
    title: str | None = None
    leaf_kind: str = ""
    flee_npc: str | None = None

    def say(self, fact: str) -> None:
        if fact:
            self.facts.append(fact)


class TomeEngine:
    def __init__(self, codex: Codex | None = None, *, transducer=None, voice=None, director: Director | None = None,
                 seed: int = 0, class_id: str | None = None, player_name: str = "Bearer", place: str | None = None,
                 lower: str | None = None, length: str = "short") -> None:
        self.codex = codex or Codex.load()
        self.codex.validate()
        self.places: dict[str, Place] = {pid: load_place(pid) for pid in place_ids()}
        self.transducer = transducer or HeuristicTransducer()
        banks: dict[str, list[str]] = {}
        for p in self.places.values():
            banks.update(p.narration)
        self.voice = voice or OfflineVoice(banks, seed=seed)
        self.director = director or Director(seed=seed)
        self.player_name = player_name
        self.length = length
        self.rng = random.Random(seed)
        self.seq = 0
        self.npcs: dict[str, NPCSpec] = {}
        for p in self.places.values():
            for nid, spec in p.npcs.items():
                if nid in self.npcs:
                    raise ValueError(f"npc id {nid!r} is used by two places")
                self.npcs[nid] = spec
        self.brains: dict[str, NPCBrain] = {
            nid: NPCBrain(spec, seed=seed * 1000 + spec.numeric_id) for nid, spec in self.npcs.items()
        }
        classes = self.codex.classes
        cid = class_id if class_id in classes else self.rng.choice(sorted(classes))
        bearer = make_bearer(classes[cid], player_name, self.rng, self.codex.entries("origins"))
        lowered = Stat(lower) if lower in {s.value for s in STATS} else None
        opening = opening_roll(bearer, self.rng, lowered)
        campaign = Campaign.begin(bearer, self.codex.entries("agendas"), self.rng)
        self.state = GameState(seed=seed, campaign=campaign, opening=opening.to_dict())
        self._forced_place = place if place in self.places else None

    # ================================================================ public API
    @property
    def bearer(self):
        return self.state.campaign.bearer

    @property
    def trial(self) -> Trial | None:
        return self.state.trial

    def start(self) -> TomeFrame:
        t = Turn(text="")
        b = self.bearer
        cls = self.codex.classes[b.class_id]
        op = self.state.opening
        blocks = [text_block("title", "TOME", "tome", glow=0.9, effect="ripple"),
                  text_block("narration", self._creation_text(cls, op))]
        self._begin_trial(t, first=True)
        blocks += self._event_blocks(t)
        narration = f"{self.state.threshold}"
        if self.state.place:
            narration += f"\n\n{self.places[self.state.place].arrival}"
        self._remember(narration)
        blocks.append(text_block("narration", narration, "tome", glow=0.5, effect="ripple"))
        self._offer(t)
        blocks += self._event_blocks(t)
        return self._frame(blocks + [self._prompt_block()], t)

    def turn(self, text: str) -> TomeFrame:
        st = self.state
        t = Turn(text=text)
        st.turn += 1
        if st.stage == "over":
            return self._frame([text_block("system", "The TOME is closed. Begin again.")], t)
        if st.stage == "haven":
            self._haven_turn(t)
        elif st.stage == "fallen":
            self._legacy_turn(t)
        else:
            self._trial_turn(t)
        return self._compose(t)

    # ================================================================ creation and Thresholds
    def _creation_text(self, cls: dict, op: dict) -> str:
        b = self.bearer
        origin = self.codex.by_id("origins", b.origin)
        debt = self.codex.by_id("origins", b.debt)
        stats = "  ".join(f"{STAT_GLYPH[s.value]} {s.value.title()} {b.base(s)}" for s in STATS)
        shift = (f"The Opening roll ({op['roll']}) raised {op['stat']}"
                 + (f" and lowered {op['lowered']}." if op.get("lowered") else ".")) if op.get("raised") else \
            f"The Opening roll ({op['roll']}) points to {op['stat']}."
        parts = [f"{self.player_name}, {cls['name']} ({cls.get('tag', '')}). {stats}.", shift,
                 cls.get("how_it_came", ""), cls.get("bond", "")]
        if origin:
            parts.append(f"The night it woke: {origin['text']}")
        if debt:
            parts.append(f"What it took: {debt['text']}")
        if cls.get("bond_question"):
            parts.append(f"Your Bond Question: {cls['bond_question']}")
        return "\n\n".join(p for p in parts if p)

    def _begin_trial(self, t: Turn, first: bool = False) -> None:
        st, b, c = self.state, self.bearer, self.state.campaign
        st.place, st.final = "", False
        crucible_text = ""
        if first and self._forced_place:
            place = self.places[self._forced_place]
            st.place, threshold, crucible, crucible_text = place.id, place.threshold, place.crucible, \
                place.crucible_text
        elif first:
            op = self.codex.opening(b.class_id, self.state.opening["stat"]) or {}
            threshold = op.get("threshold") or self._threshold_roll()
            crucible = op.get("crucible") or self._crucible_form()
        elif c.last_threshold:
            agenda = self.codex.by_id("agendas", c.agenda) or {}
            st.final = True
            threshold = agenda.get("last_threshold", "This is the last Threshold.")
            crucible = "the Agenda itself"
            crucible_text = agenda.get("wants", "")
        elif st.legacy_threshold:
            threshold, crucible = st.legacy_threshold, self._crucible_form()
            st.legacy_threshold = ""
        else:
            options = [("t", e["threshold"]) for e in self.codex.thresholds(c.fragments)]
            options += [("p", pid) for pid in sorted(self.places) if self.places[pid].name not in c.atlas]
            kind, value = self.rng.choice(options)
            if kind == "p":
                place = self.places[value]
                st.place, threshold, crucible, crucible_text = place.id, place.threshold, place.crucible, \
                    place.crucible_text
            else:
                threshold, crucible = value, self._crucible_form()
        st.threshold, st.crucible_form, st.crucible_text = threshold, crucible, crucible_text
        st.used_leaves, st.npcs_present, st.leaf = [], [], None
        st.stage = "trial"
        st.tension = 0.2
        st.trial = Trial.begin(b, self.rng, hunger=c.hunger, length=self.length, pages_open=c.pages_open)
        t.title = "THE THRESHOLD"
        self._drain(t)

    def _threshold_roll(self) -> str:
        options = self.codex.thresholds(self.state.campaign.fragments)
        return self.rng.choice(options)["threshold"] if options else "Walk."

    def _crucible_form(self) -> str:
        e = self.codex.by_roll("crucible_forms", self.rng.randint(1, 6))
        return e["form"] if e else "The Crucible"

    # ================================================================ the trial
    def _trial_turn(self, t: Turn) -> None:
        tr = self.trial
        low = t.text.lower().strip()
        if self._global_command(t, low):
            self._after(t)
            return
        p = tr.pending
        if p:
            self._answer_pending(t, p, low)
        elif tr.phase in ("chapter", "crucible"):
            self._chapter_action(t)
        self._after(t)

    def _global_command(self, t: Turn, low: str) -> bool:
        tr, b = self.trial, self.bearer
        if tr.pending and tr.pending["type"] in ("bargain", "judgment"):
            return False
        if low.startswith("?") or low.startswith("ask "):
            self._oracle(t, low.lstrip("?").removeprefix("ask ").strip())
            return True
        if "medkit" in low or "patch" in low and "wound" in low:
            if tr.use_medkit():
                t.variant = None
            else:
                t.say("The medkit is empty, or it cannot help you now.")
            self._drain(t)
            return True
        page = find_page(low)
        if page and ("page" in low or low.startswith(page)) and page != "borrow":
            ok, why = use_page(tr, page, self.rng)
            if ok and page == "speak":
                t.clue = self._truth()
            if not ok:
                t.say(why)
            self._drain(t)
            if ok and page == "step":
                self._drain(t)
            return True
        if "abandon" in low and tr.chapter and tr.chapter.kind == "witness":
            tr.abandon_witness(self.rng)
            self.state.npcs_present = [n for n in self.state.npcs_present if self._is_ally(n)
                                       and n not in (self.state.leaf.npcs if self.state.leaf else [])]
            self._drain(t)
            return True
        if b.class_id == "hauler" and "repair" in low and "hauler_repair" not in tr.used:
            broken = [i for i in b.items if i.uses == 0]
            target = b.find_item(low) if b.find_item(low) in broken else (broken[0] if broken else None)
            if target is not None:
                tr.used.append("hauler_repair")
                target.uses = 1
                t.say(f"You get the {target.name} working again, barely.")
                return True
        return False

    def _answer_pending(self, t: Turn, p: dict, low: str) -> None:
        tr, rng = self.trial, self.rng
        kind = p["type"]
        if kind == "draw":
            if self._talk_without_drawing(t):
                return
            choice = 0
            if p["shapes"]:
                m = re.search(r"\b([1-3])\b", low)
                if m:
                    choice = int(m.group(1)) - 1
                else:
                    for i, shape in enumerate(p["shapes"]):
                        if shape in low or (shape == "ring" and "circle" in low):
                            choice = i
                            break
            t.variant = "touch"
            tr.draw(rng, choice)
        elif kind == "push":
            borrow = "borrow" in low and "borrow" in tr.pages_open
            if borrow:
                tr.add_stress(1, rng)
                tr.decide_push(rng, False, borrow=True)
            else:
                tr.decide_push(rng, _says(low, YES) or "again" in low or "reroll" in low)
        elif kind == "bargain":
            refuse = _says(low, NO)
            stat = next((s for s in STATS if s.value in low), None)
            tr.decide_bargain(rng, not refuse, stat)
            if refuse:
                t.ending = "dead"
        elif kind == "breather":
            choice = "stress" if "stress" in low or "breathe" in low or "calm" in low else \
                "use" if ("use" in low or "restore" in low or "item" in low) else "grit"
            tr.decide_breather(choice, low)
        elif kind == "warning":
            tr.decide_warning("swap" in low or "second" in low or "2" in low or "reverse" in low)
        elif kind == "temptation":
            tr.decide_temptation(_says(low, YES))
        elif kind == "judgment":
            tr.decide_judgment(t.text)
        self._drain(t)

    def _talk_without_drawing(self, t: Turn) -> bool:
        """At a draw, speaking to someone who walks with you is talk, not a touch."""
        st = self.state
        if not st.npcs_present:
            return False
        s = self.transducer.extract(t.text, self._targets())
        if s.intent is not Intent.TALK:
            return False
        t.sensory = s
        self._listen(t)
        return True

    def _chapter_action(self, t: Turn) -> None:
        tr, st = self.trial, self.state
        s = t.sensory = self.transducer.extract(t.text, self._targets())
        ch = tr.chapter
        if s.intent is Intent.FIGHT and s.target in st.npcs_present:
            name = self.npcs[s.target].name
            t.say(f"You turn on {name}, who twists away and puts the dark between you.")
            t.variant, t.npc_name, t.flee_npc = "npc_attack", name, s.target
            self._listen(t)
            return
        stat = s.effective_stat
        explicit = s.stat is not Stat.NONE      # "I hold steady" names Nerve without a verb the senses know
        if s.intent not in ENGAGING and not explicit:
            if s.intent is Intent.TALK or (s.target in st.npcs_present):
                self._listen(t)
            t.variant = "look" if s.intent is Intent.LOOK else "wait"
            return
        if s.intent is Intent.TALK and ch.active_kind in (Kind.HAZARD, Kind.RELIC) and not ch.crucible:
            t.variant = "npc" if st.npcs_present else "alone"
            self._listen(t)
            return
        if "fence_had" not in tr.used and self.bearer.class_id == "fence" and (
                "already had" in t.text.lower() or "i had" in t.text.lower()):
            tr.used.append("fence_had")
            extra = 1
            t.say("You already had just the thing, of course.")
        else:
            extra = 0
        if stat not in ch.accepts:
            stat = self.bearer.best_stat(ch.accepts)
        mode = s.mode if tr.chapter.active_kind is Kind.HOSTILE else ""
        ally = self._ally_present(s.target)
        roll = tr.cast(stat, self.rng, mode=mode, ally=ally is not None, extra=extra)
        t.stat = stat.value
        if roll is None:
            t.variant = "lost"
        elif not tr.pending:
            t.outcome = roll.result.value
        if roll is not None:
            self._social(t, s, roll.result.value)
        self._listen(t)
        self._drain(t)

    def _after(self, t: Turn) -> None:
        """After an input: settle the trial (verdict, death), then offer the next Leaf and pick a beat."""
        st, tr = self.state, self.trial
        if tr is None or st.stage != "trial":
            return
        if tr.phase == "dead":
            self._fallen(t)
            return
        if tr.phase == "verdict":
            self._verdict(t)
            return
        if tr.phase == "draw" and not tr.pending:
            self._beat(t, narrative=True)
            self._offer(t)
        elif tr.phase in ("chapter", "crucible") and not tr.pending:
            self._beat(t, narrative=True)

    def _offer(self, t: Turn) -> None:
        tr = self.trial
        if tr is None or tr.pending or tr.phase != "draw":
            return
        beat, probs = self.director.choose(self._features(t), self.rng, set(DRAWS))
        tr.offer(beat.value, self.rng)
        t.beat_probs.update({f"draw.{k}": v for k, v in probs.items() if Beat(k) in DRAWS})
        self._drain(t)

    # ================================================================ rule events -> leaves, blocks, facts
    def _drain(self, t: Turn) -> None:
        """Move the trial's new events into this turn, attaching Leaf content as Leaves rise."""
        tr = self.trial
        if tr is None:
            return
        events, tr.log[:] = list(tr.log), []
        for e in events:
            t.events.append(e)
            if e.kind == "leaf":
                self._rise_leaf(t, e.data["kind"], e.data["rank"])
            elif e.kind == "crucible":
                self._rise_crucible(t)
            elif e.kind == "cleared":
                if self.state.leaf and self.state.leaf.cleared:
                    t.cleared_text = self.state.leaf.cleared
                self._leave_leaf(t, cleared=True)
                continue
            elif e.kind in ("fled", "collapsed"):
                self._leave_leaf(t, cleared=False)
            elif e.kind == "panic":
                t.panicked = True
                row = self.codex.by_roll("panic", e.data["total"]) or {}
                e.text = f"PANIC: {row.get('name', '').upper()}"
                e.fact = f"You panic. {row.get('effect', '')}".strip()
                if e.data["total"] == 8:
                    for nid in self.state.npcs_present:
                        t.decisions[nid] = self.brains[nid].perceive(SensoryState(aggression=0.9, rapport=-0.8))
            elif e.kind == "echo":
                row = self.codex.by_roll("echoes", e.data["roll"]) or {}
                e.fact = f"The TOME speaks: {row.get('name', '')}."
            elif e.kind == "intervention":
                row = self.codex.by_roll("interventions", e.data["roll"]) or {}
                e.fact = row.get("effect", "")
                t.title = "INTERVENTION"
            elif e.kind == "fragment":
                self._fragment_text(e)
            elif e.kind == "temptation":
                page = e.data["page"]
                e.fact += f" The {page.title()} Page: {PAGE_USE.get(page, '')}."
            elif e.kind == "cast":
                r = e.data["roll"]
                t.blocks.append(dice_block(r["dice"], r["pool"], r["hits"], r["difficulty"], r["result"],
                                           e.data["stat"], e.data["notes"]))
                continue
            elif e.kind in ("push", "borrow"):
                r = e.data["roll"]
                t.blocks.append(dice_block(r["dice"], r["pool"], r["hits"], r["difficulty"], r["result"],
                                           "none", [], label=e.kind.upper()))
                t.outcome = r["result"]
                t.say(e.fact)
                continue
            elif e.kind == "dim":
                t.blocks.append(self._lumen_dice_block(e.data["dice"]))
                continue
            elif e.kind in ("death", "verdict", "taken"):
                pass
            if e.text:
                t.blocks.append(Block(role="event", spans=[span(e.text, e.tone, effect=self._effect(e))],
                                      data={"kind": e.kind}))
            t.say(e.fact)

    @staticmethod
    def _effect(e: Event) -> str:
        return {"panic": "flicker", "lumen_out": "flicker", "dark": "flicker", "crucible": "ripple",
                "cleared": "ripple", "verdict": "ripple", "wound": "pulse", "last_rites": "flicker"}.get(e.kind, "rise")

    @staticmethod
    def _lumen_dice_block(dice: list[int]) -> Block:
        spans = [span("LUMEN ", "lumen", 0.3)]
        for d in dice:
            spans.append(span(f"[{d}]", "absence" if d == 1 else "lumen", 0.0 if d == 1 else 0.6,
                              "flicker" if d == 1 else "rise"))
        return Block(role="dice", spans=spans, data={"dice": dice, "lumen": True})

    def _event_blocks(self, t: Turn) -> list[Block]:
        out, t.blocks = t.blocks, []
        return out

    def _rise_leaf(self, t: Turn, kind: str, rank: str) -> None:
        st, tr = self.state, self.trial
        place = self.places.get(st.place)
        view = None
        if tr.chapter is not None and tr.chapter.rival:
            view = LeafView(id="rival", kind=kind, title="A rival Bearer, slab in hand",
                            text={"plain": "Another Bearer steps out of the dark, their own slab open and lit. "
                                           "They want yours."})
        if view is None and place is not None:
            options = [x for x in place.leaves_of(kind, rank) if x.id not in st.used_leaves]
            if options:
                x = self.rng.choice(options)
                view = LeafView(id=x.id, kind=kind, title=x.title, text=dict(x.text), npcs=list(x.npcs),
                                clue=x.clue, cleared=x.cleared)
        if view is None:
            options = [e for e in self.codex.leaves(kind) if e["id"] not in st.used_leaves] or self.codex.leaves(kind)
            e = self.rng.choice(options) if options else {"id": kind, "title": kind.title()}
            view = LeafView(id=e["id"], kind=kind, title=e.get("title", kind.title()),
                            text={r: e[r] for r in ("plain", "uneasy", "eldritch", "breakthrough") if e.get(r)})
        st.used_leaves.append(view.id)
        st.leaf = view
        followers = [n for n in st.npcs_present if self._is_ally(n)]
        st.npcs_present = list(dict.fromkeys(view.npcs + followers))
        for nid in st.npcs_present:
            if nid not in st.npcs_met:
                st.npcs_met.append(nid)
        t.entering, t.title, t.leaf_kind = True, view.title, kind

    def _rise_crucible(self, t: Turn) -> None:
        st = self.state
        st.leaf = LeafView(id="crucible", kind="crucible", title=st.crucible_form,
                           text={"plain": st.crucible_text} if st.crucible_text else {})
        st.npcs_present = [n for n in st.npcs_present if self._is_ally(n)]
        t.entering, t.title, t.leaf_kind = True, st.crucible_form.upper(), "crucible"

    def _leave_leaf(self, t: Turn, cleared: bool) -> None:
        st = self.state
        leaf = st.leaf
        if leaf is None:
            return
        if cleared and leaf.clue and leaf.clue not in st.clues_revealed and leaf.kind in ("relic", "witness"):
            st.clues_revealed.append(leaf.clue)
            t.clue = t.clue or leaf.clue
        if cleared and leaf.kind == "witness":
            place = self.places.get(st.place)
            for r in (place.rumors if place else []):
                if r not in st.campaign.rumors:
                    st.campaign.rumors.append(r)
                    t.say(f"Someone here names a place you have never seen: {r}")
                    break
        t.leaf_kind = t.leaf_kind or leaf.kind
        st.npcs_present = [n for n in st.npcs_present if self._is_ally(n)]
        st.leaf = None
        self._offscreen(t)

    def _fragment_text(self, e: Event) -> None:
        c = self.state.campaign
        agenda = self.codex.by_id("agendas", c.agenda) or {}
        held = c.fragments + self.trial.fragments_gain
        shown = Campaign(bearer=c.bearer, agenda=c.agenda, fragments=held).fragment_text(
            agenda.get("wants", ""), self.state.seed)
        e.data["glyphs"] = shown
        e.text = shown

    # ================================================================ verdict, Haven, Legacy
    def _verdict(self, t: Turn) -> None:
        st, c = self.state, self.state.campaign
        tr = self.trial
        c.absorb(tr)
        place = self.places.get(st.place)
        where = place.name if place else st.threshold
        if where not in c.atlas:
            c.atlas.append(where)
        if st.final:
            st.stage = "over"
            t.ending = "campaign"
            t.say("You have faced what the TOME wanted. The campaign is over.")
            return
        restocked = c.haven()
        st.stage = "haven"
        st.trial = None
        st.leaf, st.npcs_present = None, []
        t.ending = "verdict"
        t.title = "THE HAVEN"
        t.say("The TOME releases you to a Haven. Your Stress settles, your Grit returns and your Wounds close."
              + (f" You restock the {' and the '.join(restocked)}." if restocked else ""))
        t.blocks.append(self._haven_block())

    def _fallen(self, t: Turn) -> None:
        st, c = self.state, self.state.campaign
        tr = self.trial
        c.absorb(tr)
        c.last_crucible = st.crucible_form
        st.legacy_threshold = f"Where the last Bearer fell, {st.crucible_form.lower()} is waiting."
        st.stage = "fallen"
        st.trial = None
        st.leaf, st.npcs_present = None, []
        t.ending = t.ending or ("taken" if self.bearer.brands and len(self.bearer.brands) >= 2 else "dead")
        t.title = "LEGACY"
        t.say("The TOME outlives its Bearers. Choose who carries it next.")
        names = ", ".join(e["name"] for e in self.codex.entries("classes"))
        t.blocks.append(text_block("prompt", f"Name the next Bearer's class ({names}), or say continue.",
                                   "system"))

    def _legacy_turn(self, t: Turn) -> None:
        st, c = self.state, self.state.campaign
        low = t.text.lower()
        classes = self.codex.classes
        cid = next((k for k, v in classes.items() if v["name"].lower() in low or k in low), None) or \
            self.rng.choice(sorted(classes))
        heir = make_bearer(classes[cid], self.player_name, self.rng, self.codex.entries("origins"))
        op = opening_roll(heir, self.rng)
        st.opening = op.to_dict()
        c.legacy(heir)
        t.say(f"The TOME passes to a new pair of hands: a {classes[cid]['name']}. It keeps its open Pages, its "
              f"Hunger and its Agenda.")
        t.blocks.append(text_block("narration", self._creation_text(classes[cid], st.opening)))
        self._begin_trial(t)
        t.say(st.threshold)
        self._offer(t)

    def _haven_turn(self, t: Turn) -> None:
        c = self.state.campaign
        low = t.text.lower()
        msg = ""
        if low.startswith("?") or low.startswith("ask "):
            self._oracle(t, low.lstrip("?").removeprefix("ask ").strip())
            return
        stat = next((s for s in STATS if s.value in low), None)
        kind = next((k.value for k in KINDS if k.value in low), None)
        page = find_page(low)
        if any(w in low for w in ("onward", "continue", "begin", "next", "threshold", "go on", "leave")):
            if "long" in low:
                self.length = "long"
            elif "short" in low:
                self.length = "short"
            self._begin_trial(t)
            t.say(self.state.threshold)
            if self.state.place:
                t.say(self.places[self.state.place].arrival)
            self._offer(t)
            return
        if "raise" in low and stat:
            _, msg = c.raise_stat(stat)
        elif "talent" in low and kind:
            _, msg = c.buy_talent(kind)
        elif "minimum" in low or "steady" in low:
            _, msg = c.lower_min_stress()
        elif "open" in low and page:
            _, msg = c.open_page(page)
        elif "feed" in low:
            _, msg = c.feed_relic()
        elif low.startswith("name"):
            ok, msg = c.name_agenda(low.removeprefix("name").strip())
            if ok:
                agenda = self.codex.by_id("agendas", c.agenda) or {}
                msg += f" It wants this: {agenda.get('wants', '')}"
        else:
            t.variant = "haven"
        if msg:
            t.say(msg)
        t.blocks.append(self._haven_block())

    def _haven_block(self) -> Block:
        c, b = self.state.campaign, self.bearer
        opts = [f"Insight {b.insight}"]
        for s in STATS:
            if b.base(s) < 4:
                opts.append(f"raise {s.value} ({c.raise_cost(s)})")
        opts.append("talent <hostile|hazard|relic|witness> (5)")
        if b.min_stress > 0:
            opts.append("steady: lower minimum Stress (4)")
        closed = [p for p in PAGE_IDS if p not in c.pages_open]
        if closed:
            opts.append(f"open <page> (4): {', '.join(closed)}")
        if c.relics and c.hunger:
            opts.append("feed the TOME a relic")
        if c.fragments >= FRAGMENTS_TO_NAME and not c.named:
            opts.append("name <what it wants>")
        opts.append("onward (short or long)")
        return text_block("prompt", " · ".join(opts), "system")

    # ================================================================ the oracle and truths
    def _oracle(self, t: Turn, question: str) -> None:
        a = ask(likelihood_of(question), self.rng)
        verdict = "YES" if a.yes else "NO"
        twist = {"complication": ", BUT", "benefit": ", AND"}.get(a.twist, "")
        t.blocks.append(Block(role="event", spans=[span(f"{verdict}{twist}", "tome", effect="ripple")],
                              data={"oracle": a.to_dict(), "question": question}))
        t.say(f"You ask the TOME: {question or 'a question'}. It answers {verdict.lower()}"
              + (", with a complication." if a.twist == "complication" else
                 ", and something more besides." if a.twist == "benefit" else "."))
        if a.unasked:
            t.clue = self._truth()

    def _truth(self) -> str:
        """Something true the TOME tells unasked: a place's lore first, then the canon's hidden half."""
        st = self.state
        place = self.places.get(st.place)
        pool = (place.lore if place else []) + [e["hidden"] for e in self.codex.entries("canon") if e.get("hidden")]
        for truth in pool:
            if truth not in st.truths:
                st.truths.append(truth)
                return truth
        return "The TOME tells you something you already knew, and it is worse the second time."

    # ================================================================ limbic: people
    def _targets(self) -> dict[str, list[str]]:
        return {nid: list(self.npcs[nid].aliases) for nid in self.state.npcs_present}

    def _listen(self, t: Turn) -> None:
        st, s = self.state, t.sensory
        if not st.npcs_present or s is None:
            return
        addressed = s.target if s.target in st.npcs_present else None
        if addressed is None and s.intent is Intent.TALK and len(st.npcs_present) == 1:
            addressed = st.npcs_present[0]
        for nid in st.npcs_present:
            heard = s if nid == addressed else self._overheard(s)
            t.decisions[nid] = self.brains[nid].perceive(heard)
        if addressed is not None:
            t.npc_name = self.npcs[addressed].name
            self._speak(t, addressed, unprompted=False)
        if t.flee_npc in st.npcs_present:
            st.npcs_present.remove(t.flee_npc)

    @staticmethod
    def _overheard(s: SensoryState) -> SensoryState:
        return SensoryState(intent=s.intent, stat=s.stat, target=s.target,
                            aggression=0.5 * s.aggression, deception=0.5 * s.deception,
                            leverage=0.5 * s.leverage, exposure=0.5 * s.exposure,
                            rapport=0.5 * s.rapport, boldness=s.boldness)

    def _social(self, t: Turn, s: SensoryState, result: str) -> None:
        """A lie or a threat aimed at someone is cast with Nerve; how it lands changes what they feel."""
        if s.target not in self.state.npcs_present and not (self.state.npcs_present and s.intent is Intent.TALK):
            return
        if s.deception > 0.5:
            if result == "success":
                s.deception *= 0.25
                t.say("The lie lands. They believe it, for now.")
            elif result == "fail":
                s.deception = min(1.0, s.deception + 0.3)
                s.exposure = min(1.0, s.exposure + 0.2)
                t.say("The lie is transparent. They see straight through it.")
        elif s.aggression > 0.6:
            if result == "success":
                s.leverage = min(1.0, s.leverage + 0.4)
                t.say("The threat lands. They are afraid of you now.")
            elif result == "fail":
                s.rapport = max(-1.0, s.rapport - 0.3)
                t.say("The threat rings hollow and only earns contempt.")

    def _speak(self, t: Turn, nid: str, unprompted: bool) -> None:
        brain = self.brains[nid]
        spec = brain.spec
        decision = t.decisions.get(nid) or brain.last_decision
        if decision is None:
            decision = brain.perceive(SensoryState())
            t.decisions[nid] = decision
        source = self._npc_name_by_numeric(decision.branch_source_npc)
        prompt = compile_npc_prompt(spec, decision, self.player_name, source)
        line = self.voice.speak(DialogueRequest(
            npc=spec, decision=decision, system_prompt=prompt, player_text=t.text,
            player_name=self.player_name, unprompted=unprompted, source_name=source,
            history=self.state.history[-2:],
        ))
        t.npc_lines.append(Block(role="npc", speaker=spec.name, data={"npc": nid},
                                 spans=[Span(f"{spec.name}: ", color=spec.color, glow=0.7, tone="witness"),
                                        Span(f"“{line}”", color=spec.color, glow=0.35, tone="witness")]))

    def _npc_name_by_numeric(self, numeric_id: int) -> str | None:
        return next((s.name for s in self.npcs.values() if s.numeric_id == numeric_id), None)

    def _offscreen(self, t: Turn) -> None:
        """Between chapters, people who know the Bearer talk about them behind their back."""
        met = [n for n in self.state.npcs_met if n in self.brains]
        for a in met:
            for b in sorted(self.brains):
                if a == b or self.rng.random() > 0.5:
                    continue
                trust = self.brains[b].spec.trust.get(a, 0.0)
                if trust == 0.0:
                    continue
                t.gossip.append(gossip(self.brains[a], self.brains[b], trust).describe())

    def _is_ally(self, nid: str) -> bool:
        brain = self.brains[nid]
        return brain.spec.plots[int(brain.momentum.argmax())].role == "ally"

    def _ally_present(self, target: str) -> str | None:
        """Someone whose dominant plot is to help you lends a die (the app's take on Help)."""
        for nid in self.state.npcs_present:
            if nid != target and self._is_ally(nid):
                return nid
        return None

    # ================================================================ cortex: the Director's beat
    def _beat(self, t: Turn, narrative: bool) -> None:
        st = self.state
        allowed = {Beat.CALM, Beat.FORESHADOW}
        if self._next_clue(peek=True):
            allowed.add(Beat.CLUE)
        if st.npcs_present and not t.npc_lines:
            allowed.add(Beat.NPC)
        beat, probs = self.director.choose(self._features(t), self.rng, allowed & NARRATIVE)
        t.beat = beat
        t.beat_probs.update({k: v for k, v in probs.items() if Beat(k) in NARRATIVE})
        if beat is Beat.CALM:
            st.tension = max(0.0, st.tension - 0.08)
            t.beat_note = "Let the moment breathe. Nothing new happens."
        elif beat is Beat.FORESHADOW:
            st.tension = min(1.0, st.tension + 0.12)
            t.beat_note = "Something ahead draws closer. Foreshadow it without revealing it."
        elif beat is Beat.CLUE:
            t.clue = t.clue or self._next_clue()
            t.beat_note = "Reveal the truth given through what the Bearer notices."
        elif beat is Beat.NPC:
            nid = max(st.npcs_present, key=self._hostility)
            t.beat_note = "Someone acts on their own agenda; their words are shown separately."
            self._speak(t, nid, unprompted=True)

    def _hostility(self, nid: str) -> float:
        brain = self.brains[nid]
        b = brain.spec.role_index("betray")
        return float(brain.momentum[b]) if b is not None else 0.0

    def _next_clue(self, peek: bool = False) -> str | None:
        st = self.state
        leaf = st.leaf
        if leaf and leaf.clue and leaf.clue not in st.clues_revealed:
            if not peek:
                st.clues_revealed.append(leaf.clue)
            return leaf.clue
        return None

    def _features(self, t: Turn) -> DirectorFeatures:
        st, b, tr = self.state, self.bearer, self.trial
        ch = tr.chapter if tr else None
        s = t.sensory
        if t.outcome:
            st.last_outcome = t.outcome
            st.tension = min(1.0, max(0.0, st.tension + {"fail": 0.08, "short": 0.04, "success": -0.05}[t.outcome]))
        last = st.last_outcome
        return DirectorFeatures(
            lumen=(tr.lumen / LUMEN_MAX) if tr else 1.0, grit=b.grit / max(1, b.grit_max), stress=b.stress / 10,
            wounds=b.open_wounds / 3, progress=(tr.cleared / max(1, tr.need)) if tr else 0.0, tension=st.tension,
            in_chapter=1.0 if ch else 0.0, depth_left=(ch.remaining / max(1, ch.depth)) if ch else 0.0,
            crucible=1.0 if ch and ch.crucible else 0.0, last_fail=float(last == "fail"),
            last_short=float(last == "short"), last_success=float(last == "success"),
            aggression=s.aggression if s else 0.0, boldness=s.boldness if s else 0.0,
            npc_present=1.0 if st.npcs_present else 0.0,
            npc_hostility=max((self._hostility(n) for n in st.npcs_present), default=0.0),
            hunger=st.campaign.hunger / 6,
        )

    # ================================================================ voice + frame
    def _register(self, t: Turn) -> str:
        return self.codex.register_for(self.bearer.stress, panicked=t.panicked)

    def _compose(self, t: Turn) -> TomeFrame:
        st = self.state
        leaf = st.leaf
        register = self._register(t)
        place = self.places.get(st.place)
        req = NarrationRequest(
            player_text=t.text, intent=t.sensory.intent.value if t.sensory else "touch", stat=t.stat,
            register=register, register_style=self.codex.register_style(register), facts=t.facts,
            place=place.name if place else "", leaf_title=leaf.title if leaf else "",
            leaf_kind=t.leaf_kind or (leaf.kind if leaf else ""),
            leaf_text=(leaf.prose(register) if leaf else "") if t.entering else "",
            entering=t.entering, outcome=t.outcome, variant=t.variant, npc_name=t.npc_name,
            cleared_text=t.cleared_text, beat=t.beat, beat_fact=t.beat_fact, beat_note=t.beat_note, clue=t.clue,
            tension=st.tension, lumen=(self.trial.lumen / LUMEN_MAX) if self.trial else 1.0,
            history=st.history[-3:], ending=t.ending,
        )
        narration = self.voice.narrate(req)
        self._remember(narration)
        blocks: list[Block] = [text_block("echo", f"> {t.text}", effect="none")]
        if t.title:
            blocks.append(text_block("title", t.title, "tome" if t.title.isupper() else None, glow=0.6,
                                     effect="ripple"))
        blocks += t.blocks
        tone = "tome" if register == "breakthrough" else None
        blocks.append(text_block("narration", narration, tone))
        blocks += t.npc_lines
        if st.stage not in ("over",):
            blocks.append(self._prompt_block())
        return self._frame(blocks, t)

    def _prompt_block(self) -> Block:
        st, tr = self.state, self.trial
        if st.stage == "haven":
            return Block(role="prompt", spans=[span("The Haven. What will you do?", "system")], data={"hint": True})
        if st.stage == "fallen" or tr is None:
            return Block(role="prompt", spans=[span("Who carries it next?", "system")], data={"hint": True})
        p = tr.pending
        if p:
            kind = p["type"]
            if kind == "draw":
                if p["mode"] == "shapes":
                    spans = [span("Shapes rise from the stone:  ", "system")]
                    for i, shape in enumerate(p["shapes"], 1):
                        spans.append(span(f"{i} {SHAPE_GLYPH.get(shape, '?')} {shape}   ", "tome", 0.8, "pulse"))
                    return Block(role="draw", spans=spans, data={"mode": "shapes", "shapes": p["shapes"]})
                if p["mode"] == "dot":
                    return Block(role="draw", spans=[span("•", "tome", 1.0, "pulse"),
                                                     span("  touch the stone", "system")], data={"mode": "dot"})
                return Block(role="draw", spans=[span("…", "absence", 0.0, "none")], data={"mode": "dark"})
            hints = {
                "push": "Push it? (push / let it stand" + (" / borrow" if "borrow" in tr.pages_open else "") + ")",
                "bargain": "The TOME offers a Bargain. (accept, naming the stat to lose / refuse)",
                "breather": "Take one mercy: stress / grit / use",
                "warning": "Keep their order, or swap them?",
                "temptation": "Open the Page? (yes / no)",
                "judgment": f"{p.get('question') or 'Answer your Bond Question again.'}",
            }
            return Block(role="prompt", spans=[span(hints.get(kind, ""), "tome" if kind in ("bargain", "judgment")
                                                    else "system")], data={"pending": kind, "hint": True})
        ch = tr.chapter
        if ch is None:
            return Block(role="prompt", spans=[span("What do you do?", "system")], data={"hint": True})
        spans = []
        for s in ch.accepts:
            spans.append(span(f"{STAT_GLYPH[s.value]} {s.value}  ", "stone", 0.3, "pulse"))
        ring = "◉" * ch.progress + "○" * ch.remaining
        spans.append(span(f"  {ring}  ", "absence"))
        spans.append(span("What do you do?", "system"))
        return Block(role="prompt", spans=spans, data={"accepts": [s.value for s in ch.accepts], "hint": True})

    def _frame(self, blocks: list[Block], t: Turn) -> TomeFrame:
        st, b, tr, c = self.state, self.bearer, self.trial, self.state.campaign
        self.seq += 1
        ch = tr.chapter if tr else None
        status = {
            "bearer": {"name": b.name, "class": b.class_name, "stats": b.stats(),
                       "penalties": {s.value: b.penalty(s) for s in STATS if b.penalty(s)},
                       "min_stress": b.min_stress, "insight": b.insight,
                       "wounds": [w.to_dict() for w in b.wounds], "brands": list(b.brands),
                       "phobias": list(b.phobias), "talents": list(b.talents), "dead": b.dead},
            "grit": [b.grit, b.grit_max], "stress": b.stress, "lumen": [tr.lumen if tr else 0, LUMEN_MAX],
            "items": [{"name": i.name, "uses": i.uses, "max": i.max_uses} for i in b.items],
            "stage": st.stage, "threshold": st.threshold,
            "chapter": {"title": st.leaf.title if st.leaf else (st.threshold[:60] if tr else ""),
                        "kind": ch.active_kind.value if ch else "", "depth": ch.depth if ch else 0,
                        "progress": ch.progress if ch else 0, "difficulty": ch.difficulty if ch else 0,
                        "crucible": bool(ch and ch.crucible),
                        "index": tr.cleared if tr else 0, "total": tr.need if tr else 0},
            "pending": tr.pending if tr else None,
            "peek": list(tr.peek) if tr else [],
            "campaign": {"hunger": c.hunger, "pages": list(c.pages_open), "fragments": c.fragments,
                         "named": c.named, "atlas": list(c.atlas), "rumors": list(c.rumors), "relics": c.relics,
                         "fallen": list(c.fallen), "trials": c.trials},
            "npcs": [{"id": n, "name": self.npcs[n].name, "color": self.npcs[n].color} for n in st.npcs_present],
            "won": st.stage == "haven" or (st.stage == "over" and not b.dead),
        }
        register = self.codex.register_for(b.stress, panicked=t.panicked)
        mood = {
            "lumen": round((tr.lumen / LUMEN_MAX) if tr else 1.0, 3), "tension": round(st.tension, 3),
            "stress": round(b.stress / 10, 3), "dark": bool(tr and tr.lumen_out), "register": register,
            "pulse": "heartbeat" if b.stress >= 5 else "breath",
        }
        debug = {
            "sensory": t.sensory.to_dict() if t.sensory else None, "beat": t.beat.value, "beat_probs": t.beat_probs,
            "decisions": {k: v.to_dict() for k, v in t.decisions.items()}, "gossip": t.gossip,
            "events": [e.to_dict() for e in t.events], "agenda": c.agenda,
        }
        return TomeFrame(seq=self.seq, blocks=blocks, status=status, mood=mood, effects=self._effects(t),
                         over=st.stage == "over", debug=debug)

    @staticmethod
    def _effects(t: Turn) -> list[dict]:
        out = []
        for e in t.events:
            if e.kind in ("bite", "wound", "grit"):
                out.append({"kind": "shudder", "strength": 0.5})
            elif e.kind in ("dark", "lumen_out") or (e.kind == "dim" and e.data.get("lost")):
                out.append({"kind": "dim", "strength": 0.8 if e.kind == "lumen_out" else 0.4})
            elif e.kind == "panic":
                out.append({"kind": "panic", "strength": 1.0})
            elif e.kind in ("kindle", "verdict"):
                out.append({"kind": "flare", "strength": 0.8})
        return out

    def _remember(self, text: str) -> None:
        if text:
            self.state.history = (self.state.history + [text])[-6:]

    # ================================================================ persistence
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        (d / "brains").mkdir(parents=True, exist_ok=True)
        rng_state = self.rng.getstate()
        voice_rng = getattr(getattr(self.voice, "fallback", self.voice), "rng", None)
        blob = {
            "version": SAVE_VERSION, "player_name": self.player_name, "length": self.length, "seq": self.seq,
            "state": self.state.to_dict(),
            "rng": [rng_state[0], list(rng_state[1]), rng_state[2]],
            "voice_rng": None if voice_rng is None else
            [voice_rng.getstate()[0], list(voice_rng.getstate()[1]), voice_rng.getstate()[2]],
        }
        (d / "campaign.json").write_text(json.dumps(blob, indent=1))
        for nid, brain in self.brains.items():
            brain.save(d / "brains" / f"{nid}.pt")
        return d

    def load(self, directory: str | Path) -> None:
        d = Path(directory)
        blob = json.loads((d / "campaign.json").read_text())
        if blob.get("version") != SAVE_VERSION:
            raise ValueError("save does not match this engine version")
        self.player_name = blob["player_name"]
        self.length = blob.get("length", "short")
        self.seq = blob["seq"]
        self.state = GameState.from_dict(blob["state"])
        r = blob["rng"]
        self.rng.setstate((r[0], tuple(r[1]), r[2]))
        voice_rng = getattr(getattr(self.voice, "fallback", self.voice), "rng", None)
        if voice_rng is not None and blob.get("voice_rng"):
            v = blob["voice_rng"]
            voice_rng.setstate((v[0], tuple(v[1]), v[2]))
        for nid, brain in self.brains.items():
            brain.load(d / "brains" / f"{nid}.pt")


__all__ = ["TomeEngine", "GameState", "LeafView", "SHAPES", "COLORED", "roll_2d6"]
