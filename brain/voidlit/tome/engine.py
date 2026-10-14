"""The Tome: the Dungeon Master's turn loop.

One player input runs through this pipeline::

    senses     transducer (llama.cpp GBNF pass, or heuristics) -> SensoryState
    rules      intent + approach -> dice pools, obstacles, harm, stress, items, movement
    limbic     every NPC in the scene perceives (BindsNET, R-STDP); the addressed one answers
    pressure   panic checks, the Lumen gutters
    cortex     the Director (SpikingJelly) picks a beat; the rules apply it
    voice      narration (llama.cpp, or the offline phrase banks)
    frame      semantic blocks for the voxel Tome (or the terminal)

The LLM never decides an outcome. By the time the voice is called, every die
has been rolled and every consequence applied. It is only told what
happened.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from ..cortex.director import Director
from ..cortex.features import Beat, DirectorFeatures
from ..limbic.contagion import gossip
from ..limbic.npc_brain import BrainDecision, NPCBrain
from ..rules.character import Approach, Character, PANIC_THRESHOLD
from ..rules.clocks import Clock
from ..rules.dice import Outcome, PoolRoll
from ..rules.lumen import LUMEN_MAX, Lumen
from ..rules.obstacles import Obstacle, resolve_exchange
from ..rules.panic import panic_check
from ..rules.sector import Chapter, SectorDeck
from ..senses.heuristic import HeuristicTransducer
from ..senses.state import Intent, SensoryState
from ..voice.narrator import OfflineVoice
from ..voice.prompts import compile_npc_prompt
from ..voice.requests import DialogueRequest, NarrationRequest
from .content import Scenario, load_scenario
from .frames import APPROACH_GLYPH, PALETTE, Block, Span, TomeFrame, dice_block, text_block

ENGAGING = {Intent.FIGHT, Intent.FORCE, Intent.SNEAK, Intent.SEARCH, Intent.USE}
SAVE_VERSION = 1


@dataclass
class GameState:
    seed: int
    character: Character
    lumen: Lumen
    deck: SectorDeck
    clocks: list[Clock]
    turn: int = 0
    chapter_turns: int = 0
    obstacles: list[Obstacle] = field(default_factory=list)
    obstacle_pips0: dict[str, int] = field(default_factory=dict)
    npcs_present: list[str] = field(default_factory=list)
    npcs_met: list[str] = field(default_factory=list)
    clues_revealed: list[str] = field(default_factory=list)
    tension: float = 0.2
    since_threat: int = 0
    since_respite: int = 0
    last_outcome: str | None = None
    rest_used: bool = False
    light_taken: bool = False
    position_bonus: int = 0
    lose_turn: bool = False
    twisted: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    over: bool = False
    won: bool = False

    def to_dict(self) -> dict:
        return {
            "seed": self.seed, "character": self.character.to_dict(), "lumen": self.lumen.to_dict(),
            "deck": self.deck.to_dict(), "clocks": [c.to_dict() for c in self.clocks], "turn": self.turn,
            "chapter_turns": self.chapter_turns, "obstacles": [o.to_dict() for o in self.obstacles],
            "obstacle_pips0": self.obstacle_pips0, "npcs_present": self.npcs_present, "npcs_met": self.npcs_met,
            "clues_revealed": self.clues_revealed, "tension": self.tension, "since_threat": self.since_threat,
            "since_respite": self.since_respite, "last_outcome": self.last_outcome, "rest_used": self.rest_used,
            "light_taken": self.light_taken, "position_bonus": self.position_bonus, "lose_turn": self.lose_turn,
            "twisted": self.twisted,
            "history": self.history, "over": self.over, "won": self.won,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GameState":
        d = dict(d)
        d["character"] = Character.from_dict(d["character"])
        d["lumen"] = Lumen.from_dict(d["lumen"])
        d["deck"] = SectorDeck.from_dict(d["deck"])
        d["clocks"] = [Clock.from_dict(c) for c in d["clocks"]]
        d["obstacles"] = [Obstacle.from_dict(o) for o in d["obstacles"]]
        return cls(**d)


@dataclass
class Turn:
    """Everything that happened in one turn, gathered for the voice and the frame."""

    text: str
    sensory: SensoryState | None = None
    facts: list[str] = field(default_factory=list)
    dice: list[Block] = field(default_factory=list)
    events: list[Block] = field(default_factory=list)
    npc_lines: list[Block] = field(default_factory=list)
    effects: list[dict] = field(default_factory=list)
    decisions: dict[str, BrainDecision] = field(default_factory=dict)
    outcome: Outcome | None = None
    obstacle: Obstacle | None = None
    defeated: str | None = None
    clue: str | None = None
    entering: Chapter | None = None
    describe: bool = False
    beat: Beat = Beat.CALM
    beat_fact: str = ""
    beat_note: str = ""
    beat_probs: dict[str, float] = field(default_factory=dict)
    gossip: list[str] = field(default_factory=list)
    ending: str | None = None
    panic_due: bool = False
    variant: str | None = None
    npc_name: str | None = None
    flee_npc: str | None = None

    def event(self, text: str, color_key: str, glow: float = 0.5, effect: str = "rise") -> None:
        self.events.append(text_block("event", text, color_key, glow=glow, effect=effect))


class TomeEngine:
    def __init__(self, scenario: Scenario | None = None, *, transducer=None, voice=None,
                 director: Director | None = None, seed: int = 0, background: str | None = None,
                 player_name: str = "Wanderer") -> None:
        self.scenario = scenario or load_scenario()
        self.transducer = transducer or HeuristicTransducer()
        self.voice = voice or OfflineVoice(self.scenario.narration, seed=seed)
        self.director = director or Director(seed=seed)
        self.player_name = player_name
        self.rng = random.Random(seed)
        self.seq = 0
        bg_id = background or self.rng.choice(sorted(self.scenario.backgrounds))
        bg = self.scenario.backgrounds[bg_id]
        self.state = GameState(
            seed=seed, character=bg.make_character(player_name),
            lumen=Lumen(interval=self.scenario.lumen_interval),
            deck=SectorDeck.build(self.scenario.chapters, self.scenario.per_act, self.rng),
            clocks=self.scenario.new_clocks(),
        )
        self.brains: dict[str, NPCBrain] = {
            nid: NPCBrain(spec, seed=seed * 1000 + spec.numeric_id) for nid, spec in self.scenario.npcs.items()
        }

    # ================================================================ public API
    @property
    def chapter(self) -> Chapter:
        return self.scenario.chapters[self.state.deck.current]

    def start(self) -> TomeFrame:
        t = Turn(text="")
        self._enter_chapter(t, self.state.deck.current, gossip_ok=False)
        narration = f"{self.scenario.intro}\n\n{self.chapter.text}"
        self._remember(narration)
        blocks = [text_block("title", self.scenario.title.upper(), glow=0.9, effect="ripple"),
                  text_block("title", self.chapter.title, glow=0.6),
                  text_block("narration", narration)]
        return self._frame(blocks + t.npc_lines + t.events + [self._prompt_block()], t)

    def turn(self, text: str) -> TomeFrame:
        st = self.state
        if st.over:
            return self._frame([text_block("system", "The Tome is closed. Begin a new run.")], Turn(text=text))
        t = Turn(text=text)
        t.sensory = self.transducer.extract(text, self._targets())
        st.turn += 1
        st.chapter_turns += 1
        st.since_threat += 1
        st.since_respite += 1

        if st.lose_turn:
            st.lose_turn = False
            t.facts.append("Panic steals the moment; whatever the player meant to do, they don't.")
            t.event("PANIC holds you. You lose the moment.", "stress", effect="flicker")
        else:
            self._act(t)
        if not st.over:
            self._listen(t)
        if not st.over:
            if t.panic_due:
                self._panic(t)
            self._lumen(t)
        if not st.over:
            self._beat(t)
        if st.character.dead and not st.over:
            st.over, t.ending = True, "dead"
        return self._compose(t)

    # ================================================================ rules: actions
    def _act(self, t: Turn) -> None:
        s, st = t.sensory, self.state
        ob = self._target_obstacle(s)
        intent = s.intent

        if intent is Intent.USE and self._use_item(t, ob):
            return
        if intent is Intent.FIGHT and s.target in st.npcs_present:
            name = self.scenario.npcs[s.target].name
            t.facts.append(f"The player attacks {name}, who twists away and puts distance between them.")
            t.variant, t.npc_name = "npc_attack", name
            t.flee_npc = s.target
            return
        if ob is not None and intent in ENGAGING:
            self._engage(t, ob, s)
            return
        if intent is Intent.SEARCH:
            self._search(t)
        elif intent is Intent.LOOK:
            t.describe = True
            t.facts.append("The player takes in their surroundings.")
        elif intent is Intent.REST:
            self._rest(t)
        elif intent is Intent.ADVANCE:
            self._advance(t)
        elif intent is Intent.FLEE:
            self._flee(t, ob)
        elif intent is Intent.TALK:
            if not st.npcs_present:
                t.facts.append("The player speaks, but no one is here to answer.")
                t.variant = "alone"
            elif s.target in st.npcs_present or len(st.npcs_present) == 1:
                t.variant = "npc"
                self._social_roll(t)
                if t.outcome is not None:
                    t.variant = None
        else:
            t.facts.append("The player acts; nothing in the scene resists or changes.")

    def _engage(self, t: Turn, ob: Obstacle, s: SensoryState) -> None:
        st = self.state
        approach = s.effective_approach
        if approach is Approach.NONE:
            approach = Approach.GRIT
        bonus, helper = self._help(s)
        roll, notes = st.character.roll(approach, self.rng, intent=s.intent.value,
                                        bonus=st.position_bonus + bonus)
        notes += self._bonus_notes(st.position_bonus, helper)
        st.position_bonus = 0
        ex = resolve_exchange(ob, approach, roll, dark=st.lumen.dark)
        t.obstacle, t.outcome = ob, roll.outcome
        self._show_roll(t, roll, approach, notes)
        t.facts.append(f"The player tackles {ob.name} with {approach.value}. Result: {roll.outcome.value}.")
        if not ex.matched and not ex.removed:
            t.facts.append(f"That approach finds no purchase on {ob.name}; it is unharmed.")
            t.variant = "unmatched"
        elif not ex.matched:
            t.facts.append(f"The player brute-forces {ob.name} the hard way; it gives a little.")
        elif ex.removed and not ex.cleared:
            t.facts.append(f"{ob.name} is weakened but not overcome.")
        if ex.cleared:
            t.defeated = ob.defeat_text
            st.obstacles.remove(ob)
            t.event(f"OVERCOME: {ob.name}", "success", glow=0.9, effect="ripple")
            if st.character.relieve_stress(1):
                t.event("-1 STRESS", "hull", glow=0.3)
            if not self._blocking():
                t.facts.append(f"The way on is open. {self.chapter.exit_text}")
        else:
            t.event(f"{ob.name}  {self._pips(ob)}", "system")
        self._bite(t, ex.bite_harm, ex.bite_stress, ob.name)
        if not ex.cleared and roll.outcome is Outcome.FAIL and not (ob.harm or ob.stress):
            t.facts.append("The failed attempt is loud, and the noise carries deep into the station.")
            self._tick_worst_clock(t, 1)
        self._tension_from(roll.outcome)

    @staticmethod
    def _bonus_notes(position: int, helper: str | None) -> list[str]:
        notes = [f"+{position} position"] if position else []
        return notes + ([f"+1 {helper} helps"] if helper else [])

    def _help(self, s: SensoryState) -> tuple[int, str | None]:
        """An NPC whose dominant plot is alliance lends a hand (+1 die); schemers and the attacked don't."""
        for nid in self.state.npcs_present:
            if nid != s.target and self._is_ally(nid):
                return 1, self.scenario.npcs[nid].name.split()[0]
        return 0, None

    def _search(self, t: Turn) -> None:
        st = self.state
        bonus, helper = self._help(t.sensory)
        roll, notes = st.character.roll(Approach.INSIGHT, self.rng, intent="search",
                                        bonus=st.position_bonus + bonus)
        notes += self._bonus_notes(st.position_bonus, helper)
        st.position_bonus = 0
        t.outcome = roll.outcome
        self._show_roll(t, roll, Approach.INSIGHT, notes)
        if roll.outcome is Outcome.FAIL:
            t.facts.append("The search turns up nothing but the feeling of being watched.")
            self._bite(t, 0, 1, "the dark")
        else:
            clue = self._next_clue(lore=True)
            if clue:
                t.clue = clue
            if roll.outcome.succeeded and self.chapter.light and not st.light_taken:
                st.light_taken = True
                gained = st.lumen.restore(self.chapter.light)
                if gained:
                    t.facts.append("The player finds a light source and feeds it to the Tome.")
                    t.event(f"+{gained} LUMEN", "lumen", glow=1.0, effect="ripple")
                    t.effects.append({"kind": "flare", "strength": 0.8})
            if roll.outcome is Outcome.MIXED:
                t.facts.append("It takes too long; something in the station notices the delay.")
                self._tick_worst_clock(t, 1)
        self._tension_from(roll.outcome)

    def _rest(self, t: Turn) -> None:
        st, ch = self.state, self.chapter
        active = self._active()
        if active:
            t.facts.append(f"There is no rest with {active[0].name} here.")
            t.obstacle, t.variant = active[0], "denied"
            self._bite(t, 0, 1, active[0].name)
            return
        if st.rest_used:
            t.facts.append("The player tries to rest again, but the station won't allow it twice.")
            t.variant = "spent"
            return
        st.rest_used = True
        big = ch.respite
        healed = st.character.heal(3 if big else 2)
        relieved = st.character.relieve_stress(3 if big else 2)
        st.tension = max(0.0, st.tension - 0.2)
        st.since_respite = 0
        t.facts.append("The player rests, briefly and uneasily.")
        if healed:
            t.event(f"+{healed} HULL", "hull", glow=0.6)
        if relieved:
            t.event(f"-{relieved} STRESS", "hull", glow=0.4)

    def _advance(self, t: Turn) -> None:
        st = self.state
        blocking = self._blocking()
        if blocking:
            t.facts.append(f"{blocking[0].name} still blocks the way on.")
            t.obstacle, t.variant = blocking[0], "blocked"
            return
        hazards = self._active()
        if hazards:
            ob = hazards[0]
            roll, notes = st.character.roll(Approach.GUILE, self.rng, intent="advance", bonus=st.position_bonus)
            notes += self._bonus_notes(st.position_bonus, None)
            st.position_bonus = 0
            t.obstacle, t.outcome = ob, roll.outcome
            self._show_roll(t, roll, Approach.GUILE, notes)
            t.facts.append(f"The player dashes past {ob.name}. Result: {roll.outcome.value}.")
            if roll.outcome is Outcome.FAIL:
                self._bite(t, ob.harm, ob.stress, ob.name)
            elif roll.outcome is Outcome.MIXED and (ob.harm or ob.stress):
                self._bite(t, 0, 1, ob.name)
            self._tension_from(roll.outcome)
            if st.character.dead:
                return
        if st.deck.at_end:
            st.over, st.won, t.ending = True, True, "won"
            t.facts.append("The player escapes the sector.")
            t.event("ESCAPED", "lumen", glow=1.0, effect="ripple")
            return
        st.deck.advance()
        self._enter_chapter(t, st.deck.current)

    def _flee(self, t: Turn, ob: Obstacle | None) -> None:
        st = self.state
        if ob is None:
            t.facts.append("The player backs away, but nothing pursues.")
            return
        roll, notes = st.character.roll(Approach.GUILE, self.rng, intent="flee")
        t.obstacle, t.outcome = ob, roll.outcome
        self._show_roll(t, roll, Approach.GUILE, notes)
        if roll.outcome.succeeded or roll.outcome is Outcome.MIXED:
            st.position_bonus = 1
            t.facts.append(f"The player breaks contact with {ob.name} and finds a better position (+1 die next roll).")
            if not ob.blocking:
                st.obstacles.remove(ob)
                t.facts.append(f"{ob.name} is left behind.")
        if roll.outcome is Outcome.FAIL:
            self._bite(t, ob.harm, ob.stress, ob.name)
        elif roll.outcome is Outcome.MIXED:
            self._bite(t, 0, 1, ob.name)
        self._tension_from(roll.outcome)

    def _use_item(self, t: Turn, ob: Obstacle | None) -> bool:
        """Light and healing items have their own effect. Returns False to fall through to an engage."""
        st = self.state
        st_char = st.character
        text = t.text
        item = st_char.find_item(text, tag="light" if "light" in text.lower() or "flare" in text.lower() else None)
        if item is None and any(w in text.lower() for w in ("heal", "medkit", "stim", "patch", "bandage")):
            item = st_char.find_item(text, tag="heal")
        if item is None:
            return False
        if "light" in item.tags:
            st_char.spend_item(item)
            gained = st.lumen.restore(item.light)
            t.facts.append(f"The player spends the {item.name}; its light pours into the Tome.")
            t.event(f"+{gained} LUMEN  ({item.name})", "lumen", glow=1.0, effect="ripple")
            t.effects.append({"kind": "flare", "strength": 1.0})
            t.variant = "light"
            return True
        if "heal" in item.tags:
            st_char.spend_item(item)
            healed = st_char.heal(2)
            st_char.relieve_stress(1)
            t.facts.append(f"The player uses the {item.name} on their wounds.")
            t.event(f"+{healed} HULL  ({item.name})", "hull", glow=0.7)
            t.variant = "heal"
            return True
        return False

    def _social_roll(self, t: Turn) -> None:
        """Lies and threats aimed at an NPC are rolled; the result changes what the NPC *feels*."""
        s, st = t.sensory, self.state
        if s.deception > 0.5:
            roll, notes = st.character.roll(Approach.GUILE, self.rng, intent="talk")
            self._show_roll(t, roll, Approach.GUILE, notes)
            t.outcome = roll.outcome
            if roll.outcome.succeeded:
                s.deception *= 0.25
                t.facts.append("The lie lands. They believe it, for now.")
            elif roll.outcome is Outcome.FAIL:
                s.deception = min(1.0, s.deception + 0.3)
                s.exposure = min(1.0, s.exposure + 0.2)
                t.facts.append("The lie is transparent. They see straight through it.")
        elif s.aggression > 0.6:
            roll, notes = st.character.roll(Approach.GRIT, self.rng, intent="talk")
            self._show_roll(t, roll, Approach.GRIT, notes)
            t.outcome = roll.outcome
            if roll.outcome.succeeded:
                s.leverage = min(1.0, s.leverage + 0.4)
                t.facts.append("The threat lands. They are afraid of the player now.")
            elif roll.outcome is Outcome.FAIL:
                s.rapport = max(-1.0, s.rapport - 0.3)
                t.facts.append("The threat rings hollow and only earns contempt.")

    # ================================================================ limbic: NPCs listen
    def _listen(self, t: Turn) -> None:
        st, s = self.state, t.sensory
        if not st.npcs_present:
            return
        addressed = s.target if s.target in st.npcs_present else None
        if addressed is None and s.intent is Intent.TALK and len(st.npcs_present) == 1:
            addressed = st.npcs_present[0]
        for nid in st.npcs_present:
            heard = s if nid == addressed else self._overheard(s)
            t.decisions[nid] = self.brains[nid].perceive(heard)
            if nid not in st.npcs_met:
                st.npcs_met.append(nid)
        if addressed is not None:
            t.npc_name = self.scenario.npcs[addressed].name
            self._speak(t, addressed, unprompted=False)
        if t.flee_npc in st.npcs_present:
            st.npcs_present.remove(t.flee_npc)

    @staticmethod
    def _overheard(s: SensoryState) -> SensoryState:
        return SensoryState(intent=s.intent, approach=s.approach, target=s.target,
                            aggression=0.5 * s.aggression, deception=0.5 * s.deception,
                            leverage=0.5 * s.leverage, exposure=0.5 * s.exposure,
                            rapport=0.5 * s.rapport, boldness=s.boldness)

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
                                 spans=[Span(f"{spec.name}: ", color=spec.color, glow=0.7),
                                        Span(f"“{line}”", color=spec.color, glow=0.35)]))
        t.facts.append(f"{spec.name} speaks to the player (their words are shown separately).")

    def _npc_name_by_numeric(self, numeric_id: int) -> str | None:
        for spec in self.scenario.npcs.values():
            if spec.numeric_id == numeric_id:
                return spec.name
        return None

    # ================================================================ pressure
    def _bite(self, t: Turn, harm: int, stress: int, source: str) -> None:
        ch = self.state.character
        if harm > 0:
            res = ch.take_harm(harm, self.rng)
            t.facts.append(f"{source} hurts the player.")
            t.event(f"-{res.hull_lost} HULL" if res.hull_lost else "HULL BREACHED", "fail", glow=0.8, effect="pulse")
            t.effects.append({"kind": "shudder", "strength": min(1.0, 0.3 * harm)})
            self.state.tension = min(1.0, self.state.tension + 0.05 * harm)
            if res.last_light is not None:
                lr = res.last_light
                t.dice.append(dice_block(list(lr.dice), lr.pool, lr.hits, lr.outcome.value, "grit", ["last light"]))
                if res.dead:
                    t.facts.append("The player's last light fails. They die.")
                    t.event("YOU DIE", "fail", glow=1.0, effect="flicker")
                else:
                    t.facts.append("Past their limit, the player stays standing through sheer grit, but bleeding.")
                    t.event(f"-{res.attribute_lost} GRIT  (bleeding)", "fail", glow=0.8)
        if stress > 0:
            if ch.add_stress(stress):
                t.panic_due = True
            t.event(f"+{stress} STRESS  ({ch.stress})", "stress", glow=0.5, effect="flicker")

    def _panic(self, t: Turn) -> None:
        ch = self.state.character
        check = panic_check(ch, self.rng)
        if check.held:
            t.event(f"PANIC CHECK {check.dice[0]}+{check.dice[1]}+{ch.insight} vs {check.stress}: you hold", "system")
            return
        eff = check.effect
        ch.relieve_stress(2)  # the fear discharges into the panic itself
        t.event(f"PANIC: {eff.name.upper()}", "stress", glow=1.0, effect="flicker")
        t.effects.append({"kind": "panic", "strength": 1.0})
        t.facts.append(f"The player panics: {eff.text}")
        for e in eff.effects:
            kind, _, arg = e.partition(":")
            if kind == "stress":
                n = int(arg)
                if n > 0:
                    ch.add_stress(n)
                else:
                    ch.relieve_stress(-n)
            elif kind == "condition":
                ch.conditions.add(arg)
            elif kind == "lose_turn":
                self.state.lose_turn = True
            elif kind == "drop_item" and ch.inventory:
                dropped = ch.inventory.pop(self.rng.randrange(len(ch.inventory)))
                t.event(f"LOST: {dropped.name}", "fail")
            elif kind == "harm":
                self._bite(t, int(arg), 0, "panic")
            elif kind == "lumen":
                self.state.lumen.burn(-int(arg))
                t.effects.append({"kind": "dim", "strength": 0.6})
            elif kind == "rapport":
                for nid in self.state.npcs_present:
                    snap = SensoryState(aggression=0.8, rapport=-0.8)
                    t.decisions[nid] = self.brains[nid].perceive(snap)

    def _lumen(self, t: Turn) -> None:
        lum = self.state.lumen
        was_dark = lum.dark
        roll = lum.advance(self.rng)
        if roll is not None and roll.lost:
            t.event(f"-{roll.lost} LUMEN  ({lum.dice}/{LUMEN_MAX})", "lumen", glow=0.3, effect="flicker")
            t.effects.append({"kind": "dim", "strength": 0.4 * roll.lost})
            t.facts.append("The Tome's light gutters and shrinks.")
            if lum.dice <= 2 and self.state.character.add_stress(1):
                t.panic_due = True
        if lum.dark and not was_dark:
            t.event("THE LUMEN IS OUT. THE DARK HAS THE SECTOR.", "dread", glow=1.0, effect="flicker")
            t.facts.append("The Tome's light goes out entirely. Everything here can see the player now.")

    def _tension_from(self, outcome: Outcome) -> None:
        st = self.state
        delta = {Outcome.FAIL: 0.08, Outcome.MIXED: 0.04, Outcome.SUCCESS: -0.04, Outcome.CRITICAL: -0.08}[outcome]
        st.tension = min(1.0, max(0.0, st.tension + delta))
        st.last_outcome = outcome.value

    # ================================================================ cortex: the Director's beat
    def _beat(self, t: Turn) -> None:
        st = self.state
        allowed = self._allowed_beats(spoke=bool(t.npc_lines))
        beat, probs = self.director.choose(self._features(t), self.rng, allowed)
        t.beat, t.beat_probs = beat, probs
        if beat is Beat.CALM:
            st.tension = max(0.0, st.tension - 0.08)
            t.beat_note = "Let the moment breathe with quiet, unsettling atmosphere. Nothing new happens."
        elif beat is Beat.ESCALATE:
            t.beat_note = "Something unseen draws closer. Foreshadow it without revealing it."
            self._tick_worst_clock(t, 2 if st.lumen.dark else 1)
            st.tension = min(1.0, st.tension + 0.15)
            st.since_threat = 0
            t.effects.append({"kind": "shudder", "strength": 0.3})
        elif beat is Beat.CLUE:
            t.clue = t.clue or self._next_clue()
            t.beat_note = "Reveal the truth given below through what the player notices."
        elif beat is Beat.NPC:
            self._npc_beat(t)
        elif beat is Beat.HAZARD:
            self._hazard_beat(t)
        elif beat is Beat.RESPITE:
            ch = st.character
            if ch.hull < ch.hull_max and ch.hull / ch.hull_max <= 1.0 - ch.stress / 10:
                t.event(f"+{ch.heal(1)} HULL", "hull", glow=0.5)
            else:
                t.event(f"-{ch.relieve_stress(1)} STRESS", "hull", glow=0.4)
            st.tension = max(0.0, st.tension - 0.25)
            st.since_respite = 0
            t.beat_note = "Offer a small mercy: warmth, air, a moment of safety."
        elif beat is Beat.TWIST:
            self._twist_beat(t)
        elif beat is Beat.ADVANCE:
            t.beat_fact = self.chapter.exit_text
            t.beat_note = "Draw the player's eye to the way onward; this place has nothing more to give."

    def _allowed_beats(self, spoke: bool = False) -> set[Beat]:
        st = self.state
        ch = st.character
        allowed = {Beat.CALM, Beat.ESCALATE}
        if st.turn > 1:
            allowed.add(Beat.TWIST)
            if not self._active() and st.since_threat >= 2:
                allowed.add(Beat.HAZARD)
        if self.brains and st.turn > 1 and not spoke:
            allowed.add(Beat.NPC)
        if self._next_clue(peek=True):
            allowed.add(Beat.CLUE)
        if not self._active() and not st.lumen.dark and (ch.hull < ch.hull_max or ch.stress > 0):
            allowed.add(Beat.RESPITE)
        if not self._blocking() and st.chapter_turns >= 1:
            allowed.add(Beat.ADVANCE)
        return allowed

    def _features(self, t: Turn) -> DirectorFeatures:
        st, s = self.state, t.sensory
        ch = st.character
        active = self._active()
        pips0 = sum(st.obstacle_pips0.get(o.id, len(o.pips)) for o in active) or 1
        hostility = 0.0
        for nid in st.npcs_present:
            brain = self.brains[nid]
            b = brain.spec.role_index("betray")
            if b is not None:
                hostility = max(hostility, float(brain.momentum[b]))
        last = st.last_outcome
        return DirectorFeatures(
            lumen=st.lumen.fraction, hull=ch.hull / max(1, ch.hull_max), stress=ch.stress / 10,
            progress=st.deck.progress, tension=st.tension, since_threat=min(1.0, st.since_threat / 6),
            since_respite=min(1.0, st.since_respite / 10), chapter_turns=min(1.0, st.chapter_turns / 8),
            obstacle_active=1.0 if active else 0.0,
            obstacle_left=sum(len(o.pips) for o in active) / pips0 if active else 0.0,
            last_fail=float(last == "fail"), last_mixed=float(last == "mixed"),
            last_success=float(last in ("success", "critical")),
            aggression=s.aggression if s else 0.0, boldness=s.boldness if s else 0.0,
            clock_max=max((c.fraction for c in st.clocks), default=0.0),
            npc_present=1.0 if st.npcs_present else 0.0, npc_hostility=hostility,
        )

    def _tick_worst_clock(self, t: Turn, n: int) -> None:
        live = [c for c in self.state.clocks if not c.fired]
        if not live:
            return
        clock = max(live, key=lambda c: (c.fraction, -self.state.clocks.index(c)))
        fired = clock.tick(n)
        t.event(f"⟳ {clock.name}  {clock.filled}/{clock.segments}", "dread", glow=0.6)
        if fired:
            t.event(f"{clock.name.upper()}", "dread", glow=1.0, effect="ripple")
            if clock.consequence == "lumen":
                lost = self.state.lumen.burn(2)
                t.beat_fact = f"{clock.name}: the station's power lurches and the Tome's light is dragged down with it."
                t.event(f"-{lost} LUMEN", "lumen", effect="flicker")
                t.effects.append({"kind": "dim", "strength": 0.8})
            elif clock.consequence:
                ob = self._add_obstacle(clock.consequence)
                t.beat_fact = f"{clock.name}: {ob.text}"

    def _npc_beat(self, t: Turn) -> None:
        st = self.state
        if st.npcs_present:
            def hostility(nid: str) -> float:
                b = self.brains[nid].spec.role_index("betray")
                return float(self.brains[nid].momentum[b]) if b is not None else 0.0
            nid = max(st.npcs_present, key=hostility)
        else:
            candidates = sorted(st.npcs_met) or sorted(self.brains)
            nid = self.rng.choice(candidates)
            st.npcs_present.append(nid)
            t.beat_fact = f"{self.brains[nid].spec.name} steps out of the dark."
        t.beat_note = "Someone acts on their own agenda; their words are shown separately."
        if nid not in [blk.data.get("npc") for blk in t.npc_lines]:
            self._speak(t, nid, unprompted=True)
        if nid not in st.npcs_met:
            st.npcs_met.append(nid)

    def _hazard_beat(self, t: Turn) -> None:
        st = self.state
        st.since_threat = 0
        st.tension = min(1.0, st.tension + 0.15)
        if not self._active():
            act = max(1, min(3, self.chapter.act))
            pool = self.scenario.hazards.get(act) or [o for ids in self.scenario.hazards.values() for o in ids]
            if pool:
                ob = self._add_obstacle(self.rng.choice(sorted(pool)))
                t.beat_fact = ob.text
                t.beat_note = "The environment turns dangerous: describe the new hazard."
                return
        t.beat_note = "The environment presses in; the strain shows."
        self._bite(t, 0, 1, "the station")

    def _twist_beat(self, t: Turn) -> None:
        st = self.state
        st.tension = min(1.0, st.tension + 0.12)
        active = [o for o in self._active() if o.id not in st.twisted and len(o.pips) <= 3]
        if active:
            ob = active[0]
            st.twisted.append(ob.id)
            pip = self.rng.choice(["grit", "guile", "insight"])
            ob.pips.append(pip)
            t.beat_fact = f"{ob.name} changes, adapting to the player."
            t.event(f"{ob.name} adapts  +{APPROACH_GLYPH[pip]}", "dread", glow=0.6)
        elif st.lumen.dice > 3:
            st.lumen.burn(1)
            t.beat_fact = "One of the Tome's lights gutters out, for no reason at all."
            t.event(f"-1 LUMEN  ({st.lumen.dice}/{LUMEN_MAX})", "lumen", effect="flicker")
            t.effects.append({"kind": "dim", "strength": 0.4})
        else:
            self._tick_worst_clock(t, 1)
        t.beat_note = "A complication or reversal: something the player relied on is not what it seemed."

    # ================================================================ scene management
    def _enter_chapter(self, t: Turn, chapter_id: str, gossip_ok: bool = True) -> None:
        st = self.state
        ch = self.scenario.chapters[chapter_id]
        if gossip_ok and ch.act > self.scenario.chapters[st.deck.order[st.deck.index - 1]].act:
            healed, relieved = st.character.heal(1), st.character.relieve_stress(1)
            if healed or relieved:
                t.facts.append("Between decks, the player finds a moment to steady themselves.")
                t.event(f"BETWEEN DECKS  +{healed} HULL  -{relieved} STRESS", "hull", glow=0.4)
        followers = [nid for nid in st.npcs_present if self._is_ally(nid)]
        if gossip_ok:
            self._offscreen(t)
        st.obstacles = [self.scenario.new_obstacle(o) for o in ch.obstacles]
        st.twisted = []
        st.obstacle_pips0 = {o.id: len(o.pips) for o in st.obstacles}
        st.npcs_present = list(dict.fromkeys(ch.npcs + followers))
        st.chapter_turns = 0
        st.rest_used = False
        st.light_taken = False
        t.entering = ch
        t.facts.append(f"The player enters {ch.title}.")
        for nid in followers:
            t.facts.append(f"{self.brains[nid].spec.name} follows the player.")
        for ob in st.obstacles:
            t.event(f"{ob.name}  {self._pips(ob)}", "system")

    def _offscreen(self, t: Turn) -> None:
        """Between chapters, NPCs who know the player talk about them behind their back."""
        met = [n for n in self.state.npcs_met if n in self.brains]
        for a in met:
            for b in sorted(self.brains):
                if a == b or self.rng.random() > 0.5:
                    continue
                trust = self.brains[b].spec.trust.get(a, 0.0)
                if trust == 0.0:
                    continue
                res = gossip(self.brains[a], self.brains[b], trust)
                t.gossip.append(res.describe())

    def _is_ally(self, nid: str) -> bool:
        brain = self.brains[nid]
        dom = int(brain.momentum.argmax())
        return brain.spec.plots[dom].role == "ally"

    def _add_obstacle(self, obstacle_id: str) -> Obstacle:
        ob = self.scenario.new_obstacle(obstacle_id)
        n = sum(1 for o in self.state.obstacles if o.id.split("#")[0] == obstacle_id)
        if n:
            ob.id = f"{obstacle_id}#{n + 1}"
        self.state.obstacles.append(ob)
        self.state.obstacle_pips0[ob.id] = len(ob.pips)
        return ob

    def _active(self) -> list[Obstacle]:
        return [o for o in self.state.obstacles if not o.cleared]

    def _blocking(self) -> list[Obstacle]:
        return [o for o in self._active() if o.blocking]

    def _target_obstacle(self, s: SensoryState) -> Obstacle | None:
        active = self._active()
        for o in active:
            if o.id == s.target:
                return o
        non_blocking = [o for o in active if not o.blocking]
        return (non_blocking or active or [None])[0]

    def _targets(self) -> dict[str, list[str]]:
        targets: dict[str, list[str]] = {}
        for nid in self.state.npcs_present:
            targets[nid] = list(self.scenario.npcs[nid].aliases)
        for o in self._active():
            words = [w for w in o.name.lower().replace("the ", "").split() if len(w) > 3]
            targets[o.id] = words + [o.kind]
        return targets

    def _next_clue(self, peek: bool = False, lore: bool = False) -> str | None:
        """Chapter clues first; scenario lore only when the player digs for it (a successful search)."""
        st = self.state
        for clue in self.chapter.clues + (self.scenario.lore if lore else []):
            if clue not in st.clues_revealed:
                if not peek:
                    st.clues_revealed.append(clue)
                return clue
        return None

    @staticmethod
    def _pips(ob: Obstacle) -> str:
        return " ".join(APPROACH_GLYPH[p] for p in ob.pips)

    def _show_roll(self, t: Turn, roll: PoolRoll, approach: Approach, notes: list[str]) -> None:
        t.dice.append(dice_block(list(roll.dice), roll.pool, roll.hits, roll.outcome.value, approach.value, notes))

    def _remember(self, text: str) -> None:
        if text:
            self.state.history = (self.state.history + [text])[-6:]

    # ================================================================ voice + frame
    def _compose(self, t: Turn) -> TomeFrame:
        st, s = self.state, t.sensory
        ch = t.entering or self.chapter
        ob = t.obstacle
        req = NarrationRequest(
            player_text=t.text, intent=s.intent.value if s else "other",
            approach=s.effective_approach.value if s else "none", chapter_title=ch.title,
            chapter_text=ch.text, entering_chapter=bool(t.entering) or t.describe, facts=t.facts,
            outcome=t.outcome.value if t.outcome else None, variant=t.variant, npc_name=t.npc_name,
            obstacle_name=ob.name if ob else None, obstacle_kind=ob.kind if ob else None,
            obstacle_text=ob.text if ob else None, defeated_text=t.defeated, beat=t.beat,
            beat_fact=t.beat_fact, beat_note=t.beat_note, clue=t.clue, tension=st.tension,
            lumen=st.lumen.fraction, history=st.history[-3:], ending=t.ending,
        )
        narration = self.voice.narrate(req)
        self._remember(narration)
        blocks: list[Block] = [text_block("echo", f"> {t.text}", effect="none")]
        if t.entering:
            blocks.append(text_block("title", t.entering.title, glow=0.6, effect="ripple"))
        blocks += t.dice
        blocks.append(text_block("narration", narration))
        blocks += t.npc_lines
        blocks += t.events
        if not st.over:
            blocks.append(self._prompt_block())
        return self._frame(blocks, t)

    def _prompt_block(self) -> Block:
        active = self._active()
        spans = []
        for o in active:
            spans.append(Span(f"{o.name} ", color=PALETTE["dread"], glow=0.4))
            for p in o.pips:
                spans.append(Span(APPROACH_GLYPH[p], color=PALETTE[p], glow=0.8, effect="pulse"))
            spans.append(Span("   ", effect="none"))
        spans.append(Span("What do you do?", color=PALETTE["system"], glow=0.2))
        return Block(role="prompt", spans=spans)

    def _frame(self, blocks: list[Block], t: Turn) -> TomeFrame:
        st = self.state
        ch = st.character
        self.seq += 1
        status = {
            "character": {"name": ch.name, "background": ch.background, "grit": ch.grit, "guile": ch.guile,
                          "insight": ch.insight, "conditions": sorted(ch.conditions)},
            "hull": [ch.hull, ch.hull_max], "stress": ch.stress, "lumen": [st.lumen.dice, LUMEN_MAX],
            "chapter": {"id": self.chapter.id, "title": self.chapter.title, "act": self.chapter.act,
                        "index": st.deck.index, "total": len(st.deck.order)},
            "obstacles": [{"id": o.id, "name": o.name, "kind": o.kind, "pips": list(o.pips)} for o in self._active()],
            "clocks": [{"id": c.id, "name": c.name, "filled": c.filled, "segments": c.segments} for c in st.clocks],
            "inventory": [{"name": i.name, "uses": i.uses} for i in ch.inventory],
            "npcs": [{"id": n, "name": self.scenario.npcs[n].name, "color": self.scenario.npcs[n].color}
                     for n in st.npcs_present],
            "won": st.won,
        }
        mood = {
            "lumen": round(st.lumen.fraction, 3), "tension": round(st.tension, 3),
            "stress": round(ch.stress / 10, 3), "dark": st.lumen.dark,
            "pulse": "heartbeat" if ch.stress >= PANIC_THRESHOLD else "breath",
        }
        debug = {
            "sensory": t.sensory.to_dict() if t.sensory else None,
            "beat": t.beat.value, "beat_probs": t.beat_probs,
            "decisions": {k: v.to_dict() for k, v in t.decisions.items()}, "gossip": t.gossip,
        }
        return TomeFrame(seq=self.seq, blocks=blocks, status=status, mood=mood, effects=t.effects,
                         over=st.over, debug=debug)

    # ================================================================ persistence
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        (d / "brains").mkdir(parents=True, exist_ok=True)
        rng_state = self.rng.getstate()
        voice_rng = getattr(getattr(self.voice, "fallback", self.voice), "rng", None)
        blob = {
            "version": SAVE_VERSION, "scenario": self.scenario.id, "player_name": self.player_name,
            "seq": self.seq, "state": self.state.to_dict(),
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
        if blob.get("version") != SAVE_VERSION or blob.get("scenario") != self.scenario.id:
            raise ValueError("save does not match this scenario or engine version")
        self.player_name = blob["player_name"]
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
