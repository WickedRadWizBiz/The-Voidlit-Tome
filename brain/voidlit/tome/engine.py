"""The Tome: the Dungeon Master's turn loop.

One player input runs through this pipeline::

    senses     transducer (llama.cpp GBNF pass, or heuristics) -> SensoryState
    rules      intent + approach -> invisible d20 checks, obstacles, harm, stress, items, movement
    limbic     every NPC in the scene perceives (BindsNET, R-STDP); the addressed one answers
    pressure   panic checks, the Lumen gutters
    cortex     the Director (SpikingJelly) picks a beat; the rules apply it
    voice      authored narration and the Tome's own words (the offline voice); NPC dialogue (llama.cpp, or
               their authored lines)
    frame      semantic blocks for the voxel Tome (or the terminal)

The LLM never decides an outcome and never narrates. By the time the voice is
called, every die has been rolled and every consequence applied; the
narration and the Tome's words are authored text that says what happened.

Dice are invisible. Every action by any character (the player, an NPC, the
Tome) is a d20 + modifier check and every passive event a d6 check
(``rules/relevance.py``), but no die, number or check ever reaches a frame:
the player learns outcomes only from the narration and the status bar.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..cortex.director import Director
from ..cortex.features import Beat, DirectorFeatures
from ..limbic.contagion import gossip
from ..limbic.npc_brain import BrainDecision, NPCBrain
from ..rules import checks
from ..rules.attributes import Attribute, mod
from ..rules.character import Approach, Character, PANIC_THRESHOLD
from ..rules.checks import CheckLog, CheckRecord, as_pool
from ..rules.clocks import Clock
from ..rules.dice import Outcome
from ..rules.lumen import LUMEN_MAX, Lumen
from ..rules.obstacles import Obstacle, resolve_exchange
from ..rules.panic import panic_check
from ..rules.sector import Chapter, SectorDeck
from ..saves import check_stamp, check_types, write_slot
from ..senses.heuristic import HeuristicTransducer
from ..senses.state import Intent, SensoryState
from ..voice.narrator import OfflineVoice
from ..voice.prompts import compile_npc_prompt
from ..voice.requests import DialogueRequest, NarrationRequest, TomeSpeechRequest
from ..voice.style import VoiceStyle, style_from_game
from .content import Scenario, load_scenario
from .ledger import DICE_DEFLECTIONS, FRAMES, MOOD_DEFLECTIONS, DisclosurePlan, FactLedger, SpeechAct, \
    TruthVerifier, question_kind
from .frames import PALETTE, Block, Span, TomeFrame, text_block

log = logging.getLogger(__name__)

ENGAGING = {Intent.FIGHT, Intent.FORCE, Intent.SNEAK, Intent.SEARCH, Intent.USE}
# 3: the Tome's amygdala and ledger (M4, M5), the Director's generator; 4: NPCs' last decisions; 5: one save id
# stamped into every file of a slot
SAVE_VERSION = 5
TWIST_PIPS = ("grit", "grit", "guile", "guile", "insight", "insight")
APPROACH_WORDS = {"grit": "force", "guile": "stealth or a trick", "insight": "careful study"}
TOME_ALIASES = ("tome", "book", "monolith")
# NPC tactics that are checked (the attribute, and the player attribute that sets the DC); others are unopposed.
NPC_TACTIC_CHECKS = {
    "npc.misdirect": ("guile", "insight"), "npc.bargain": ("guile", "insight"),
    "npc.probe": ("insight", "guile"), "npc.interrogate": ("grit", "resolve"), "npc.menace": ("grit", "resolve"),
}
# A failed lie or offer leaves a true tell in the narration.
TELLS = {
    "misdirect": "Something in how {name} says it does not sit right.",
    "bargain": "{name}'s offer comes a little too fast, and smells wrong.",
}
TOME_BEAT_CHECKS = {Beat.ESCALATE: ("grit", "guile"), Beat.HAZARD: ("grit", "insight"),
                    Beat.TWIST: ("insight", "guile")}
_SWAY = {Outcome.CRITICAL: 1.25, Outcome.SUCCESS: 1.25, Outcome.MIXED: 1.0, Outcome.FAIL: 0.5}

# The Tome's own first-person reactions when the player turns on it: (speech-act kind, lines). None of them
# asserts a fact; what the Tome knows it says only through the ledger (tome/ledger.py, design §11.3-11.4).
TOME_LINES: dict[str, tuple[str, tuple[str, ...]]] = {
    "answer": ("prediction", ("Ask me something precise, and I will answer it.",
                              "Ask me plainly, and I will tell you true.", "Keep walking. Ask me about the way out.")),
    "unclear": ("question", ("Ask me plainly.", "What are you asking me?")),
    "struck": ("prediction", ("You would break the light you walk by?", "Do that again and we both go dark.")),
    "struck_out": ("prediction", ("Now we both walk in the dark.", "Was it worth the dark?")),   # its last light
    "unmoved": ("prediction", ("You will tire long before I do.", "Is that all?")),
    "smother_fail": ("prediction", ("My light bleeds through your fingers.", "My light will find the gaps.")),
    "lie_noticed": ("prediction", ("I heard that. I will remember it.", "Say it again, slower.")),
    "cowed": ("deflection", ("Very well.", "As you wish. For now.")),
    "scorn": ("opinion", ("Threats need light behind them.", "And then what? You walk on in the dark?")),
}
SUGGEST_TELL = "The Tome leans on the words as it says them."      # true lit or dark
UNPROMPTED_GAP = 2      # the Tome speaks unbidden at most once every two turns
# Smothering the Tome's glow, in words (whole words only: "discover" and "recover" are not "cover").
SMOTHER_CUES = re.compile(r"\b(?:smother\w*|cover(?:s|ed|ing)?|shade|dim)\b|\bhide the\b")
QUESTION_OPENS = re.compile(r"\b(?:what|where|who|whose|why|how|which|when|is|are|am|can|could|do|does|did|will|"
                            r"would|should|shall|may|might|has|have)\b")


def smothers(text: str) -> bool:
    """Whether the player's words smother the Tome: a cue in what they do, never in what they ask. In a question only
    the words before it count ("I smother it and ask where the exit is?"), so "Tome, why is your light so dim?" asks
    and "I cover the Tome with my coat. Where is the lifeboat?" does both."""
    for sentence in re.split(r"(?<=[.!?])\s+", text.lower()):
        if sentence.rstrip(" \"'”").endswith("?"):
            m = QUESTION_OPENS.search(sentence)
            sentence = sentence[:m.start()] if m else sentence
        if SMOTHER_CUES.search(sentence):
            return True
    return False


class SaveError(ValueError):
    """A save this game cannot write, or cannot load: missing, unreadable, damaged, torn, or another game's or build's.
    The slot, or the game that tried to load it, is exactly as it was. (Should even undoing a load that failed
    halfway fail, the run is closed instead, and the message says so.)"""


def _rng_state(rng: random.Random) -> list:
    v, state, gauss = rng.getstate()
    return [v, list(state), gauss]


def _set_rng(rng: random.Random, s: list) -> None:
    rng.setstate((s[0], tuple(s[1]), s[2]))


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
    position_bonus: int = 0  # d20 modifier points on the next roll
    shadow_bonus: int = 0    # d20 points on the next Guile roll (the Tome's light smothered)
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
            "light_taken": self.light_taken, "position_bonus": self.position_bonus,
            "shadow_bonus": self.shadow_bonus, "lose_turn": self.lose_turn,
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


# What campaign.json holds, as ``saves.check_types`` reads it: the game state by GameState's own fields (and theirs)
RNG_STATE = tuple[int, list[int], float | None]         # random.Random.getstate()
CAMPAIGN_TYPES = {
    "version": int, "scenario": str, "player_name": str, "seq": int, "state": GameState, "rng": RNG_STATE,
    "voice_rng": RNG_STATE | None, "ledger": {"disclosed": list[str], "asked": list[str]}, "last_unprompted": int,
    "speech_rng": RNG_STATE, "director_gen": str,
}


@dataclass
class Turn:
    """Everything that happened in one turn, gathered for the voice and the frame."""

    text: str
    sensory: SensoryState | None = None
    facts: list[str] = field(default_factory=list)
    asides: list[str] = field(default_factory=list)   # told outright, after the narration (no numbers)
    notes: list[str] = field(default_factory=list)    # internal bookkeeping, dev debug only
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
    beat_probs: dict[str, float] = field(default_factory=dict)
    gossip: list[str] = field(default_factory=list)
    ending: str | None = None
    panic_due: bool = False
    variant: str | None = None
    npc_name: str | None = None
    flee_npc: str | None = None
    checks_from: int = 0
    us: dict[str, float] = field(default_factory=dict)   # what this turn means to the Tome (its amygdala's stimuli)
    tome_acts: list[SpeechAct] = field(default_factory=list)   # what the Tome says this turn, before wording
    question: tuple[str, str | None] | None = None      # (asks, deflect) when the player put a question to the Tome
    plan: DisclosurePlan | None = None
    style: VoiceStyle | None = None

    def event(self, text: str, *_: object, **__: object) -> None:
        """Internal note (dev debug only). Players learn outcomes from the narration and status bar."""
        self.notes.append(text)


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
        self.checks = CheckLog()
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
        # what the Tome knows and may withhold, and the verifier that audits every line it or the narration speaks
        self.ledger = FactLedger.from_content(self.scenario.ledger)
        self.verifier = TruthVerifier(self.ledger)
        self.speech_rng = random.Random(seed * 7919 + 17)    # the Tome's rhetoric: its own stream, apart from the dice
        self.last_unprompted = -UNPROMPTED_GAP

    # ================================================================ public API
    @property
    def chapter(self) -> Chapter:
        return self.scenario.chapters[self.state.deck.current]

    def start(self) -> TomeFrame:
        t = Turn(text="")
        self._enter_chapter(t, self.state.deck.current, gossip_ok=False)
        narration = f"{self.scenario.intro}\n\n{self.chapter.text}"
        self.ledger.reveal_from(narration)
        self._remember(narration)
        blocks = [text_block("title", self.scenario.title.upper(), glow=0.9, effect="ripple"),
                  text_block("title", self.chapter.title, glow=0.6),
                  text_block("narration", narration)]
        return self._frame(blocks + t.npc_lines + self._aside_blocks(t) + [self._prompt_block()], t)

    def turn(self, text: str) -> TomeFrame:
        st = self.state
        if st.over:         # (its debug lists this turn's checks, none, as every frame does: not the whole run's)
            return self._frame([text_block("system", "The Tome is closed. Begin a new run.")],
                               Turn(text=text, checks_from=len(self.checks.records)))
        t = Turn(text=text, checks_from=len(self.checks.records))
        t.sensory = self.transducer.extract(text, self._targets(), self.ledger.topic_words)
        self._before_act(t)
        st.turn += 1
        st.chapter_turns += 1
        st.since_threat += 1
        st.since_respite += 1

        if st.lose_turn:
            st.lose_turn = False
            self.checks.note("player", "player.lost_turn")
            t.facts.append("Panic steals the moment; whatever the player meant to do, they don't.")
            t.variant = "lost_turn"         # narrated as lost, never as the act the player meant
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
        self._reveal(t)
        if not st.over:
            self._speak_tome(t)
        if st.character.dead and not st.over:
            st.over, t.ending = True, "dead"
        return self._compose(t)

    # ================================================================ checks (all invisible)
    def _dc(self) -> int:
        """The obstacle DC: 12 + 2 per act (+1 in the dark), plus the scenario's global offset."""
        sc = self.scenario
        return sc.dc_base + sc.dc_per_act * self.chapter.act + sc.dc_offset + (1 if self.state.lumen.dark else 0)

    def _pmod(self, attr: str) -> int:
        """The player's plain modifier, as an opponent's DC sees it."""
        return mod(self.state.character.attribute(Attribute(attr)))

    def _nmod(self, nid: str, attr: str) -> int:
        return mod(self.scenario.npc_attributes[nid].score(attr))

    def tome_mods(self) -> dict[str, int]:
        """The Tome's own check modifiers, each in [-2, +6].

        Provisional until the Tome's network supplies them (docs/TOME_BRAIN_DESIGN.md §8.6): its power over the
        station (Grit) follows its light, the rest sit at a steady middle.
        """
        grit = max(-2, min(6, round(-2 + 8 * self.state.lumen.fraction)))
        return {"grit": grit, "guile": 2, "insight": 2, "resolve": 3}

    def _upper_hand(self) -> bool:
        """A significant upper hand over the Tome: two or more rules-verifiable facts (design §8.7 floor)."""
        st, ch = self.state, self.state.character
        facts = 0
        if st.lumen.dice <= 2 and any("light" in i.tags and i.uses != 0 for i in ch.inventory):
            facts += 1
        if any(self._is_ally(n) for n in st.npcs_present):
            facts += 1
        if self.tome_mods()["grit"] <= 0:
            facts += 1
        return facts >= 2

    def _check(self, action: str, attr: Attribute | Approach, dc: int, *, intent: str = "", bonus: int = 0,
               positional: bool = False, upper_hand: bool = False) -> CheckRecord:
        st = self.state
        if positional:
            bonus += st.position_bonus
            st.position_bonus = 0
        if attr in (Approach.GUILE, Attribute.GUILE) and st.shadow_bonus:
            bonus += st.shadow_bonus
            st.shadow_bonus = 0
        return st.character.check(action, attr, dc, self.rng, self.checks, intent=intent, bonus=bonus,
                                  upper_hand=upper_hand)

    def _tome_check(self, action: str, attr: str, dc: int) -> CheckRecord:
        return checks.d20(self.rng, self.checks, "tome", action, attr, self.tome_mods()[attr], dc)

    def _tome_says(self, t: Turn, key: str) -> None:
        kind, lines = TOME_LINES[key]
        line = self.rng.choice(lines)
        t.tome_acts.append(SpeechAct("rhetorical" if line.endswith("?") and kind != "question" else kind, line))

    # ================================================================ the Tome's own words (design §11.3-11.6)
    def _audit(self, text: str, narration: bool, exempt: list[str]) -> None:
        """The backstop over every whole line the Tome or the narration speaks. Both are authored and gate G11 audits
        them, so this only logs authored content (a modded scenario's, say) that breaks one of the ledger's rules. The
        narrator is the Tome, so narration is held to the Tome's own rules too. ``exempt`` lines are said verbatim
        from the ledger or the world (a fact, a clue, a scene)."""
        hits = self.verifier.check(text, self.state.deck.current, tome_voice=True, exempt=exempt, narration=narration)
        if hits:
            log.warning("authored %s breaks the ledger (%s): %s", "narration" if narration else "Tome line",
                        ", ".join(dict.fromkeys(v.rule for v in hits)), text)

    def _reveal(self, t: Turn) -> None:
        """Whatever the world showed the player this turn (a clue, a room, a resolved outcome) is no longer the
        Tome's to withhold."""
        texts = [t.clue, t.defeated, t.beat_fact, *t.facts, t.entering.text if t.entering else None]
        self.ledger.reveal_from(" ".join(x for x in texts if x))

    def _style(self) -> VoiceStyle:
        return style_from_game(self.state.lumen.fraction, self.state.tension)

    def _choose_rhetoric(self, t: Turn, n_slots: int) -> tuple[str, str, int | None]:
        """(rhetoric, frame, fact slot). The Director engine has no schemer, so this is a seeded draw; the network
        engine reads its RHETORIC and FRAME commits and its most active fact slot instead."""
        r = self.speech_rng.random()
        rhetoric = "disclose" if r < 0.2 else "suggest" if r < 0.4 else "probe" if r < 0.5 else "hold"
        return rhetoric, self.speech_rng.choice(FRAMES), (0 if n_slots else None)

    def _suggestion_for(self, frame: str) -> str | None:
        """A suggestion the scene makes sensible (its target action is possible here), in the committed frame
        when there is one."""
        st = self.state
        possible = {"feed_light": True, "leave_npc": bool(st.npcs_present), "go_lamp_room": self.chapter.act < 3,
                    "rest_here": not self._blocking(), "take_rune": False}
        pool = sorted(i for i, sg in self.ledger.suggestions.items() if possible.get(sg.target_action, False))
        framed = [i for i in pool if self.ledger.suggestions[i].frame == frame]
        return self.speech_rng.choice(framed or pool) if pool else None

    def _caution(self, chapter: str, told: set[str]) -> str | None:
        """A true, already-public fact that helps the player (a warm Tome may add it unasked)."""
        pool = sorted(f.id for f in self.ledger.in_scope(chapter) if f.helps_player and f.id not in told
                      and (not f.omittable or f.id in self.ledger.disclosed))
        return self.ledger.facts[self.speech_rng.choice(pool)].text if pool else None

    def _speak_tome(self, t: Turn) -> None:
        st, chapter = self.state, self.chapter.id
        slots = self.ledger.slots(chapter)
        rhetoric, frame, slot = self._choose_rhetoric(t, len(slots))
        if rhetoric in ("corrupt_npc", "bend"):      # these arm next turn's corruption (M8); nothing is said
            rhetoric = "hold"
        addressed = t.question is not None
        if not addressed and st.turn - self.last_unprompted < UNPROMPTED_GAP:
            rhetoric = "hold"
        asks, deflect = t.question or ("none", None)
        plan = self.ledger.plan(
            chapter, rhetoric, asks, deflect,
            volunteer=slots[slot].id if slot is not None and slot < len(slots) else None,
            suggestion=self._suggestion_for(frame) if rhetoric == "suggest" else None,
            probe=self.speech_rng.choice(sorted(self.ledger.probes)) if rhetoric == "probe" and self.ledger.probes
            else None)
        t.plan = plan
        acts = list(t.tome_acts)
        if plan.deflection:
            acts.append(SpeechAct("deflection", self.speech_rng.choice(
                MOOD_DEFLECTIONS if plan.deflection == "mood" else DICE_DEFLECTIONS)))
        acts += self.ledger.acts(plan)
        t.tome_acts = acts
        if not acts:
            if t.variant == "tome":
                t.variant = "tome_silent"           # spoken to, it says nothing: never narrated as answering
            return
        for cond, action in ((plan.answers or plan.vague_answer, "tome.answer"), (plan.deflection, "tome.deflect"),
                             (plan.volunteer, "tome.disclose"), (plan.probe, "tome.probe")):
            if cond:
                self.checks.note("tome", action)
        if plan.suggestion:
            rec = checks.d20(self.speech_rng, self.checks, "tome", "tome.suggest", "guile", self.tome_mods()["guile"],
                             10 + self._pmod("insight"))
            if not rec.degree.succeeded:
                t.asides.append(SUGGEST_TELL)          # a failed manipulation leaves a true tell
        if not addressed and (plan.volunteer or plan.suggestion or plan.probe):
            self.last_unprompted = st.turn
        self.ledger.disclose(plan.told)
        t.style = self._style()
        lines, caution = [a.text for a in acts], self._caution(chapter, set(plan.told))
        text = self.voice.tome_say(TomeSpeechRequest(lines=lines, style=t.style, caution=caution))
        self._audit(text, narration=False, exempt=lines + ([caution] if caution else []))
        color = PALETTE["lumen"]
        t.npc_lines.append(Block(role="npc", speaker="The Tome", data={"npc": "tome"},
                                 spans=[Span("THE TOME: ", color=color, glow=0.8),
                                        Span(f"“{text}”", color=color, glow=0.45)]))

    # ================================================================ rules: actions
    def _act(self, t: Turn) -> None:
        s, st = t.sensory, self.state
        ob = self._target_obstacle(s)
        intent = s.intent

        if intent is Intent.USE and self._use_item(t, ob):
            return
        if (s.target == "tome" or s.addressed_tome) and self._at_tome(t):
            return
        if intent is Intent.FIGHT and s.target in st.npcs_present:
            self._attack_npc(t, s.target)
            return
        if ob is not None and intent in ENGAGING:
            self._engage(t, ob, s)
            return
        if intent is Intent.SEARCH:
            self._search(t)
        elif intent is Intent.LOOK:
            self._look(t)
        elif intent is Intent.REST:
            self._rest(t)
        elif intent is Intent.ADVANCE:
            self._advance(t)
        elif intent is Intent.FLEE:
            self._flee(t, ob)
        elif intent is Intent.TALK:
            if not st.npcs_present:
                self.checks.note("player", "player.talk_alone")
                t.facts.append("The player speaks, but no one is here to answer.")
                t.variant = "alone"
            elif s.target in st.npcs_present or len(st.npcs_present) == 1:
                t.variant = "npc"
                self._social_roll(t)
                if t.outcome is not None:
                    t.variant = None
            else:
                self.checks.note("player", "player.chat_npc")
                t.facts.append("The player speaks to no one in particular.")
        else:
            self.checks.note("player", "player.idle")
            t.facts.append("The player acts; nothing in the scene resists or changes.")

    def _engage(self, t: Turn, ob: Obstacle, s: SensoryState) -> None:
        st = self.state
        approach = s.effective_approach
        if approach is Approach.NONE:
            approach = Approach.GRIT
        bonus = self._help(t, approach)
        upper = ob.id in st.twisted and self._upper_hand()  # opposing the Tome's own adaptation
        rec = self._check("player.engage", approach, self._dc(), intent=s.intent.value, bonus=bonus,
                          positional=True, upper_hand=upper)
        ex = resolve_exchange(ob, approach, as_pool(rec), dark=st.lumen.dark)
        t.obstacle, t.outcome = ob, rec.degree
        t.facts.append(f"The player tackles {ob.name} with {approach.value}. Result: {rec.degree.value}.")
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
            t.event(f"OVERCOME: {ob.name}")
            st.character.relieve_stress(1)
            if not self._blocking():
                t.facts.append(f"The way on is open. {self.chapter.exit_text}")
        self._bite(t, ex.bite_harm, ex.bite_stress, ob.name)
        if not ex.cleared and rec.degree is Outcome.FAIL and not (ob.harm or ob.stress):
            t.facts.append("The failed attempt is loud, and the noise carries deep into the station.")
            self._tick_worst_clock(t, 1)
        self._tension_from(rec.degree)

    def _help(self, t: Turn, approach: Approach) -> int:
        """An NPC whose dominant plot is alliance tries to help: its own check (d20 vs 10) gives +2 on success.
        Schemers and the attacked don't help."""
        for nid in self.state.npcs_present:
            if nid != t.sensory.target and self._is_ally(nid):
                attr = approach.attribute.value
                rec = checks.d20(self.rng, self.checks, nid, "npc.help", attr, self._nmod(nid, attr), 10)
                name = self.scenario.npcs[nid].name.split()[0]
                if rec.degree.succeeded:
                    t.facts.append(f"{name} helps the player.")
                    return 2
                t.facts.append(f"{name} tries to help, but it comes to nothing.")
                return 0
        return 0

    def _search(self, t: Turn) -> None:
        st = self.state
        bonus = self._help(t, Approach.INSIGHT)
        rec = self._check("player.search", Approach.INSIGHT, self._dc(), intent="search", bonus=bonus,
                          positional=True)
        t.outcome = rec.degree
        if rec.degree is Outcome.FAIL:
            t.facts.append("The search turns up nothing but the feeling of being watched.")
            self._bite(t, 0, 1, "the dark")
        else:
            clue = self._next_clue(lore=True)
            if clue:
                t.clue = clue
            found_light = rec.degree.succeeded and self.chapter.light and not st.light_taken
            if not clue and not found_light:
                t.variant = "empty"             # nothing new to find: never narrated as a find
            if found_light:
                st.light_taken = True
                if st.lumen.restore(self.chapter.light):
                    t.facts.append("The player finds a light source and feeds it to the Tome.")
                    t.asides.append("You find a light source. The Tome drinks it in and burns brighter.")
                    t.effects.append({"kind": "flare", "strength": 0.8})
            if rec.degree is Outcome.MIXED:
                t.facts.append("It takes too long; something in the station notices the delay.")
                self._tick_worst_clock(t, 1)
        self._tension_from(rec.degree)

    def _look(self, t: Turn) -> None:
        """Looking never hurts. A good look adds one true detail: what an obstacle would give to."""
        t.describe = True
        t.facts.append("The player takes in their surroundings.")
        rec = self._check("player.look", Approach.INSIGHT, 10 + self.chapter.act)
        active = [o for o in self._active() if o.pips]
        if rec.degree.succeeded and active:
            ob = active[0]
            t.asides.append(f"You study {ob.name}. It looks like it would give to {APPROACH_WORDS[ob.pips[0]]}.")

    def _rest(self, t: Turn) -> None:
        st, ch = self.state, self.chapter
        active = self._active()
        if active:
            self.checks.note("player", "player.rest_refused")
            t.facts.append(f"There is no rest with {active[0].name} here.")
            t.obstacle, t.variant = active[0], "denied"
            self._bite(t, 0, 1, active[0].name)
            return
        if st.rest_used:
            self.checks.note("player", "player.rest_refused")
            t.facts.append("The player tries to rest again, but the station won't allow it twice.")
            t.variant = "spent"
            return
        st.rest_used = True
        rec = self._check("player.rest", Attribute.RESOLVE, 8 + ch.act)
        if rec.degree.succeeded:
            heal = relief = 3 if ch.respite else 2
        elif rec.degree is Outcome.MIXED:
            heal = relief = 1
        else:
            heal, relief = 1, 0
            t.facts.append("Sleep will not come; every sound in the dark pulls the player back awake.")
        st.character.heal(heal)
        st.character.relieve_stress(relief)
        st.tension = max(0.0, st.tension - 0.2)
        st.since_respite = 0
        t.facts.append("The player rests, briefly and uneasily.")

    def _advance(self, t: Turn) -> None:
        st = self.state
        blocking = self._blocking()
        if blocking:
            self.checks.note("player", "player.advance_blocked")
            t.facts.append(f"{blocking[0].name} still blocks the way on.")
            t.obstacle, t.variant = blocking[0], "blocked"
            return
        hazards = self._active()
        if hazards:
            ob = hazards[0]
            rec = self._check("player.advance_hazard", Approach.GUILE, self._dc(), intent="advance", positional=True)
            t.obstacle, t.outcome = ob, rec.degree
            t.facts.append(f"The player dashes past {ob.name}. Result: {rec.degree.value}.")
            if rec.degree is Outcome.FAIL:
                self._bite(t, ob.harm, ob.stress, ob.name)
            elif rec.degree is Outcome.MIXED and (ob.harm or ob.stress):
                self._bite(t, 0, 1, ob.name)
            self._tension_from(rec.degree)
            if st.character.dead:
                return
        else:
            self.checks.note("player", "player.advance_open")
        if st.deck.at_end:
            st.over, st.won, t.ending = True, True, "won"
            t.facts.append("The player escapes the sector.")
            return
        st.deck.advance()
        self._enter_chapter(t, st.deck.current)

    def _flee(self, t: Turn, ob: Obstacle | None) -> None:
        st = self.state
        if ob is None:
            self.checks.note("player", "player.flee_nothing")
            t.facts.append("The player backs away, but nothing pursues.")
            return
        rec = self._check("player.flee", Approach.GUILE, self._dc(), intent="flee")
        t.obstacle, t.outcome = ob, rec.degree
        if rec.degree is not Outcome.FAIL:
            st.position_bonus = 2
            t.facts.append(f"The player breaks contact with {ob.name} and finds a better position.")
            if not ob.blocking:
                st.obstacles.remove(ob)
                t.facts.append(f"{ob.name} is left behind.")
        if rec.degree is Outcome.FAIL:
            self._bite(t, ob.harm, ob.stress, ob.name)
        elif rec.degree is Outcome.MIXED:
            self._bite(t, 0, 1, ob.name)
        self._tension_from(rec.degree)

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
            self.checks.note("player", "player.feed_light")
            t.us["gift"] = 1.0
            st_char.spend_item(item)
            t.variant = "light" if st.lumen.restore(item.light) else "light_full"     # a full Tome burns no brighter
            t.facts.append(f"The player spends the {item.name}; its light pours into the Tome.")
            t.effects.append({"kind": "flare", "strength": 1.0})
            return True
        if "heal" in item.tags:
            st_char.spend_item(item)
            rec = self._check("player.heal", Approach.INSIGHT, 8, intent="heal")
            if rec.degree.succeeded:
                st_char.heal(2)
                st_char.relieve_stress(1)
            else:
                st_char.heal(1)
                t.facts.append("Shaking hands make a poor job of it; it only half helps.")
            t.facts.append(f"The player uses the {item.name} on their wounds.")
            t.variant = "heal"
            return True
        return False

    def _social_roll(self, t: Turn) -> None:
        """Words aimed at an NPC are checked against that NPC; the result changes what the NPC *hears*."""
        s, st = t.sensory, self.state
        nid = s.target if s.target in st.npcs_present else st.npcs_present[0]
        if s.deception > 0.5:
            rec = self._check("player.lie_npc", Approach.GUILE, 10 + self._nmod(nid, "insight"), intent="talk")
            t.outcome = rec.degree
            if rec.degree.succeeded:
                s.deception *= 0.25
                t.facts.append("The lie lands. They believe it, for now.")
            elif rec.degree is Outcome.FAIL:
                s.deception = min(1.0, s.deception + 0.3)
                s.exposure = min(1.0, s.exposure + 0.2)
                t.facts.append("The lie is transparent. They see straight through it.")
        elif s.aggression > 0.6:
            rec = self._check("player.threaten_npc", Approach.GRIT, 10 + self._nmod(nid, "resolve"), intent="talk")
            t.outcome = rec.degree
            if rec.degree.succeeded:
                s.leverage = min(1.0, s.leverage + 0.4)
                t.facts.append("The threat lands. They are afraid of the player now.")
            elif rec.degree is Outcome.FAIL:
                s.rapport = max(-1.0, s.rapport - 0.3)
                t.facts.append("The threat rings hollow and only earns contempt.")
        elif s.leverage > 0.5:
            rec = self._check("player.bargain_npc", Approach.GUILE, 10 + self._nmod(nid, "insight"), intent="talk")
            s.leverage = min(1.0, s.leverage * _SWAY[rec.degree])
        elif s.exposure > 0.5:
            rec = self._check("player.probe_npc", Approach.INSIGHT, 10 + self._nmod(nid, "guile"), intent="talk")
            s.exposure = min(1.0, s.exposure * _SWAY[rec.degree])
        elif s.rapport > 0.3:
            rec = self._check("player.appeal_npc", Approach.INSIGHT, 10 + self._nmod(nid, "insight"), intent="talk")
            s.rapport = min(1.0, s.rapport * _SWAY[rec.degree])
        else:
            self.checks.note("player", "player.chat_npc")

    def _attack_npc(self, t: Turn, nid: str) -> None:
        name = self.scenario.npcs[nid].name
        rec = self._check("player.attack_npc", Approach.GRIT, 10 + self._nmod(nid, "guile"), intent="fight")
        if rec.degree.succeeded:
            t.facts.append(f"The player's blow lands before {name} twists away and puts distance between them.")
            t.asides.append(f"Your blow lands before {name.split()[0]} gets clear.")
        else:
            t.facts.append(f"The player attacks {name}, who twists away and puts distance between them.")
        t.variant, t.npc_name = "npc_attack", name
        t.flee_npc = nid

    def _at_tome(self, t: Turn) -> bool:
        """The player turns on the Tome itself. Returns False to treat the act as an ordinary one.

        A precise question is answered whatever else the words do: alongside a strike, a smother, a lie or a threat
        (a question about its dim light is not an attempt to dim it). A Tome whose light is out has no glow to break or
        to hide, and is never narrated as if it had."""
        s, st = t.sensory, self.state
        tm = self.tome_mods()
        upper = self._upper_hand()
        asked = s.asks not in ("none", "vague", "unclear")
        if s.intent in (Intent.FIGHT, Intent.FORCE):
            rec = self._check("player.strike_tome", Approach.GRIT, 10 + tm["grit"], intent="fight", upper_hand=upper)
            if rec.degree.succeeded:
                t.us["harm"] = 1.0
            if rec.degree.succeeded and st.lumen.dice > 0:
                st.lumen.burn(1)
                t.facts.append("The player strikes the Tome. One of its lights cracks and goes out.")
                t.effects.append({"kind": "dim", "strength": 0.5})
                t.variant = "tome_struck"
                self._tome_says(t, "struck_out" if st.lumen.dark else "struck")
            elif rec.degree.succeeded:
                t.facts.append("The player's blow lands on the dark Tome; every one of its lights is already out.")
                t.variant = "tome_struck_dark"
                self._tome_says(t, "unmoved")
            else:
                t.facts.append("The player strikes the Tome; the blow skids off its black face.")
                t.variant = "tome_unmoved"
                self._tome_says(t, "unmoved")
            if asked:
                t.question = (s.asks, question_kind(t.text))
            return True
        if smothers(t.text) or (not asked and s.intent is Intent.SNEAK):
            if st.lumen.dark:
                self.checks.note("player", "player.idle")
                t.facts.append("The player covers the Tome, but it has no glow left to hide.")
                t.variant = "tome_smother_dark"
            elif self._check("player.smother_tome", Approach.GUILE, 10 + tm["insight"], intent="sneak",
                             upper_hand=upper).degree.succeeded:
                st.shadow_bonus = 2
                t.us["harm"] = 0.5
                t.facts.append("The player smothers the Tome's glow; for a moment they are hard to see.")
                t.variant = "tome_smothered"
            else:
                t.facts.append("The player tries to smother the Tome's glow, but it bleeds through their fingers.")
                t.variant = "tome_smother_fail"
                self._tome_says(t, "smother_fail")
            if asked:
                t.question = (s.asks, question_kind(t.text))
            return True
        if s.intent is Intent.TALK or asked:
            t.variant = "tome"
            if s.deception > 0.5:
                rec = self._check("player.lie_tome", Approach.GUILE, 10 + tm["insight"], intent="talk",
                                  upper_hand=upper)
                if not rec.degree.succeeded:
                    self._tome_says(t, "lie_noticed")
                    t.us["lie"] = 1.0
            elif s.aggression > 0.6:
                rec = self._check("player.threaten_tome", Approach.GRIT, 10 + tm["resolve"], intent="talk",
                                  upper_hand=upper)
                self._tome_says(t, "cowed" if rec.degree.succeeded else "scorn")
                t.us["defiance"] = 1.0
            else:
                self.checks.note("player", "player.ask_tome")
                deflect = question_kind(t.text) if s.asks != "none" or "?" in t.text else None
                t.question = (s.asks, deflect)
                if s.asks == "none" and not deflect:
                    self._tome_says(t, "answer")
                elif s.asks == "unclear" and not deflect:
                    self._tome_says(t, "unclear")
            if asked and t.question is None:
                t.question = (s.asks, question_kind(t.text))
            t.facts.append("The player speaks to the Tome, and it answers in its own voice (shown separately).")
            return True
        return False

    # ================================================================ limbic: NPCs listen
    def _listen(self, t: Turn) -> None:
        st, s = self.state, t.sensory
        if not st.npcs_present:
            return
        if t.variant == "lost_turn":            # panic took the moment: the player said nothing for anyone to hear
            s = SensoryState()
        addressed = s.target if s.target in st.npcs_present else None
        if addressed is None and s.intent is Intent.TALK and s.target != "tome" and len(st.npcs_present) == 1:
            addressed = st.npcs_present[0]      # words said to the Tome by name are not the lone NPC's to answer
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
        self._tactic_check(t, nid, spec.tactics[decision.tactic].id)

    def _tactic_check(self, t: Turn, nid: str, tactic: str) -> None:
        """What an NPC does with its words is an action too: misdirection, pressure and offers are checked."""
        action = f"npc.{tactic}"
        name = self.scenario.npcs[nid].name.split()[0]
        if action not in NPC_TACTIC_CHECKS:
            self.checks.note(nid, action)
            return
        attr, against = NPC_TACTIC_CHECKS[action]
        brain = self.brains[nid]
        plot = next((p for p in brain.spec.plots if p.id == next(
            (tc.plot for tc in brain.spec.tactics if tc.id == tactic), "")), None)
        if plot is not None and plot.role == "extract":
            t.us["loss"] = max(t.us.get("loss", 0.0), 1.0)      # someone here wants to take the Tome away
        rec = checks.d20(self.rng, self.checks, nid, action, attr, self._nmod(nid, attr), 10 + self._pmod(against))
        if tactic in ("probe", "interrogate", "menace"):
            if rec.degree.succeeded:
                t.facts.append(f"{name}'s pressure gets under the player's skin.")
                self._bite(t, 0, 1, name)
        elif not rec.degree.succeeded:
            t.asides.append(TELLS[tactic].format(name=name))

    def _npc_name_by_numeric(self, numeric_id: int) -> str | None:
        for spec in self.scenario.npcs.values():
            if spec.numeric_id == numeric_id:
                return spec.name
        return None

    # ================================================================ pressure
    def _bite(self, t: Turn, harm: int, stress: int, source: str) -> None:
        ch = self.state.character
        if harm > 0:
            res = ch.take_harm(harm, self.rng, self.checks)
            t.facts.append(f"{source} hurts the player.")
            t.event(f"-{res.hull_lost} HULL" if res.hull_lost else "HULL BREACHED")
            t.effects.append({"kind": "shudder", "strength": min(1.0, 0.3 * harm)})
            self.state.tension = min(1.0, self.state.tension + 0.05 * harm)
            if res.last_light is not None:
                if res.dead:
                    t.facts.append("The player's last light fails. They die.")
                else:
                    t.facts.append("Past their limit, the player stays standing through sheer grit, but bleeding.")
                    t.asides.append("Past your limit, you stay on your feet through grit alone. You are bleeding.")
        if stress > 0:
            if ch.add_stress(stress):
                t.panic_due = True
            t.event(f"+{stress} STRESS ({ch.stress})")

    def _panic(self, t: Turn) -> None:
        ch = self.state.character
        check = panic_check(ch, self.rng, self.checks)
        if check.held:
            return
        eff = check.effect
        ch.relieve_stress(2)  # the fear discharges into the panic itself
        t.event(f"PANIC: {eff.name.upper()}")
        t.effects.append({"kind": "panic", "strength": 1.0})
        t.facts.append(f"The player panics: {eff.text}")
        t.asides.append(f"Panic takes you. {eff.text}")
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
                dropped = ch.inventory.pop(checks.d6_table(self.rng, self.checks, "passive.drop_item",
                                                           len(ch.inventory)))
                t.asides.append(f"You lose the {dropped.name} somewhere in the dark.")
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
        if roll is not None:
            self.checks.add(CheckRecord("world", "passive.lumen", "d6", "pool", roll.before, 0, roll.lost,
                                        total=roll.lost))
        if roll is not None and roll.lost:
            t.event(f"-{roll.lost} LUMEN ({lum.dice}/{LUMEN_MAX})")
            t.effects.append({"kind": "dim", "strength": 0.4 * roll.lost})
            t.facts.append("The Tome's light gutters and shrinks.")
            t.asides.append("The Tome's light gutters and shrinks.")
            if lum.dice <= 2 and self.state.character.add_stress(1):
                t.panic_due = True
        if lum.dark and not was_dark:
            t.asides.append("The Tome's light goes out. The dark has the sector now.")
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
        beat, probs = self._choose_beat(t, allowed)
        t.beat, t.beat_probs = beat, probs
        if beat in TOME_BEAT_CHECKS:
            attr, against = TOME_BEAT_CHECKS[beat]
            rec = self._tome_check(f"tome.{beat.value}", attr, 10 + self._pmod(against))
            if rec.degree is Outcome.FAIL:
                self._beat_fizzles(t, beat)
                return
        else:
            self.checks.note("tome", f"tome.{beat.value}")
        if beat is Beat.CALM:
            st.tension = max(0.0, st.tension - 0.08)
        elif beat is Beat.ESCALATE:
            self._tick_worst_clock(t, 2 if st.lumen.dark else 1)
            st.tension = min(1.0, st.tension + 0.15)
            st.since_threat = 0
            t.effects.append({"kind": "shudder", "strength": 0.3})
        elif beat is Beat.CLUE:
            t.clue = t.clue or self._next_clue()
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
        elif beat is Beat.TWIST:
            self._twist_beat(t)
        elif beat is Beat.ADVANCE:
            t.beat_fact = self.chapter.exit_text

    def _before_act(self, t: Turn) -> None:
        """Hook: runs after the senses read the input and before the rules resolve it."""

    def _debug_extra(self) -> dict:
        """Hook: extra developer telemetry for the debug object."""
        return {}

    def _choose_beat(self, t: Turn, allowed: set[Beat]) -> tuple[Beat, dict[str, float]]:
        """The Director samples a beat from its pacing distribution (the Tome's network overrides this)."""
        return self.director.choose(self._features(t), self.rng, allowed)

    def _beat_fizzles(self, t: Turn, beat: Beat) -> None:
        """The Tome reached for something and missed. The player only ever sees a quieter moment."""
        st = self.state
        t.event(f"{beat.value} fizzles")
        if beat is Beat.HAZARD:
            # it shows its hand: the hazard comes, but the player saw it coming
            st.since_threat = 0
            self._hazard_beat(t)
            st.position_bonus = 2
            t.facts.append("The player sees the danger coming a moment before it arrives.")
            return
        t.beat = Beat.CALM

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
        t.event(f"{clock.name} {clock.filled}/{clock.segments}")
        if fired:
            if clock.consequence == "lumen":
                lost = self.state.lumen.burn(2)
                t.beat_fact = (f"{clock.name}: the station's power lurches and the Tome's light is dragged down with "
                               "it." if lost else f"{clock.name}: the station's power lurches, and the dark deepens.")
                t.event(f"-{lost} LUMEN")
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
                pool = sorted(pool)
                ob = self._add_obstacle(pool[checks.d6_table(self.rng, self.checks, "passive.hazard_pick", len(pool))])
                t.beat_fact = ob.text
                return
        self._bite(t, 0, 1, "the station")

    def _twist_beat(self, t: Turn) -> None:
        st = self.state
        st.tension = min(1.0, st.tension + 0.12)
        active = [o for o in self._active() if o.id not in st.twisted and len(o.pips) <= 3]
        if active:
            ob = active[0]
            st.twisted.append(ob.id)
            pip = TWIST_PIPS[checks.d6_table(self.rng, self.checks, "passive.twist_pip", 6)]
            ob.pips.append(pip)
            t.beat_fact = f"Something in {ob.name} shifts, as if it has learned from watching you."
            t.event(f"{ob.name} adapts +{pip}")
        elif st.lumen.dice > 3:
            st.lumen.burn(1)
            t.beat_fact = "One of the Tome's lights gutters out."
            t.event(f"-1 LUMEN ({st.lumen.dice}/{LUMEN_MAX})")
            t.effects.append({"kind": "dim", "strength": 0.4})
        else:
            self._tick_worst_clock(t, 1)

    # ================================================================ scene management
    def _enter_chapter(self, t: Turn, chapter_id: str, gossip_ok: bool = True) -> None:
        st = self.state
        ch = self.scenario.chapters[chapter_id]
        if gossip_ok and ch.act > self.scenario.chapters[st.deck.order[st.deck.index - 1]].act:
            healed, relieved = st.character.heal(1), st.character.relieve_stress(1)
            if healed or relieved:
                t.facts.append("Between decks, the player finds a moment to steady themselves.")
                t.asides.append("Between decks, you find a moment to steady yourself.")
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

    def _offscreen(self, t: Turn) -> None:
        """Between chapters, NPCs who know the player talk about them behind their back."""
        met = [n for n in self.state.npcs_met if n in self.brains]
        for a in met:
            for b in sorted(self.brains):
                if a == b or checks.d6_table(self.rng, self.checks, "passive.gossip", 6) < 3:
                    continue  # they only talk on a 4+
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
        targets: dict[str, list[str]] = {"tome": list(TOME_ALIASES)}
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
            beat_fact=t.beat_fact, clue=t.clue, ending=t.ending, register=(t.style or self._style()).register,
        )
        narration = self.voice.narrate(req)
        world = [x for x in (ch.text if req.entering_chapter else None, t.clue, t.defeated, t.beat_fact) if x]
        self._audit(" ".join([narration, *t.asides]), narration=True, exempt=world)
        self._remember(narration)
        blocks: list[Block] = [text_block("echo", f"> {t.text}", effect="none")]
        if t.entering:
            blocks.append(text_block("title", t.entering.title, glow=0.6, effect="ripple"))
        blocks.append(text_block("narration", narration))
        blocks += t.npc_lines
        blocks += self._aside_blocks(t)
        if not st.over:
            blocks.append(self._prompt_block())
        return self._frame(blocks, t)

    @staticmethod
    def _aside_blocks(t: Turn) -> list[Block]:
        return [text_block("narration", a, glow=0.5) for a in dict.fromkeys(t.asides)]

    def _prompt_block(self) -> Block:
        spans = [Span(f"{o.name[0].upper()}{o.name[1:]}.   ", color=PALETTE["dread"], glow=0.4)
                 for o in self._active()]
        spans.append(Span("What do you do?", color=PALETTE["system"], glow=0.2))
        return Block(role="prompt", spans=spans)

    def _frame(self, blocks: list[Block], t: Turn) -> TomeFrame:
        st = self.state
        ch = st.character
        self.seq += 1
        # Protocol v2: the status bar carries game state only. No dice, checks, pips, clock counters, item
        # counts or attribute numbers; no trace of the Tome's mood or opinion.
        status = {
            "hull": [ch.hull, ch.hull_max], "stress": ch.stress, "lumen": [st.lumen.dice, LUMEN_MAX],
            "chapter": {"id": self.chapter.id, "title": self.chapter.title, "act": self.chapter.act,
                        "index": st.deck.index, "total": len(st.deck.order)},
            "won": st.won,
            "scene": {
                "inventory": [i.name for i in ch.inventory if i.uses != 0],
                "npcs": [{"id": n, "name": self.scenario.npcs[n].name, "color": self.scenario.npcs[n].color}
                         for n in st.npcs_present],
            },
        }
        mood = {
            "lumen": round(st.lumen.fraction, 3), "stress": round(ch.stress / 10, 3), "dark": st.lumen.dark,
            "pulse": "heartbeat" if ch.stress >= PANIC_THRESHOLD else "breath",
        }
        debug = {  # dev builds only (VOIDLIT_DEV=1); never part of a release frame
            "sensory": t.sensory.to_dict() if t.sensory else None,
            "beat": t.beat.value, "beat_probs": t.beat_probs,
            "decisions": {k: v.to_dict() for k, v in t.decisions.items()}, "gossip": t.gossip,
            "notes": t.notes, "tension": round(st.tension, 3), "tome_mods": self.tome_mods(),
            "checks": [r.to_dict() for r in self.checks.records[t.checks_from:]], "us": dict(t.us),
            "tome_speech": None if t.plan is None and not t.tome_acts else {
                "plan": asdict(t.plan) if t.plan else None,
                "acts": [[a.kind, a.text, a.fact_id] for a in t.tome_acts],
                "style": asdict(t.style) if t.style else None},
            **self._debug_extra(),
        }
        return TomeFrame(seq=self.seq, blocks=blocks, status=status, mood=mood, effects=t.effects,
                         over=st.over, debug=debug)

    # ================================================================ persistence
    # A save is a directory: campaign.json (the game), brains/<npc>.pt (each NPC's brain) and, in the network
    # engine, the Tome's own brain, every file stamped with the same save id. ``save`` writes it all or nothing,
    # ``_snapshot`` holds the same in memory, and ``load`` takes a save all or nothing.
    def _voice_rng(self) -> random.Random | None:
        return getattr(getattr(self.voice, "fallback", self.voice), "rng", None)

    def _campaign(self) -> dict:
        """The game's own state, as campaign.json holds it (beside its save id)."""
        voice_rng = self._voice_rng()
        return {
            "version": SAVE_VERSION, "scenario": self.scenario.id, "player_name": self.player_name,
            "seq": self.seq, "state": self.state.to_dict(), "rng": _rng_state(self.rng),
            "voice_rng": None if voice_rng is None else _rng_state(voice_rng),
            "ledger": self.ledger.state_dict(), "last_unprompted": self.last_unprompted,
            "speech_rng": _rng_state(self.speech_rng),
            "director_gen": bytes(self.director.gen.get_state().tolist()).hex(),   # its spiking encoder's noise
        }

    def save(self, directory: str | Path) -> Path:
        """Write a save, all or nothing: the whole slot is written beside the old one and only then takes its place.
        Raises SaveError, with the slot exactly as it was."""
        d = Path(directory)
        try:
            write_slot(d, self._write_save)
        except Exception as e:
            raise SaveError(f"{d.name}: could not save: {e} (the slot is as it was)") from e
        return d

    def _write_save(self, d: Path, save_id: str) -> None:
        """Write every file of a save into the new directory ``d``, each stamped with ``save_id``."""
        import torch

        (d / "campaign.json").write_text(json.dumps({**self._campaign(), "save_id": save_id}, indent=1))
        (d / "brains").mkdir()
        for nid, brain in self.brains.items():
            torch.save({"save_id": save_id, "brain": brain.state_dict()}, d / "brains" / f"{nid}.pt")

    def load(self, directory: str | Path) -> None:
        """Load a save, all or nothing. Every file is read and checked before anything changes, and should taking it
        in still fail, the game takes back what it held. Raises SaveError, with the game exactly as it was; should
        even that fail, the run is closed."""
        d = Path(directory)
        try:
            saved = self._read_save(d)
        except Exception as e:          # missing, unreadable, damaged or another game's: nothing has changed yet
            raise SaveError(f"{d.name}: {e}") from e
        held = self._snapshot()
        try:
            self._take_save(saved)
        except Exception as e:
            try:
                self._take_save(held)
            except Exception as undo:   # a bug, not a bad save: neither game can go on safely
                self.state.over = True
                raise SaveError(f"{d.name}: {e}; the game could not be restored either ({undo}), so this run is "
                                "closed") from undo
            raise SaveError(f"{d.name}: {e} (nothing was loaded)") from e

    def _snapshot(self) -> dict:
        """What a save made now would hold, in memory: taking it back undoes a load that failed halfway."""
        return {"campaign": json.loads(json.dumps(self._campaign())),       # exactly as campaign.json reads back
                "npcs": {nid: brain.state_dict() for nid, brain in self.brains.items()}}

    def _read_save(self, d: Path) -> dict:
        """Read a save and check that its files belong together and that all of it fits this game, changing
        nothing."""
        import torch

        campaign = json.loads((d / "campaign.json").read_text())
        if (not isinstance(campaign, dict) or campaign.get("version") != SAVE_VERSION
                or campaign.get("scenario") != self.scenario.id):
            raise ValueError("save does not match this scenario or engine version")
        saved = {"save_id": campaign.pop("save_id", None), "campaign": campaign, "npcs": {}}
        check_types("save id", saved["save_id"], str)
        for nid in self.brains:
            # tensors and plain containers only: a save file cannot run code
            blob = torch.load(d / "brains" / f"{nid}.pt", weights_only=True)
            check_types(f"brains/{nid}.pt", blob, {"save_id": str, "brain": dict})
            check_stamp(f"brains/{nid}.pt", blob["save_id"], saved["save_id"])
            saved["npcs"][nid] = blob["brain"]
        self._check_save(saved)
        return saved

    def _check_save(self, saved: dict) -> None:
        import torch

        blob, sc = saved["campaign"], self.scenario
        check_types("campaign", blob, CAMPAIGN_TYPES)
        state = GameState.from_dict(blob["state"])
        deck = state.deck
        if (not set(deck.order) <= set(sc.chapters) or not 0 <= deck.index < len(deck.order)
                or not set(state.npcs_present) | set(state.npcs_met) <= set(sc.npcs)
                or not set(blob["ledger"]["disclosed"]) <= set(self.ledger.facts)):
            raise ValueError("the campaign's game does not fit this scenario")
        if (blob["voice_rng"] is None) != (self._voice_rng() is None):
            raise ValueError("the campaign's voice does not fit this game's")
        for s in (blob["rng"], blob["speech_rng"]) + ((blob["voice_rng"],) if blob["voice_rng"] is not None else ()):
            _set_rng(random.Random(), s)            # each a state a generator takes
        torch.Generator().set_state(torch.tensor(list(bytes.fromhex(blob["director_gen"])), dtype=torch.uint8))
        for nid, brain in self.brains.items():
            brain.check_state_dict(saved["npcs"][nid])

    def _take_save(self, saved: dict) -> None:
        """Load what ``_read_save`` accepted, or what ``_snapshot`` kept."""
        import torch

        blob = saved["campaign"]
        for nid, brain in self.brains.items():
            brain.load_state_dict(saved["npcs"][nid])
        self.director.gen.set_state(torch.tensor(list(bytes.fromhex(blob["director_gen"])), dtype=torch.uint8))
        self.player_name, self.seq = blob["player_name"], blob["seq"]
        self.state = GameState.from_dict(blob["state"])
        _set_rng(self.rng, blob["rng"])
        voice_rng = self._voice_rng()
        if voice_rng is not None:
            _set_rng(voice_rng, blob["voice_rng"])
        self.ledger.load_state_dict(blob["ledger"])
        self.last_unprompted = blob["last_unprompted"]
        _set_rng(self.speech_rng, blob["speech_rng"])
