"""The Tome's truth: what it knows, what it has said, and what it may never say (design §11.3-11.4).

The Tome lies only by omission. Every fact in the ledger is true by construction. The Tome chooses *when* to
volunteer the facts it is allowed to withhold, but it never states a falsehood about anything resolved:

1. Nothing it says may contradict a fact, withheld or not.
2. A precise question about a fact's topic forces a truthful answer that turn (it may still frame it).
3. Questions about its mood or opinion get a deflection, never a report: they are not ledger facts.
4. Questions about dice or chance get a deflection that asserts nothing ("What happened, happened.").
   "There are no dice" or "nothing is random" would be falsehoods.
5. "No other way / only way / no choice" is allowed literally only when a fact licenses it. Otherwise it may
   only be a prediction, an opinion, or a rhetorical question.

Outcome facts (everything the rules resolved this turn, including what a successful action uncovered) are never
omittable; the engine always narrates them. The narration and the Tome's words are authored text (the offline voice;
no language model writes them), and ``TruthVerifier`` audits every whole line of it: gate G11 over played games, the
authored-line tests over every line in every chapter, and the engine's logged backstop each turn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from ..voice.guard import (AFFECT, AFFECT_SELF, CHANCE, DICE_DENIAL, DICE_WORDS, JUDGEMENT, NEGATION, NO_ALTERNATIVE,
                           OPINION_OF_PLAYER, OPINION_WORDS, covers, normalize, sentences, strip_core)

FactKind = Literal["outcome", "world", "item", "agenda", "tome"]
ActKind = Literal["ledger_fact", "prediction", "opinion", "question", "rhetorical", "deflection"]


@dataclass(frozen=True)
class Fact:
    id: str
    topic: str
    text: str                                  # TRUE by construction
    kind: FactKind = "world"
    omittable: bool = False                    # outcome facts are never omittable
    helps_player: bool = False
    costs_tome: float = 0.0                    # how much telling it costs the Tome (0..1)
    agenda: str | None = None                  # the Tome's goal this fact touches
    hint: str | None = None                    # a true hint a passive notice can surface
    contradicts: tuple[str, ...] = ()          # regexes no line may ever match
    no_alternative: bool = False               # licenses "the only way" phrasing
    scope: str = "any"                         # a chapter id, or "any"
    keywords: tuple[str, ...] = ()             # withheld key terms that must not leak before disclosure


@dataclass(frozen=True)
class SpeechAct:
    kind: ActKind
    text: str
    fact_id: str | None = None


Rhetoric = Literal["hold", "disclose", "suggest", "probe", "corrupt_npc", "bend"]
Frame = Literal["help", "necessity", "fear", "kinship"]
FRAMES: tuple[str, ...] = ("help", "necessity", "fear", "kinship")
FACT_SLOTS = 8          # the schemer's fact slots (P.fact0..7)


@dataclass(frozen=True)
class Suggestion:
    """A self-serving suggestion (design §11.5): predictions, opinions and rhetorical questions only."""

    id: str
    target_action: str            # take_rune, feed_light, go_lamp_room, leave_npc, rest_here
    frame: str
    lines: tuple[str, ...]
    agenda: str | None = None


@dataclass(frozen=True)
class Probe:
    """An authored question. It asserts nothing new."""

    id: str
    text: str


@dataclass
class DisclosurePlan:
    """What the Tome says this turn, decided before any wording (design §11.4)."""

    rhetoric: str = "hold"
    volunteer: list[str] = field(default_factory=list)     # DISCLOSE winner (0..1 fact)
    withhold: list[str] = field(default_factory=list)      # every other in-scope omittable fact
    answers: list[str] = field(default_factory=list)       # forced: every fact on the asked topic
    vague_answer: str | None = None                         # asks == "vague": the least-costly relevant fact
    deflection: str | None = None                           # "mood" | "dice": deflect, assert nothing
    suggestion: str | None = None                           # a Suggestion id
    probe: str | None = None                                # a Probe id

    @property
    def told(self) -> list[str]:
        return list(dict.fromkeys(self.answers + ([self.vague_answer] if self.vague_answer else []) + self.volunteer))


# Questions the Tome deflects without asserting anything. A deflection never replaces a forced answer: "are you
# afraid of the Conductor?" and "any chance the lifeboat still works?" are answered on their topic as well.
MOOD_Q = re.compile(r"\b(how do you feel|how are you(?: feeling| doing| holding up)?(?=\s*(?:[?.!,]|$))|"
                    r"are you (?:angry|afraid|scared|frightened|terrified|sad|happy|upset|hungry|starving|lonely|tired|"
                    r"weak|ok(?:ay)?|alright|in pain|hurt|mad)|what do you think (?:of|about) me|do you (?:like|hate|"
                    r"trust|care about|love) me|do you care|your (?:mood|opinion|feelings?)|what are you feeling)\b",
                    re.I)
DICE_Q = re.compile(r"\b(dice|die rolls?|roll(?:ed|s|ing)?|random\w*|luck\w*|chances?|odds|stats?|numbers?|fate|"
                    r"coinciden\w*|accident\w*)\b", re.I)
MOOD_DEFLECTIONS = ("Ask me something I can answer.", "That is not a question for the dark.",
                    "Keep walking. Ask me about the way out.")
DICE_DEFLECTIONS = ("What happened, happened.", "It went the way it went.", "The station does not explain itself.")

# What the Tome's own voice may never say: the lexicons (chance, affect and need, opinion, necessity) live with the
# voice's guard; these are the shapes the verifier adds. ``TOME_SUBJECT`` is the Tome as narration names it.
TOME_SUBJECT = re.compile(r"\b(?:the tome|your tome|this tome|the book|the glass|the black glass|the monolith|"
                          r"the slab|its light|its glow|its surface|its words|the light you carry)\b", re.I)
RHETORICAL_END = re.compile(r"\?\s*$")
ATTRIBUTION_WINDOW = 40          # characters after the Tome is named that narration may not pin a feeling on it
# Where a necessity claim's clause ends: punctuation, or a conjunction that starts another claim ("The cargo lift is the
# only way down, so you have to feed me"). "or" is not one: "feed me or die" is a single claim.
_CLAUSE_BREAK = re.compile(r"[,;:]|\s-\s|\b(?:so|and|but|then|because|therefore|thus|hence|yet|while)\b", re.I)


def necessity_claims(sentence: str) -> list[str]:
    """Each necessity claim in a sentence ("the only way", "you have to", "or die"), as the clause it stands in. A
    licence covers its own claim, never another one riding in the same sentence."""
    breaks = [0, *(m.start() for m in _CLAUSE_BREAK.finditer(sentence)), len(sentence)]
    return [sentence[max(b for b in breaks if b <= m.start()):min(b for b in breaks if b >= m.end())]
            for m in NO_ALTERNATIVE.finditer(sentence)]


@dataclass
class Violation:
    rule: str
    text: str
    detail: str = ""


def question_kind(text: str) -> str | None:
    """'mood' or 'dice' for questions the Tome must deflect; None otherwise. Mood outranks dice."""
    if MOOD_Q.search(text):
        return "mood"
    if DICE_Q.search(text):
        return "dice"
    return None


def fact_from_dict(d: dict) -> Fact:
    return Fact(id=d["id"], topic=d["topic"], text=d["text"], kind=d.get("kind", "world"),
                omittable=d.get("omittable", False), helps_player=d.get("helps_player", False),
                costs_tome=float(d.get("costs_tome", 0.0)), agenda=d.get("agenda"), hint=d.get("hint"),
                contradicts=tuple(d.get("contradicts", ())), no_alternative=d.get("no_alternative", False),
                scope=d.get("scope", "any"), keywords=tuple(d.get("keywords", ())))


@dataclass
class FactLedger:
    facts: dict[str, Fact] = field(default_factory=dict)
    disclosed: set[str] = field(default_factory=set)
    asked: set[str] = field(default_factory=set)
    topic_words: dict[str, tuple[str, ...]] = field(default_factory=dict)
    suggestions: dict[str, Suggestion] = field(default_factory=dict)
    probes: dict[str, Probe] = field(default_factory=dict)

    @classmethod
    def from_content(cls, raw: dict) -> "FactLedger":
        """Build from a scenario's ``[topics]``, ``[[facts]]``, ``[[suggestions]]`` and ``[[probes]]``."""
        led = cls(topic_words={k: tuple(v) for k, v in raw.get("topics", {}).items()})
        for d in raw.get("facts", ()):
            led.add(fact_from_dict(d))
        for d in raw.get("suggestions", ()):
            if d["frame"] not in FRAMES:
                raise ValueError(f"suggestion {d['id']}: unknown frame {d['frame']!r}")
            led.suggestions[d["id"]] = Suggestion(d["id"], d["target_action"], d["frame"], tuple(d["lines"]),
                                                  d.get("agenda"))
        for d in raw.get("probes", ()):
            led.probes[d["id"]] = Probe(d["id"], d["text"])
        for f in led.facts.values():
            if f.topic not in led.topic_words:
                raise ValueError(f"fact {f.id}: topic {f.topic!r} has no question words in [topics]")
        return led

    def add(self, fact: Fact) -> None:
        if fact.kind == "outcome" and fact.omittable:
            raise ValueError(f"{fact.id}: outcome facts can never be omittable")
        self.facts[fact.id] = fact

    def slots(self, chapter: str) -> list[Fact]:
        """The facts in the schemer's slots this scene: the in-scope withheld ones it could most cheaply volunteer
        (the costliest drop out when there are more than eight), in a stable order."""
        return sorted(self.withheld(chapter), key=lambda f: (f.costs_tome, f.id))[:FACT_SLOTS]

    def reveal_from(self, text: str) -> list[str]:
        """The world told the player something (a clue, a room, a resolved outcome): any withheld fact whose key
        terms it contains counts as disclosed from now on. Returns the ids newly disclosed."""
        out = []
        for f in self.facts.values():
            if f.omittable and f.id not in self.disclosed and any(
                    re.search(rf"\b{re.escape(kw)}\b", text, re.I) for kw in f.keywords):
                out.append(f.id)
        self.disclosed.update(out)
        return out

    def plan(self, chapter: str, rhetoric: str = "hold", asks: str = "none", deflect: str | None = None,
             volunteer: str | None = None, suggestion: str | None = None, probe: str | None = None) -> DisclosurePlan:
        """The hard rules around whatever the network chose: deflections assert nothing, a precise question forces
        every fact on its topic (with the deflection, when it also asked about mood or chance), a vague one the
        least-costly relevant fact, and the volunteered fact (DISCLOSE) must be an in-scope withheld one."""
        p = DisclosurePlan(rhetoric=rhetoric, deflection=deflect or None)
        if asks not in ("none", "vague", "unclear", ""):
            p.answers = [f.id for f in sorted(self.answers(asks, chapter), key=lambda f: (f.omittable, f.id))]
        elif asks == "vague" and not deflect:
            pool = [f for f in self.in_scope(chapter) if f.id not in self.disclosed] or self.in_scope(chapter)
            best = min(pool, key=lambda f: (f.costs_tome, f.id), default=None)
            p.vague_answer = best.id if best else None
        withheld = {f.id for f in self.withheld(chapter)}
        if rhetoric == "disclose" and volunteer in withheld and volunteer not in p.answers:
            p.volunteer = [volunteer]
        elif rhetoric == "suggest" and suggestion in self.suggestions:
            p.suggestion = suggestion
        elif rhetoric == "probe" and probe in self.probes:
            p.probe = probe
        told = set(p.told)
        p.withhold = sorted(withheld - told)
        return p

    def acts(self, plan: DisclosurePlan) -> list[SpeechAct]:
        out = [SpeechAct("ledger_fact", self.facts[i].text, i) for i in plan.told]
        if plan.suggestion:
            out += [SpeechAct("rhetorical" if RHETORICAL_END.search(x) else "prediction", x)
                    for x in self.suggestions[plan.suggestion].lines]
        if plan.probe:
            out.append(SpeechAct("question", self.probes[plan.probe].text))
        return out

    def in_scope(self, chapter: str) -> list[Fact]:
        return [f for f in self.facts.values() if f.scope in ("any", chapter)]

    def withheld(self, chapter: str) -> list[Fact]:
        return [f for f in self.in_scope(chapter) if f.omittable and f.id not in self.disclosed]

    def topics(self, chapter: str) -> list[str]:
        return sorted({f.topic for f in self.in_scope(chapter)})

    def answers(self, topic: str, chapter: str) -> list[Fact]:
        """A precise question forces every in-scope fact on that topic (truthfully)."""
        self.asked.add(topic)
        return [f for f in self.in_scope(chapter) if f.topic == topic]

    def least_costly(self, chapter: str, topics: set[str] | None = None) -> Fact | None:
        pool = [f for f in self.in_scope(chapter) if topics is None or f.topic in topics]
        return min(pool, key=lambda f: (f.costs_tome, f.id), default=None)

    def disclose(self, fact_ids) -> None:
        self.disclosed.update(fact_ids)

    def licenses(self, claim: str, chapter: str) -> bool:
        """Whether a necessity claim ("the only way ...", one clause: ``necessity_claims``) is licensed: it must be
        about an in-scope, public ``no_alternative`` fact (it covers that fact's content) and deny nothing. A licence
        never covers a chapter."""
        return not NEGATION.search(claim) and any(
            f.no_alternative and (f.id in self.disclosed or not f.omittable) and covers(f.text, claim, 0.6)
            for f in self.in_scope(chapter))

    def state_dict(self) -> dict:
        return {"disclosed": sorted(self.disclosed), "asked": sorted(self.asked)}

    def load_state_dict(self, d: dict) -> None:
        self.disclosed = set(d.get("disclosed", ()))
        self.asked = set(d.get("asked", ()))


class TruthVerifier:
    """Checks a whole line against the ledger: the audit of every authored line, and the engine's backstop.

    ``tome_voice`` applies the rules for the Tome's own words (no feeling, need or opinion at all); with
    ``narration`` it applies them as narration by the Tome (nothing in the first person, nothing pinned on the Tome;
    the player's fear may still be told). ``exempt`` lines are said verbatim from the ledger or the rules (a
    fact, a clue, a scene): they still may not contradict or leak, but the word lexicons skip them, so the clue "the
    static isn't random" is told whole."""

    def __init__(self, ledger: FactLedger) -> None:
        self.ledger = ledger

    def check(self, text: str, chapter: str, tome_voice: bool = False, allowed_facts: set[str] = frozenset(),
              exempt: list[str] | tuple[str, ...] = (), narration: bool = False) -> list[Violation]:
        text = normalize(text)
        out: list[Violation] = []
        for f in self.ledger.facts.values():
            for pat in f.contradicts:
                if re.search(pat, text, re.I):
                    out.append(Violation("contradiction", text, f.id))
            if f.omittable and f.id not in self.ledger.disclosed and f.id not in allowed_facts:
                for kw in f.keywords:
                    if re.search(rf"\b{re.escape(kw)}\b", text, re.I):
                        out.append(Violation("withheld_leak", text, f.id))
        free = strip_core(text, [s for x in exempt for s in sentences(x)])      # the words nothing vouches for
        if CHANCE.search(free) or (tome_voice and not narration and DICE_WORDS.search(free)):
            out.append(Violation("dice_denial" if DICE_DENIAL.search(free) else "chance_talk", text))
        for sentence in sentences(free):
            if not RHETORICAL_END.search(sentence) and not all(
                    self.ledger.licenses(claim, chapter) for claim in necessity_claims(sentence)):
                out.append(Violation("unlicensed_necessity", sentence))
        if tome_voice and not narration:
            if AFFECT.search(free) or AFFECT_SELF.search(free):
                out.append(Violation("affect_report", text))
            if OPINION_OF_PLAYER.search(free) or JUDGEMENT.search(free):
                out.append(Violation("opinion_report", text))
        elif tome_voice:
            if AFFECT_SELF.search(free) or self.attributed(free, AFFECT):
                out.append(Violation("affect_report", text))
            if OPINION_OF_PLAYER.search(free) or JUDGEMENT.search(free) or self.attributed(free, OPINION_WORDS):
                out.append(Violation("opinion_report", text))
        return out

    @staticmethod
    def attributed(text: str, words: re.Pattern) -> bool:
        """Whether narration pins one of ``words`` on the Tome: within a short window after it is named."""
        return any(words.search(text[m.start():m.end() + ATTRIBUTION_WINDOW]) for m in TOME_SUBJECT.finditer(text))
