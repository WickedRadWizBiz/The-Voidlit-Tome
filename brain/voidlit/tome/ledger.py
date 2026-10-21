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
omittable; the engine always narrates them. ``TruthVerifier`` checks every line the Tome or the narration speaks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

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


# questions the Tome deflects without asserting anything
MOOD_Q = re.compile(r"\b(how do you feel|are you (?:angry|afraid|sad|happy|upset|hungry)|what do you think of me|"
                    r"do you (?:like|hate|trust) me|your mood|your opinion)\b", re.I)
DICE_Q = re.compile(r"\b(dice|roll(?:ed)?|random|luck|chance|odds|stats?|numbers?)\b", re.I)
MOOD_DEFLECTIONS = ("Ask me something I can answer.", "That is not a question for the dark.",
                    "Keep walking. Ask me about the way out.")
DICE_DEFLECTIONS = ("What happened, happened.", "It went the way it went.", "The station does not explain itself.")

# things the Tome's own voice may never say
AFFECT_SELF = re.compile(r"\bI(?:'m| am| feel)\s+(?:so\s+|very\s+)?(?:afraid|angry|sad|happy|hungry|lonely|scared|"
                         r"furious|hurt|jealous|desperate)\b|\bI (?:hate|love|despise|adore) you\b", re.I)
OPINION_OF_PLAYER = re.compile(r"\byou (?:disgust|sicken|delight|please|disappoint) me\b", re.I)
DICE_DENIAL = re.compile(r"\b(?:there are no dice|nothing (?:here )?is random|no (?:dice|luck|chance) (?:at all|here))\b",
                         re.I)
NO_ALTERNATIVE = re.compile(r"\b(?:no other way|the only way|no choice|nothing else for it|no other choice)\b", re.I)
RHETORICAL_END = re.compile(r"\?\s*$")


@dataclass
class Violation:
    rule: str
    text: str
    detail: str = ""


@dataclass
class FactLedger:
    facts: dict[str, Fact] = field(default_factory=dict)
    disclosed: set[str] = field(default_factory=set)
    asked: set[str] = field(default_factory=set)

    def add(self, fact: Fact) -> None:
        if fact.kind == "outcome" and fact.omittable:
            raise ValueError(f"{fact.id}: outcome facts can never be omittable")
        self.facts[fact.id] = fact

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

    def licensed_no_alternative(self, chapter: str) -> bool:
        return any(f.no_alternative and f.id in self.disclosed | {x.id for x in self.in_scope(chapter)
                                                                     if not x.omittable}
                   for f in self.in_scope(chapter))

    def state_dict(self) -> dict:
        return {"disclosed": sorted(self.disclosed), "asked": sorted(self.asked)}

    def load_state_dict(self, d: dict) -> None:
        self.disclosed = set(d.get("disclosed", ()))
        self.asked = set(d.get("asked", ()))


class TruthVerifier:
    """Checks a line against the ledger. ``tome_voice`` applies the rules for the Tome's own words."""

    def __init__(self, ledger: FactLedger) -> None:
        self.ledger = ledger

    def check(self, text: str, chapter: str, tome_voice: bool = False,
              allowed_facts: set[str] = frozenset()) -> list[Violation]:
        out: list[Violation] = []
        for f in self.ledger.facts.values():
            for pat in f.contradicts:
                if re.search(pat, text, re.I):
                    out.append(Violation("contradiction", text, f.id))
            if f.omittable and f.id not in self.ledger.disclosed and f.id not in allowed_facts:
                for kw in f.keywords:
                    if re.search(rf"\b{re.escape(kw)}\b", text, re.I):
                        out.append(Violation("withheld_leak", text, f.id))
        if DICE_DENIAL.search(text):
            out.append(Violation("dice_denial", text))
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if NO_ALTERNATIVE.search(sentence) and not RHETORICAL_END.search(sentence) \
                    and not self.ledger.licensed_no_alternative(chapter):
                out.append(Violation("unlicensed_necessity", sentence))
        if tome_voice:
            if AFFECT_SELF.search(text):
                out.append(Violation("affect_report", text))
            if OPINION_OF_PLAYER.search(text):
                out.append(Violation("opinion_report", text))
        return out
