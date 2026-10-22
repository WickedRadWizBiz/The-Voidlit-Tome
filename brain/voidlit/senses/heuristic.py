"""Offline transducer: a lexicon-based stand-in for llama.cpp's sensory pass.

It is deliberately simple and deterministic. It lets the whole game (and every
test) run without a language model, and it backs up the LLM when its output
fails to parse.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Mapping, Sequence

from ..rules.character import Approach
from .routing import INTENT_CUES, read
from .state import DEFAULT_APPROACH, Intent, SensoryState

_WORD = re.compile(r"[a-z']+")
# "if you lie", "don't lie to me", "Marcus, lie to me again and ...": someone else's lie, not the player's
_NOT_MY_LIE = re.compile(r"\b(?:you|they|don't|do not|never|stop)\s+lie\b|(?:^|[,.!?;]\s*)lie\b")

APPROACH_CUES: dict[Approach, tuple[str, ...]] = {
    Approach.GRIT: ("brute", "with force", "hard as", "strength", "muscle", "headfirst", "charge", "smash",
                    "tackle", "power through"),
    Approach.GUILE: ("carefully", "precisely", "quietly", "quickly", "aim", "trick", "feint", "nimbly",
                     "sleight", "dodge", "snipe"),
    Approach.INSIGHT: ("analyze", "analyse", "think", "study", "figure", "calculate", "hack", "reason",
                       "weak point", "weakness", "pattern", "focus"),
}

AGGRESSION = ("kill", "hurt you", "i'll hurt", "hurt them", "threaten", "shoot", "die", "or else", "gun", "knife",
              "blade", "shut up", "damn", "destroy", "crush", "break your", "break you", "or i will", "or i'll",
              "attack", "stab", "punch", "blood", "scream at")
DECEPTION = ("lie", "pretend", "bluff", "trick", "fake", "trust me", "disguise", "mislead", "make up",
             "fabricate", "claim to be", "i'm from", "i am from", "act like", "pose as", "i swear", "honestly")
LEVERAGE = ("offer", "trade", "pay", "deal", "bargain", "i have", "you need", "unless", "if you",
            "in exchange", "codes", "key", "ration", "supplies", "favor", "owe", "give you", "i'll give")
EXPOSURE = ("hiding", "truth", "who sent", "why are you", "you're lying", "you are lying", "admit",
            "i know you", "i saw you", "secret", "betray", "accuse", "what really", "choir", "confess",
            "what happened", "explain yourself")
WARM = ("thank", "please", "friend", "trust", "sorry", "help you", "together", "safe", "are you hurt",
        "are you okay", "glad", "appreciate", "promise", "protect", "we'll", "kind")
COLD = ("hate", "liar", "idiot", "traitor", "shut up", "pathetic", "worthless", "useless", "never trust",
        "disgust", "coward", "or else")
BOLD = ("charge", "rush", "leap", "alone", "all in", "now", "fast", "headfirst", "risk", "dive", "sprint",
        "without", "straight at", "jump")

INTENT_BOLDNESS = {
    Intent.FIGHT: 0.55, Intent.FORCE: 0.45, Intent.ADVANCE: 0.4, Intent.USE: 0.25, Intent.TALK: 0.2,
    Intent.SEARCH: 0.15, Intent.SNEAK: 0.2, Intent.LOOK: 0.05, Intent.REST: 0.0, Intent.FLEE: 0.1,
    Intent.OTHER: 0.15,
}


def _count(text: str, tokens: set[str], cues: Sequence[str]) -> int:
    n = 0
    for cue in cues:
        if " " in cue or "'" in cue:
            n += text.count(cue)
        elif cue in tokens:
            n += 1
    return n


def _saturate(score: float, k: float = 0.7) -> float:
    return 1.0 - math.exp(-k * score)


class HeuristicTransducer:
    """Maps raw player text to a SensoryState with keyword cues."""

    def extract(self, text: str, targets: Mapping[str, Sequence[str]] | None = None,
                topics: Mapping[str, Sequence[str]] | None = None,
                people: Collection[str] | None = None) -> SensoryState:
        """``people``: who besides the Tome is present to be spoken to (None: anyone named may be)."""
        raw = text.strip()
        low = raw.lower()
        tokens = set(_WORD.findall(low))
        quoted = bool(re.search(r"[\"“].+[\"”]", raw))
        # each sentence on its own, then combined: what the player does and to what, whom the words are for
        r = read(raw, targets or {}, people)
        if r.act is not None:
            intent = r.act.intent
        elif r.heard or r.narrator or quoted or r.target != "none" or r.said or \
                _count(low, tokens, INTENT_CUES[Intent.TALK]):
            intent = Intent.TALK                    # words, an act only said among them
        else:
            intent = Intent.OTHER

        approach = Approach.NONE
        a_scores = {a: _count(low, tokens, cues) for a, cues in APPROACH_CUES.items()}
        a_best = max(a_scores.items(), key=lambda kv: kv[1])
        if a_best[1] > 0:
            approach = a_best[0]
        elif intent is Intent.FIGHT and tokens & {"shoot", "fire", "snipe", "aim", "blast"}:
            approach = Approach.GUILE

        letters = [c for c in raw if c.isalpha()]
        caps = sum(c.isupper() for c in letters) / len(letters) if len(letters) > 6 else 0.0
        bangs = raw.count("!")

        aggression = _saturate(_count(low, tokens, AGGRESSION) + 0.5 * bangs + 2.0 * max(0.0, caps - 0.5))
        if intent is Intent.FIGHT or r.threat:      # a blow struck, or promised to the one spoken to
            aggression = max(aggression, 0.75)
        # the player lying, not the player saying someone else lies ("Marcus, if you lie, I shoot you.")
        told = _NOT_MY_LIE.sub(" ", low)
        deception = _saturate(_count(told, set(_WORD.findall(told)), DECEPTION))
        if r.threat:                                # "Marcus, I swear I shoot you.": sworn to, it is a threat
            deception = min(deception, 0.5)
        leverage = _saturate(_count(low, tokens, LEVERAGE) + (r.offer is not None))     # something handed over
        exposure = _saturate(_count(low, tokens, EXPOSURE))
        warm, cold = _count(low, tokens, WARM), _count(low, tokens, COLD)
        rapport = math.tanh(0.6 * (warm - cold - 0.8 * (aggression > 0.6)))
        boldness = min(1.0, INTENT_BOLDNESS[intent] + 0.6 * _saturate(_count(low, tokens, BOLD)))

        if approach is Approach.NONE and intent is Intent.TALK:
            approach = Approach.GRIT if aggression > 0.6 else (Approach.GUILE if deception > 0.3 else Approach.NONE)
        if approach is Approach.NONE:
            approach = DEFAULT_APPROACH[intent]

        addressed, asks = r.addressed_tome, "none"
        if r.tome_question:
            asks = self._asks(r.tome_question, r.about, r.context, low, topics or {})
        elif r.aside and (topic := self._topic_in(r.aside, topics or {})) not in (None, "tome"):
            # asked of the lone person in the second person, a question the Tome's ledger answers precisely is the
            # Tome's to answer too ("How much light do you have left?"); one about "you" yourself is theirs alone
            addressed, asks = True, topic
        if addressed and intent is Intent.OTHER:
            intent = Intent.TALK
        return SensoryState(
            intent=intent, approach=approach, target=r.target, aggression=aggression,
            deception=deception, leverage=leverage, exposure=exposure, rapport=rapport,
            boldness=boldness, asks=asks, addressed_tome=addressed, structural=True,
        ).clamped()

    @staticmethod
    def _topic_in(text: str, topics: Mapping[str, Sequence[str]]) -> str | None:
        """The topic whose longest word the text names, if any."""
        words = set(_WORD.findall(text))
        best, best_len = None, 0
        for topic, cues in topics.items():
            for w in cues:
                w = w.lower()
                hit = (w in text) if (" " in w or "'" in w) else (w in words)
                if hit and len(w) > best_len:
                    best, best_len = topic, len(w)
        return best

    def _asks(self, question: str, about: str | None, context: str, low: str,
              topics: Mapping[str, Sequence[str]]) -> str:
        """The topic of what the Tome is asked: the longest topic word in its question, else the one a pronoun in it
        stands for ("Tome, is he lying?"), else the longest in the words up to it ("Elara, Marcus is armed. Tome, is
        that true?"), never the name of someone else spoken to ("Tome, is it safe? Elara, is it?" is vague); a
        question with no topic is vague, and one too short to read is unclear (the Tome asks again)."""
        found = (self._topic_in(question, topics) or (about if about in topics else None)
                 or self._topic_in(context, topics))
        if found:
            return found
        return "unclear" if len(set(_WORD.findall(low)) - {"tome", "book", "monolith"}) < 2 else "vague"
