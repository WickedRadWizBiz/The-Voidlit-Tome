"""Offline transducer: a lexicon-based stand-in for llama.cpp's sensory pass.

It is deliberately simple and deterministic. It lets the whole game (and every
test) run without a language model, and it backs up the LLM when its output
fails to parse.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

from ..rules.character import Approach
from .state import DEFAULT_APPROACH, Intent, SensoryState

_WORD = re.compile(r"[a-z']+")

# Intent cues. Multi-word cues are matched as substrings; single words by token.
INTENT_CUES: dict[Intent, tuple[str, ...]] = {
    Intent.FLEE: ("run away", "flee", "retreat", "fall back", "get out", "bolt", "escape back", "turn back"),
    Intent.FIGHT: ("attack", "shoot", "stab", "kill", "fight", "strike", "fire at", "punch", "slash", "swing at",
                   "hit it", "hit him", "hit her", "open fire", "gun down", "blast", "club"),
    Intent.FORCE: ("force", "pry", "break", "smash", "kick", "lift", "shove", "bash", "ram", "wrench",
                   "cut through", "tear", "climb", "haul", "brace", "jump"),
    Intent.SNEAK: ("sneak", "hide", "slip past", "creep", "tiptoe", "quietly", "pick the lock", "distract",
                   "steal", "pickpocket", "crawl", "stay low", "shadow", "unseen", "cover", "smother", "shade"),
    Intent.SEARCH: ("search", "examine", "inspect", "analyze", "analyse", "scan", "hack", "investigate",
                    "study", "decode", "check", "listen", "read", "rummage", "loot", "figure out", "trace"),
    Intent.USE: ("use", "activate", "apply", "inject", "light the", "turn on", "plug", "equip", "drink",
                 "flare", "medkit", "switch on", "power up"),
    Intent.REST: ("rest", "sleep", "camp", "catch my breath", "bandage", "recover", "patch up", "breathe",
                  "sit down", "take a moment"),
    Intent.ADVANCE: ("go on", "move on", "continue", "proceed", "head to", "onward", "next", "descend",
                     "go deeper", "go through", "enter", "keep going", "press on", "go forward", "leave"),
    Intent.TALK: ("say", "ask", "tell", "talk", "speak", "greet", "whisper", "persuade", "threaten",
                  "lie to", "promise", "offer", "convince", "beg", "answer", "reply", "call out", "hello"),
    Intent.LOOK: ("look", "look around", "where am i", "describe", "what do i see", "survey", "observe"),
}

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

QUESTION_WORDS = {"what", "where", "who", "why", "how", "which", "when", "is", "are", "can", "could", "do", "does",
                  "did", "will", "would", "should", "tell", "explain"}
# Words that may open an address before the name of the one spoken to ("Hey Tome, ...", "Tell me, Tome, ...",
# "So, Tome, ...", "O Tome, ..."), and the question words that may follow a bare name ("Tome what does Elara want?").
VOCATIVE_LEADS = r"(?:hey|hi|oi|oh|o|ah|so|well|now|then|listen|look|okay|ok|right|alright|please|and|but|say|tell me)"
WH_WORDS = r"(?:what|where|who|whose|why|how|which|when)"

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
                topics: Mapping[str, Sequence[str]] | None = None) -> SensoryState:
        raw = text.strip()
        low = raw.lower()
        tokens = set(_WORD.findall(low))
        quoted = bool(re.search(r"[\"“].+[\"”]", raw))

        scores = {intent: float(_count(low, tokens, cues)) for intent, cues in INTENT_CUES.items()}
        if quoted:
            scores[Intent.TALK] += 2.0
        # words are for the name they are spoken to ("Hey Tome, what does Elara want?"), however long another name is
        spoken, rest, head = self._spoken_to(low, targets or {})
        target = spoken or self._match_target(low, tokens, targets or {})
        if target != "none" and scores[Intent.TALK] > 0:
            scores[Intent.TALK] += 1.0
        if head:
            scores[Intent.TALK] += 3.0  # "Elara, ..." / "Tome: ...": words addressed to someone
        best = max(scores.items(), key=lambda kv: kv[1])
        intent = best[0] if best[1] > 0 else (Intent.TALK if target != "none" else Intent.OTHER)

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
        if intent is Intent.FIGHT:
            aggression = max(aggression, 0.75)
        deception = _saturate(_count(low, tokens, DECEPTION))
        leverage = _saturate(_count(low, tokens, LEVERAGE))
        exposure = _saturate(_count(low, tokens, EXPOSURE))
        warm, cold = _count(low, tokens, WARM), _count(low, tokens, COLD)
        rapport = math.tanh(0.6 * (warm - cold - 0.8 * (aggression > 0.6)))
        boldness = min(1.0, INTENT_BOLDNESS[intent] + 0.6 * _saturate(_count(low, tokens, BOLD)))

        if approach is Approach.NONE and intent is Intent.TALK:
            approach = Approach.GRIT if aggression > 0.6 else (Approach.GUILE if deception > 0.3 else Approach.NONE)
        if approach is Approach.NONE:
            approach = DEFAULT_APPROACH[intent]

        question = self._is_question(low, rest)
        # a question to no one in particular is put to the narrator, and the narrator is the Tome
        addressed = target == "tome" or (target == "none" and question and (
            intent is Intent.OTHER or (intent is Intent.TALK and bool(tokens & {"you", "your"}))))
        if addressed and intent is Intent.OTHER:
            intent = Intent.TALK
        asks = self._asks(low, tokens, topics or {}) if addressed and question else "none"

        return SensoryState(
            intent=intent, approach=approach, target=target, aggression=aggression,
            deception=deception, leverage=leverage, exposure=exposure, rapport=rapport,
            boldness=boldness, asks=asks, addressed_tome=addressed,
        ).clamped()

    @staticmethod
    def _is_question(low: str, start: int = 0) -> bool:
        """A question mark, or a question word opening the words after any address (from ``start``)."""
        body = low[start:].lstrip("\"“' ")
        if "," in body[:24]:                       # "Listen, what are you?" -> "what are you?"
            body = body.split(",", 1)[1]
        words = _WORD.findall(body)
        return "?" in low or bool(words and words[0] in QUESTION_WORDS)

    @staticmethod
    def _asks(low: str, tokens: set[str], topics: Mapping[str, Sequence[str]]) -> str:
        """The topic of a question put to the Tome: the longest matching topic word wins; a question with no topic
        is vague, and one too short to read is unclear (the Tome asks again)."""
        best, best_len = None, 0
        for topic, words in topics.items():
            for w in words:
                w = w.lower()
                hit = (w in low) if (" " in w or "'" in w) else (w in tokens)
                if hit and len(w) > best_len:
                    best, best_len = topic, len(w)
        if best:
            return best
        return "unclear" if len(tokens - {"tome", "book", "monolith"}) < 2 else "vague"

    @staticmethod
    def _spoken_to(low: str, targets: Mapping[str, Sequence[str]]) -> tuple[str | None, int, bool]:
        """(the target named as the one spoken to, where the words to them start, named at the head). The name is
        spoken to at the head of the line, after any opener and before a comma, colon, dash or question word ("Hey
        Tome, ...", "Tell me, Tome, ...", "Tome: ...", "Tome - ...", "Tome what ...?"), or after the last comma
        ("..., Tome?"). It outranks a longer name mentioned in passing."""
        body = low.rstrip()
        lead = len(body) - len(body.lstrip("\"“”' "))
        names = {tid: "|".join(map(re.escape, sorted({tid, *(a.lower() for a in aliases)}, key=len, reverse=True)))
                 for tid, aliases in targets.items()}
        for tid, name in names.items():
            m = re.match(rf"(?:{VOCATIVE_LEADS}\W+)*(?:the\s+)?(?:{name})"
                         rf"(?:\s*(?:[,:;!?.]|-+|–|—)|(?=\s+{WH_WORDS}\b)|\s*[\"”']*$)", body[lead:])
            if m:
                return tid, lead + m.end(), True
        for tid, name in names.items():
            if re.search(rf",\s*(?:{VOCATIVE_LEADS}\s+)?(?:the\s+)?(?:{name})\s*[?!.]*[\"”']*$", body):
                return tid, 0, False
        return None, 0, False

    @staticmethod
    def _match_target(low: str, tokens: set[str], targets: Mapping[str, Sequence[str]]) -> str:
        best, best_len = "none", 0
        for tid, aliases in targets.items():
            for alias in (tid, *aliases):
                a = alias.lower()
                hit = (a in tokens) if " " not in a else (a in low)
                if hit and len(a) > best_len:
                    best, best_len = tid, len(a)
        return best
