"""Offline transducer: a lexicon-based stand-in for llama.cpp's sensory pass.

It is deliberately simple and deterministic. It lets the whole game (and every
test) run without a language model, and it backs up the LLM when its output
fails to parse.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import NamedTuple

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
# "Dear Tome, ...", "Excuse me, Tome, ..."), the few that may come before a name said last ("..., O Tome?"), and the
# question words that may follow a bare name ("Tome what does Elara want?").
VOCATIVE_LEADS = (r"(?:hey|hi|oi|oh|o|ah|so|well|now|then|there|listen|look|okay|ok|right|alright|please|and|but|say|"
                  r"tell me|dear|my|excuse me|pardon me|pardon|sorry)")
TRAILING_LEADS = r"(?:hey|oh|o|my|dear)"
WH_WORDS = r"(?:what|where|who|whose|why|how|which|when)"
# Speaking to someone in the third person: "I ask the Tome who Marcus is", "I turn to the Tome and ask ...".
ADDRESS_VERBS = r"(?:ask|asks|question|tell|tells|say to|whisper to|turn to|call to|beg|plead with)"
ASKING = ("ask", "asks", "question")
_SENTENCE = re.compile(r"[^.!?]*(?:[.!?]+|$)")

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


# The player telling what they do, right after naming someone ("Tome! I smash you.", "Marcus - I attack him."): an act,
# not words to the one named.
_ACT_CUES = sorted({c for i, cues in INTENT_CUES.items() if i is not Intent.TALK for c in cues}, key=len, reverse=True)
ACT_OPENING = re.compile(rf"\W*(?:and\s+|then\s+)?i\s+(?:\w+ly\s+|then\s+|just\s+)?"
                         rf"(?:{'|'.join(map(re.escape, _ACT_CUES))})\b")


def sentences(low: str) -> list[tuple[int, str]]:
    """The sentences of a line, each with where it starts (its closing punctuation kept)."""
    return [(m.start(), m.group()) for m in _SENTENCE.finditer(low) if re.search(r"[a-z]", m.group())]


def is_question(text: str) -> bool:
    """A question mark, or a question word opening the words, or opening them after a short opener ("Listen, what are
    you?")."""
    body = text.lstrip("\"“' ")
    if "?" in body:
        return True
    for part in (body, body.split(",", 1)[1] if "," in body[:24] else ""):
        words = _WORD.findall(part)
        if words and words[0] in QUESTION_WORDS:
            return True
    return False


def _names(targets: Mapping[str, Sequence[str]]) -> dict[str, str]:
    return {tid: "|".join(map(re.escape, sorted({tid, *(a.lower() for a in aliases)}, key=len, reverse=True)))
            for tid, aliases in targets.items()}


def _address(start: int, sent: str, names: Mapping[str, str]) -> tuple[str, int, bool, bool] | None:
    """Whom one sentence names as the one spoken to, if anyone: (target, where the words to them start, named at its
    head, asked in the third person)."""
    lead = len(sent) - len(sent.lstrip("\"“”' "))
    for tid, name in names.items():
        # an opener takes the bare name ("Hey Tome"); "Look, the Tome!" points at it instead
        m = re.match(rf"(?:(?:{VOCATIVE_LEADS}\W+)+(?:{name})|(?:the\s+)?(?:{name}))"
                     rf"(?:\s*(?:[,:;!?.]|-+|–|—)|(?=\s+{WH_WORDS}\b)|\s*[\"”']*$)", sent[lead:])
        if m:
            return tid, start + lead + m.end(), True, False
    for tid, name in names.items():
        if re.search(rf",\s*(?:{TRAILING_LEADS}\s+)?(?:the\s+)?(?:{name})\s*[?!.]*[\"”']*\s*$", sent):
            return tid, start, False, False
    # in the third person, the first one named after a verb of speaking ("I tell Marcus to ask the Tome": Marcus)
    hits = [(m, tid) for tid, name in names.items() if (m := re.search(
        rf"\b({ADDRESS_VERBS})\s+(?:the\s+)?(?:{name})\b(?:\s*[,:]?\s*and\s+(ask|say|tell|whisper)\b)?", sent))]
    if hits:
        m, tid = min(hits, key=lambda h: h[0].start())
        return tid, start + m.end(), False, m.group(1) in ASKING or m.group(2) == "ask"
    return None


class Address(NamedTuple):
    """The one a line names as spoken to, and where the words to them are."""

    target: str | None = None
    start: int = 0              # where the words to them start
    end: int = 0                # where they end: the next one spoken to hears the rest
    words: bool = False         # named at the head of the line, and not followed by the player telling what they do
    asked: bool = False         # something is asked of them, with or without a question ("I ask the Tome about ...")


def spoken_to(low: str, targets: Mapping[str, Sequence[str]]) -> Address:
    """The one named as spoken to, if anyone. In any sentence, a name is spoken to at its head, after any opener and
    before a comma, colon, dash or question word ("Hey Tome, ...", "Tome: ...", "Tome what ...?"), last after a comma
    ("..., Tome?"), or in the third person ("I ask the Tome who ..."). It outranks a name mentioned in passing; of
    several, the first one a question is put to wins ("Elara, Marcus is armed. Tome, who is Marcus?")."""
    sents, names = sentences(low), _names(targets)
    found = [(i, a) for i, (start, sent) in enumerate(sents) if (a := _address(start, sent, names))]
    heard = []                          # each one spoken to hears the sentences from theirs up to the next one's
    for k, (i, (tid, rest, head, asking)) in enumerate(found):
        end = sents[found[k + 1][0]][0] if k + 1 < len(found) else len(low)
        asked = asking or any(is_question(low[max(rest, s):min(end, s + len(sent))]) for s, sent in sents[i:])
        heard.append(Address(tid, rest, end, head and i == 0 and not ACT_OPENING.match(low[rest:]), asked))
    return next((a for a in heard if a.asked), heard[0] if heard else Address())


class HeuristicTransducer:
    """Maps raw player text to a SensoryState with keyword cues."""

    def extract(self, text: str, targets: Mapping[str, Sequence[str]] | None = None,
                topics: Mapping[str, Sequence[str]] | None = None) -> SensoryState:
        raw = text.strip()
        low = raw.lower()
        tokens = set(_WORD.findall(low))
        quoted = bool(re.search(r"[\"“].+[\"”]", raw))
        targets = targets or {}

        scores = self._scores(low, quoted)
        # words are for the name they are spoken to ("Hey Tome, what does Elara want?"), whoever else is named
        to = spoken_to(low, targets)
        target = to.target or self._match_target(low, targets)
        if target != "none" and scores[Intent.TALK] > 0:
            scores[Intent.TALK] += 1.0
        if to.words:
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

        question = "?" in low or to.asked or any(is_question(sent) for _, sent in sentences(low))
        # a question to no one in particular is put to the narrator, and the narrator is the Tome, whatever else the
        # line does ("I take cover behind a crate. Where is the lifeboat?")
        addressed = target == "tome" or (to.target is None and self._to_the_narrator(low, targets))
        if addressed and intent is Intent.OTHER:
            intent = Intent.TALK
        asks = self._asks(low, self._questions(low, to), topics or {}) if addressed and question else "none"

        return SensoryState(
            intent=intent, approach=approach, target=target, aggression=aggression,
            deception=deception, leverage=leverage, exposure=exposure, rapport=rapport,
            boldness=boldness, asks=asks, addressed_tome=addressed,
        ).clamped()

    @staticmethod
    def _scores(low: str, quoted: bool) -> dict[Intent, float]:
        tokens = set(_WORD.findall(low))
        scores = {intent: float(_count(low, tokens, cues)) for intent, cues in INTENT_CUES.items()}
        if quoted:
            scores[Intent.TALK] += 2.0
        return scores

    def _to_the_narrator(self, low: str, targets: Mapping[str, Sequence[str]]) -> bool:
        """Whether a sentence of the line is a question to no one in particular: it names no one and nothing but the
        Tome, and is neither a look around ("where am I?") nor words said aloud for anyone to hear."""
        for _, sent in sentences(low):
            if not is_question(sent) or self._match_target(sent, targets) not in ("none", "tome"):
                continue
            best = max(self._scores(sent, bool(re.search(r"[\"“”]", sent))).items(), key=lambda kv: kv[1])
            own = best[0] if best[1] > 0 else Intent.OTHER
            if own is Intent.OTHER or (own is Intent.TALK and set(_WORD.findall(sent)) & {"you", "your"}):
                return True
        return False

    @staticmethod
    def _questions(low: str, to: Address) -> str:
        """The words of the line that ask something: the questions in the words the one spoken to hears (all of those
        words when none is a question: "I ask the Tome about the lifeboat"), else every question in the line."""
        if to.target is None:
            return " ".join(sent for _, sent in sentences(low) if is_question(sent))
        heard = [p for s, sent in sentences(low) if (p := low[max(to.start, s):min(to.end, s + len(sent))]).strip()]
        return " ".join([p for p in heard if is_question(p)] or heard)

    @staticmethod
    def _asks(low: str, questions: str, topics: Mapping[str, Sequence[str]]) -> str:
        """The topic of a question put to the Tome: the longest topic word in the words that ask, else anywhere in the
        line, wins; a question with no topic is vague, and one too short to read is unclear (the Tome asks again)."""
        for text in (questions, low):
            words = set(_WORD.findall(text))
            best, best_len = None, 0
            for topic, cues in topics.items():
                for w in cues:
                    w = w.lower()
                    hit = (w in text) if (" " in w or "'" in w) else (w in words)
                    if hit and len(w) > best_len:
                        best, best_len = topic, len(w)
            if best:
                return best
        return "unclear" if len(set(_WORD.findall(low)) - {"tome", "book", "monolith"}) < 2 else "vague"

    @staticmethod
    def _match_target(low: str, targets: Mapping[str, Sequence[str]]) -> str:
        """The first one or thing named in passing (of two names that start together, the longer: "the warden turret",
        not the warden). The Tome, always at hand, is the target only when nothing else is named."""
        best: tuple[tuple[bool, int, int], str] | None = None
        for tid, aliases in targets.items():
            for alias in {tid, *aliases}:
                a = alias.lower()
                m = re.search(rf"(?<![a-z']){re.escape(a)}(?![a-z])", low)
                if m and (best is None or (tid == "tome", m.start(), -len(a)) < best[0]):
                    best = ((tid == "tome", m.start(), -len(a)), tid)
        return best[1] if best else "none"
