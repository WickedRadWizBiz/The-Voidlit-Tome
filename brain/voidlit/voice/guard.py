"""Output guard for the local LLM voice: as far as the player knows, there are no dice.

Any narration or dialogue that names a die, a roll, a check or a modifier, or (for narration) any digit at all,
is rejected and the offline voice speaks instead. The offline banks are the reference and never trip this guard.
"""

from __future__ import annotations

import re

_MECHANICS = re.compile(
    r"\b(dice|die rolls?|d4|d6|d8|d10|d12|d20|d100|modifiers?|pips?|hit points?|saving throws?|skill checks?|"
    r"ability checks?|stat checks?|natural (?:one|twenty|1|20)|nat ?(?:1|20)|you rolled|rolled an?|"
    r"roll(?:ed|s)? (?:a|an|the) (?:\d+|one|two|three|four|five|six|twenty)|difficulty class|game mechanics?)\b",
    re.IGNORECASE)
_DC = re.compile(r"\bDC\s*\d*\b")
_DIGIT = re.compile(r"\d")


def leaks_mechanics(text: str, allow_digits: bool = False) -> bool:
    if _MECHANICS.search(text) or _DC.search(text):
        return True
    return not allow_digits and bool(_DIGIT.search(text))


_STOP = frozenset("the a an and or but of to in on at by for with from into onto is are was were be been it its "
                  "this that "
                  "these those there here as so not no you your i my me we our they them their he she his her what "
                  "while only every one".split())
_CONTENT = re.compile(r"[a-z][a-z']+")


def covers(required: str, text: str, threshold: float = 0.5) -> bool:
    """Whether ``text`` still says ``required`` (a forced answer, a clue, a resolved outcome): at least half of its
    content words survive the paraphrase. Omission of what the rules force is a lie the pattern guard cannot see."""
    words = {w.strip("'") for w in _CONTENT.findall(required.lower()) if w not in _STOP and len(w) > 2}
    if not words:
        return True
    have = {w.strip("'") for w in _CONTENT.findall(text.lower())}
    return len(words & have) / len(words) >= threshold
