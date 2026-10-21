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
