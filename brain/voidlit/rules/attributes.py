"""The four attributes every character has, and how a score becomes a d20 modifier."""

from __future__ import annotations

from enum import Enum

MAX_SCORE = 4


class Attribute(str, Enum):
    GRIT = "grit"        # force, violence, endurance; the backup health under Last Light
    GUILE = "guile"      # stealth, trickery, precision, speed, lies
    INSIGHT = "insight"  # analysis, systems, perception, reading people, noticing what goes unsaid
    RESOLVE = "resolve"  # will: holds against panic, intimidation and corruption


ALL_ATTRIBUTES = (Attribute.GRIT, Attribute.GUILE, Attribute.INSIGHT, Attribute.RESOLVE)


def mod(score: int) -> int:
    """d20 modifier for a 0-4 score: 0 -> -2, 1 -> 0, 2 -> +2, 3 -> +4, 4 -> +6."""
    return 2 * score - 2


def score_of(modifier: int) -> int:
    """The score a modifier stands for, for d6 tables (used for the Tome, whose modifiers come from its network)."""
    return max(0, min(MAX_SCORE, (modifier + 2) // 2))
