"""The TOME's rules, as written in TOME: Core Rules (Draft 0.2): d6 pools, four stats, Grit and Wounds,
Stress and Panic, the Lumen, a hidden deck of Leaves ending in the Crucible, Pages, Hunger and Legacy."""

from .bearer import KINDS, STATS, Bearer, Item, Kind, Stat, Wound, parse_kit
from .campaign import Campaign, Opening, make_bearer, opening_roll
from .dice import PoolRoll, Result, push, roll_2d6, roll_pool
from .events import Event, Log
from .oracle import Answer, ask, likelihood_of
from .pages import find_page, use_page
from .trial import ACCEPTS, LUMEN_MAX, PAGE_IDS, Chapter, Leaf, Trial, build_deck

__all__ = [
    "KINDS", "STATS", "Bearer", "Item", "Kind", "Stat", "Wound", "parse_kit", "Campaign", "Opening",
    "make_bearer", "opening_roll", "PoolRoll", "Result", "push", "roll_2d6", "roll_pool", "Event", "Log",
    "Answer", "ask", "likelihood_of", "find_page", "use_page", "ACCEPTS", "LUMEN_MAX", "PAGE_IDS", "Chapter",
    "Leaf", "Trial", "build_deck",
]
