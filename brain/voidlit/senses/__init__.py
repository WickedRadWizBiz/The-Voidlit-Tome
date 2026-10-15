"""Senses: turning player text into numbers the spiking brains can feel."""

from .grammar import SensoryParseError, build_sensory_gbnf, parse_sensory_json, sanitize_id
from .heuristic import HeuristicTransducer
from .state import DEFAULT_STAT, NPC_CHANNELS, Intent, SensoryState

__all__ = [
    "SensoryParseError", "build_sensory_gbnf", "parse_sensory_json", "sanitize_id",
    "HeuristicTransducer", "DEFAULT_STAT", "NPC_CHANNELS", "Intent", "SensoryState",
]
