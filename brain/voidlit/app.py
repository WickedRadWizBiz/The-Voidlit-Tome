"""Wiring: build an engine with or without a language model behind it."""

from __future__ import annotations

import logging

from .tome.content import load_scenario
from .tome.engine import TomeEngine
from .voice.llama_bridge import LlamaServer, LlamaTransducer, LlamaVoice
from .voice.narrator import OfflineVoice

log = logging.getLogger(__name__)


def build_engine(seed: int = 0, background: str | None = None, name: str = "Wanderer",
                 scenario: str = "hollow_lantern", llama_url: str | None = None) -> tuple[TomeEngine, bool]:
    """Returns (engine, llm_online). With no reachable llama-server the game runs fully offline."""
    sc = load_scenario(scenario)
    offline = OfflineVoice(sc.narration, seed=seed)
    if llama_url:
        server = LlamaServer(llama_url)
        if server.healthy():
            return TomeEngine(sc, transducer=LlamaTransducer(server), voice=LlamaVoice(server, offline),
                              seed=seed, background=background, player_name=name), True
        log.warning("llama-server at %s is not reachable; running offline", llama_url)
    return TomeEngine(sc, voice=offline, seed=seed, background=background, player_name=name), False
