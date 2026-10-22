"""Wiring: build an engine, with or without a local language model behind it."""

from __future__ import annotations

from pathlib import Path

from .tome.content import load_scenario
from .tome.engine import TomeEngine
from .voice.llama_local import ChatModel, LocalLlama, LocalTransducer, LocalVoice
from .voice.narrator import OfflineVoice


def load_models(model: str | Path | None = None, senses_model: str | Path | None = None, *, n_ctx: int = 4096,
                n_gpu_layers: int = 0) -> tuple[ChatModel | None, ChatModel | None]:
    """Load local .gguf files into this process. Returns (voice model, senses model).

    The senses pass defaults to the voice model; a small, fast model can be given for it instead.
    Raises LocalModelError for anything that is not a usable local file.
    """
    voice = LocalLlama.load(model, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers) if model else None
    senses = LocalLlama.load(senses_model, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers) if senses_model else voice
    return voice, senses


def build_engine(seed: int = 0, background: str | None = None, name: str = "Wanderer",
                 scenario: str = "hollow_lantern", voice_llm: ChatModel | None = None,
                 senses_llm: ChatModel | None = None, brain: str = "net") -> tuple[TomeEngine, bool]:
    """Returns (engine, llm_in_use). Without models the game runs fully offline.

    ``brain="net"`` (the default) runs the Tome's one spiking network; ``"director"`` runs the classic rules engine
    whose beats come from the stand-alone Director SNN (faster to start, used for tests and comparisons)."""
    sc = load_scenario(scenario)
    offline = OfflineVoice(sc.narration, seed=seed)
    transducer = LocalTransducer(senses_llm) if senses_llm is not None else None
    names = [n for spec in sc.npcs.values() for n in (spec.name, *spec.aliases)]
    voice = LocalVoice(voice_llm, offline, names=names) if voice_llm is not None else offline
    if brain not in ("net", "director"):
        raise ValueError(f"unknown brain {brain!r}: use 'net' or 'director'")
    cls = TomeEngine
    if brain == "net":
        from .tome.net_engine import TomeNetEngine
        cls = TomeNetEngine
    engine = cls(sc, transducer=transducer, voice=voice, seed=seed, background=background, player_name=name)
    return engine, voice_llm is not None or senses_llm is not None
