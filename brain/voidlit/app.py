"""Wiring: build an engine, with or without a local language model behind it."""

from __future__ import annotations

from pathlib import Path

from .tome.content import Codex, all_places
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


def build_engine(seed: int = 0, class_id: str | None = None, name: str = "Bearer", place: str | None = None,
                 voice_llm: ChatModel | None = None, senses_llm: ChatModel | None = None, *,
                 lower: str | None = None, length: str = "short", codex: str | Path | None = None,
                 ) -> tuple[TomeEngine, bool]:
    """Returns (engine, llm_in_use). Without models the game runs fully offline.

    ``codex`` is a Forge content folder or export bundle; the default is the snapshot shipped with the app.
    """
    content = Codex.load(codex)
    banks: dict[str, list[str]] = {}
    for p in all_places(content).values():
        banks.update(p.narration)
    offline = OfflineVoice(banks, seed=seed)
    transducer = LocalTransducer(senses_llm) if senses_llm is not None else None
    voice = LocalVoice(voice_llm, offline) if voice_llm is not None else offline
    engine = TomeEngine(content, transducer=transducer, voice=voice, seed=seed, class_id=class_id,
                        player_name=name, place=place, lower=lower, length=length)
    return engine, voice_llm is not None or senses_llm is not None
