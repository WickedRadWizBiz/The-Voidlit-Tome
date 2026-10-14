"""Voice: local, in-process llama.cpp transduction and narration, with an offline fallback for everything."""

from .llama_local import (ChatModel, LocalLlama, LocalModelError, LocalTransducer, LocalVoice,
                          resolve_model_path)
from .narrator import DEFAULT_BANKS, OfflineVoice
from .prompts import TOME_SYSTEM, compile_narration_prompt, compile_npc_prompt
from .requests import DialogueRequest, NarrationRequest

__all__ = [
    "ChatModel", "LocalLlama", "LocalModelError", "LocalTransducer", "LocalVoice", "resolve_model_path",
    "DEFAULT_BANKS", "OfflineVoice", "TOME_SYSTEM", "compile_narration_prompt", "compile_npc_prompt",
    "DialogueRequest", "NarrationRequest",
]
