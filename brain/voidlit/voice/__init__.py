"""Voice: authored narration and Tome speech (the offline voice), and local, in-process llama.cpp for the sensory
pass and the NPCs' own dialogue, with an offline fallback for both."""

from .llama_local import (ChatModel, LocalLlama, LocalModelError, LocalTransducer, LocalVoice,
                          resolve_model_path)
from .narrator import DEFAULT_BANKS, OfflineVoice
from .prompts import compile_npc_prompt
from .requests import DialogueRequest, NarrationRequest

__all__ = [
    "ChatModel", "LocalLlama", "LocalModelError", "LocalTransducer", "LocalVoice", "resolve_model_path",
    "DEFAULT_BANKS", "OfflineVoice", "compile_npc_prompt", "DialogueRequest", "NarrationRequest",
]
