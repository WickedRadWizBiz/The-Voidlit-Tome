"""Voice: llama.cpp transduction and narration, with an offline fallback for everything."""

from .llama_bridge import LlamaServer, LlamaTransducer, LlamaVoice
from .narrator import DEFAULT_BANKS, OfflineVoice
from .prompts import TOME_SYSTEM, compile_narration_prompt, compile_npc_prompt
from .requests import DialogueRequest, NarrationRequest

__all__ = [
    "LlamaServer", "LlamaTransducer", "LlamaVoice", "DEFAULT_BANKS", "OfflineVoice", "TOME_SYSTEM",
    "compile_narration_prompt", "compile_npc_prompt", "DialogueRequest", "NarrationRequest",
]
