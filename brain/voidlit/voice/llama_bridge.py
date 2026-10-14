"""llama.cpp bridge: the sensory pass (GBNF-constrained) and the vocal pass.

Talks to ``llama-server`` over its OpenAI-compatible HTTP API using only the
standard library. Every call has an offline fallback, so a missing, slow or
misbehaving model degrades the prose but never breaks the game.

Start a server with, for example::

    llama-server -m your-model.gguf --port 8080 -c 4096
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence

from ..senses.grammar import SensoryParseError, build_sensory_gbnf, parse_sensory_json
from ..senses.heuristic import HeuristicTransducer
from ..senses.state import SensoryState
from .narrator import OfflineVoice
from .prompts import compile_narration_prompt
from .requests import DialogueRequest, NarrationRequest

log = logging.getLogger(__name__)

SENSORY_SYSTEM = """You are the perception layer of a game master. Read the player's input and rate it.
intent: what the player is trying to do (look, talk, fight, force, sneak, search, use, rest, advance, flee, other).
approach: how (grit = force/violence/endurance, guile = stealth/trickery/precision, insight = analysis/systems/will,
none = no roll implied).
target: who or what it is aimed at, from the allowed ids, else none.
aggression: hostility or threat of violence. deception: lying or bluffing. leverage: bargaining power, offers,
conditional threats. exposure: probing someone's hidden motives or secrets. rapport: warmth (+) or contempt (-).
boldness: appetite for risk. Use two decimals. Be calibrated: most inputs are near 0 on most axes."""


class LlamaServer:
    def __init__(self, url: str = "http://127.0.0.1:8080", timeout: float = 60.0, model: str = "local") -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.model = model

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=3.0) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def chat(self, messages: list[dict], *, temperature: float = 0.8, max_tokens: int = 256,
             grammar: str | None = None, seed: int | None = None) -> str:
        body: dict = {"model": self.model, "messages": messages, "temperature": temperature,
                      "max_tokens": max_tokens, "stream": False}
        if grammar is not None:
            body["grammar"] = grammar
        if seed is not None:
            body["seed"] = seed
        req = urllib.request.Request(
            f"{self.url}/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            payload = json.loads(r.read().decode())
        return payload["choices"][0]["message"]["content"].strip()


class LlamaTransducer:
    """Pass 1: player text -> SensoryState, grammar-locked to the scene's entity ids."""

    def __init__(self, server: LlamaServer, fallback: HeuristicTransducer | None = None) -> None:
        self.server = server
        self.fallback = fallback or HeuristicTransducer()

    def extract(self, text: str, targets: Mapping[str, Sequence[str]] | None = None) -> SensoryState:
        targets = targets or {}
        grammar = build_sensory_gbnf(targets.keys())
        listing = ", ".join(f"{tid} ({'/'.join(a)})" for tid, a in targets.items()) or "none"
        messages = [
            {"role": "system", "content": SENSORY_SYSTEM},
            {"role": "user", "content": f"Allowed target ids: {listing}\nPlayer input: {text}"},
        ]
        try:
            out = self.server.chat(messages, temperature=0.1, max_tokens=160, grammar=grammar, seed=0)
            return parse_sensory_json(out, targets.keys())
        except (urllib.error.URLError, OSError, KeyError, ValueError, SensoryParseError) as e:
            log.warning("sensory pass fell back to heuristics: %s", e)
            return self.fallback.extract(text, targets)


class LlamaVoice:
    """Pass 2: compiled prompts -> prose and dialogue."""

    def __init__(self, server: LlamaServer, fallback: OfflineVoice | None = None,
                 temperature: float = 0.85) -> None:
        self.server = server
        self.fallback = fallback or OfflineVoice()
        self.temperature = temperature

    def narrate(self, req: NarrationRequest) -> str:
        try:
            return self.server.chat(compile_narration_prompt(req), temperature=self.temperature, max_tokens=220)
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            log.warning("narration fell back to offline voice: %s", e)
            return self.fallback.narrate(req)

    def speak(self, req: DialogueRequest) -> str:
        user = ("(The player has said nothing to you. Act on your own agenda.)" if req.unprompted
                else f"{req.player_name} says: {req.player_text}")
        messages = [{"role": "system", "content": req.system_prompt}]
        if req.history:
            messages.append({"role": "assistant", "content": " ".join(req.history[-2:])})
        messages.append({"role": "user", "content": user})
        try:
            out = self.server.chat(messages, temperature=self.temperature, max_tokens=90)
            return out.strip().strip('"')
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            log.warning("dialogue fell back to offline voice: %s", e)
            return self.fallback.speak(req)
