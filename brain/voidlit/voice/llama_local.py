"""llama.cpp, in-process and local-only: the sensory pass (GBNF-locked) and the NPCs' own dialogue.

The model never writes narration or the Tome's words: those are always authored (the offline voice, from the rules'
verbatim speech acts, register wrappers, phrase banks and core sentences). It reads the player (perception) and speaks
for Elara and Marcus, whose minds are their own.

The language model is a ``.gguf`` file on this machine, loaded with llama-cpp-python, which compiles llama.cpp
into this Python process. There is no server, no socket and no download path:

* model paths must name an existing local ``.gguf`` file; URLs and model-hub references are refused;
* models are never fetched (``Llama.from_pretrained`` is never used);
* nothing in this module imports an HTTP client.

Every call has an offline fallback, so a missing or misbehaving model degrades the prose but never breaks the game.

Install the bindings with ``pip install -e ".[llm]"`` (add your backend's CMAKE_ARGS for GPU builds), then::

    python -m voidlit play --model /path/to/model.gguf
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from ..senses.grammar import SensoryParseError, build_sensory_gbnf, parse_sensory_json
from ..senses.heuristic import HeuristicTransducer
from ..senses.state import SensoryState
from .guard import LINE_BREAK, fold, plain_npc_line
from .narrator import OfflineVoice
from .requests import DialogueRequest, NarrationRequest, TomeSpeechRequest

log = logging.getLogger(__name__)

# What a failed generation can raise: llama-cpp-python signals context overflow and decode failures with
# ValueError/RuntimeError, and a malformed completion surfaces as a lookup error.
GENERATION_ERRORS = (RuntimeError, ValueError, KeyError, IndexError, TypeError, OSError)

SENSORY_SYSTEM = """You are the perception layer of a game master. Read the player's input and rate it.
intent: what the player is trying to do (look, talk, fight, force, sneak, search, use, rest, advance, flee, other).
approach: how (grit = force/violence/endurance, guile = stealth/trickery/precision, insight = analysis/systems/will,
none = no roll implied).
target: who or what it is aimed at, from the allowed ids, else none.
aggression: hostility or threat of violence. deception: lying or bluffing. leverage: bargaining power, offers,
conditional threats. exposure: probing someone's hidden motives or secrets. rapport: warmth (+) or contempt (-).
boldness: appetite for risk. Use two decimals. Be calibrated: most inputs are near 0 on most axes.
Words are aimed at the name they are put to, in any sentence: at its head ("Tome, ...", "Hey Tome, ...",
"Dear Tome, ...", "Tome: ...", "Tome what ...?"), after its last comma ("..., Elara?"), or in the third person
("I ask the Tome who ..."); else at the first one named. A question to no one in particular is put to the Tome, even
beside an act ("I take cover. Where is the lifeboat?"). Telling what you do after a name ("Tome! I smash you.") is
that act, not talk. An act is aimed at its own object, in its own sentence: "I shoot the drone. Elara, run!" shoots
the drone, and "Marcus, I attack the drone." attacks the drone. Something of someone's is not them ("I smash
Elara's chains" does not attack Elara), and a place is not a target ("I hide behind the Tome"). A plan or a condition
("If he moves, I shoot him.", "I'll shoot the drone.") is talk, not an act; giving someone something is talk to them.
A blow at a person or the Tome is only "I <strike> <them>" ("I shoot Marcus."); anything else that names them
("I hurl insults at Marcus", "I search Marcus") is talk to them.
asks: when the player puts a question to the Tome (the narrator), its topic from the allowed topics; vague for a
question with no clear topic; unclear if you cannot tell what they asked; none if it is not a question to the Tome.
claims: a claim id the player asserts, else none. addressed_tome: true if the words are aimed at the Tome."""

_URL_SCHEME = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)
_HUB_PREFIXES = ("hf:", "hf.co/", "huggingface.co/", "www.huggingface.co/")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_QUOTE_MARKS = str.maketrans("", "", "\"“”„‟«»‹›「」『』〝〞〟")     # the frame puts its own round an NPC's words


def clean_text(text: str) -> str:
    """Strip control characters a model may emit; the Tome renders only printable text and newlines."""
    return _CONTROL.sub("", text).strip()


def npc_line(text: str, names: Iterable[str] = ()) -> str:
    """A model's words for an NPC as they may reach the screen, look-alikes folded (``fold``), quote marks stripped (the
    frame quotes them) and whitespace folded; or "" unless they are one line in the closed shape of plain sentences
    (``plain_npc_line``: ordinary words and punctuation only, no word of outcomes, dice or luck, no opening a label
    could be, any word of ``names`` among them). The NPC then says their authored line."""
    text = fold(text).strip()
    if LINE_BREAK.search(text):
        return ""
    line = " ".join(clean_text(text).translate(_QUOTE_MARKS).split()).strip("'‘’ ")
    return line if plain_npc_line(line, names) else ""


class LocalModelError(ValueError):
    """The model could not be used as a local file (bad path, a URL, or llama-cpp-python missing)."""


class ChatModel(Protocol):
    def chat(self, messages: list[dict], *, temperature: float = 0.8, max_tokens: int = 256,
             grammar: str | None = None, seed: int | None = None) -> str: ...


def resolve_model_path(path: str | Path) -> Path:
    """Accept only an existing local .gguf file. URLs and model-hub references are refused outright.

    The .gguf suffix is judged on the name given, so symlinks into content-addressed stores (such as a
    Hugging Face cache snapshot pointing at an extensionless blob) still work.
    """
    text = str(path).strip()
    if _URL_SCHEME.match(text) or text.lower().startswith(_HUB_PREFIXES):
        raise LocalModelError(f"refusing {text!r}: models must be local .gguf files, never URLs or hub names")
    given = Path(text).expanduser()
    if given.suffix.lower() != ".gguf":
        raise LocalModelError(f"{given} is not a .gguf model file")
    p = given.resolve()
    if not p.is_file():
        raise LocalModelError(f"no model file at {given}")
    return p


def _import_llama_cpp():
    try:
        import llama_cpp
    except (ImportError, OSError, RuntimeError) as e:  # OSError: the compiled library failed to load
        raise LocalModelError(
            "llama-cpp-python is not installed; run `pip install -e \".[llm]\"` in brain/ to use a local model"
        ) from e
    return llama_cpp


class LocalLlama:
    """One local model loaded into this process. Shared by every engine that asks for the same file."""

    _cache: dict[tuple, LocalLlama] = {}
    _cache_lock = threading.Lock()

    def __init__(self, path: str | Path, *, n_ctx: int = 4096, n_gpu_layers: int = 0, seed: int = 0,
                 n_threads: int | None = None) -> None:
        self.path = resolve_model_path(path)
        llama_cpp = _import_llama_cpp()
        self._llama_cpp = llama_cpp
        self._grammar_type = llama_cpp.LlamaGrammar
        self._valid_grammars: set[str] = set()
        try:
            self._llama = llama_cpp.Llama(model_path=str(self.path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                                          seed=seed, n_threads=n_threads, verbose=False)
        except Exception as e:  # corrupt/unsupported GGUF, allocation failure, broken chat template...
            raise LocalModelError(f"llama.cpp could not load {self.path}: {e}") from e
        self._lock = threading.Lock()  # a llama.cpp context must not be used from two threads at once
        self.system_role = self._probe_system_role()
        if not self.system_role:
            log.info("%s: chat template has no usable system role; system text is merged into the user turn",
                     self.path.name)

    @classmethod
    def load(cls, path: str | Path, *, n_ctx: int = 4096, n_gpu_layers: int = 0, seed: int = 0,
             n_threads: int | None = None) -> LocalLlama:
        key = (resolve_model_path(path), n_ctx, n_gpu_layers, seed, n_threads)
        with cls._cache_lock:
            if key not in cls._cache:
                cls._cache[key] = cls(key[0], n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, seed=seed, n_threads=n_threads)
            return cls._cache[key]

    def _probe_system_role(self) -> bool:
        """Does this model's chat template accept [system, user]? Many (Gemma-2, Mistral) raise on a system
        message, and llama-cpp-python's built-in 'mistral-instruct' formatter silently drops it."""
        if getattr(self._llama, "chat_format", None) == "mistral-instruct":
            return False
        try:
            self._llama.create_chat_completion(
                messages=[{"role": "system", "content": "Reply."}, {"role": "user", "content": "Hi."}],
                max_tokens=1, temperature=0.0)
        except ValueError:
            return False
        return True

    def layout(self, messages: list[dict]) -> list[dict]:
        """Rewrite any message list into a shape every chat template accepts: an optional system turn (only if the
        template supports one) followed by exactly one user turn. Our callers only send system text, context and a
        user request, so nothing is lost: context that arrives as assistant turns is folded into the user text."""
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system" and m["content"])
        context = " ".join(m["content"] for m in messages if m["role"] == "assistant" and m["content"])
        user = "\n\n".join(m["content"] for m in messages if m["role"] == "user" and m["content"])
        if context:
            user = f"Recent context: {context}\n\n{user}"
        if system and self.system_role:
            return [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return [{"role": "user", "content": f"{system}\n\n{user}" if system else user}]

    def check_grammar(self, grammar: str) -> None:
        """Parse a GBNF grammar with llama.cpp itself before sampling with it.

        llama-cpp-python hands grammar text straight to llama.cpp; if it fails to parse, the sampler is NULL and
        the next sample segfaults the whole process. Checking first turns that into a ValueError (and the caller's
        offline fallback) instead.
        """
        if grammar in self._valid_grammars:
            return
        cpp = self._llama_cpp
        sampler = cpp.llama_sampler_init_grammar(self._llama._model.vocab, grammar.encode("utf-8"), b"root")
        if not sampler:
            raise ValueError("llama.cpp could not parse the GBNF grammar")
        cpp.llama_sampler_free(sampler)
        self._valid_grammars.add(grammar)

    def chat(self, messages: list[dict], *, temperature: float = 0.8, max_tokens: int = 256,
             grammar: str | None = None, seed: int | None = None) -> str:
        compiled = None
        if grammar:
            with self._lock:
                self.check_grammar(grammar)
            compiled = self._grammar_type.from_string(grammar, verbose=False)
        with self._lock:
            out = self._llama.create_chat_completion(messages=self.layout(messages), temperature=temperature,
                                                     max_tokens=max_tokens, grammar=compiled, seed=seed)
        return (out["choices"][0]["message"]["content"] or "").strip()


class LocalTransducer:
    """Pass 1: player text -> SensoryState, grammar-locked to the scene's entity ids."""

    def __init__(self, llm: ChatModel, fallback: HeuristicTransducer | None = None, max_tokens: int = 256) -> None:
        self.llm = llm
        self.fallback = fallback or HeuristicTransducer()
        self.max_tokens = max_tokens

    def extract(self, text: str, targets: Mapping[str, Sequence[str]] | None = None,
                topics: Mapping[str, Sequence[str]] | None = None,
                people: Collection[str] | None = None) -> SensoryState:
        targets, topics = targets or {}, topics or {}
        grammar = build_sensory_gbnf(targets.keys(), topics.keys())
        listing = ", ".join(f"{tid} ({'/'.join(a)})" for tid, a in targets.items()) or "none"
        messages = [
            {"role": "system", "content": SENSORY_SYSTEM},
            {"role": "user", "content": f"Allowed target ids: {listing}\nAllowed topics: "
                                        f"{', '.join(topics) or 'none'}\nPlayer input: {text}"},
        ]
        try:
            out = self.llm.chat(messages, temperature=0.1, max_tokens=self.max_tokens, grammar=grammar, seed=0)
            return parse_sensory_json(out, targets.keys(), topics.keys())
        except (*GENERATION_ERRORS, SensoryParseError) as e:
            log.warning("sensory pass fell back to heuristics: %s", e)
            return self.fallback.extract(text, targets, topics, people)


class LocalVoice:
    """Pass 2: the NPCs' own dialogue. The narration and the Tome's words are the offline voice's, exactly: authored
    text only, and the model is never asked for them."""

    def __init__(self, llm: ChatModel, fallback: OfflineVoice | None = None, temperature: float = 0.85,
                 names: Iterable[str] = ()) -> None:
        self.llm = llm
        self.fallback = fallback or OfflineVoice()
        self.temperature = temperature
        self.names = tuple(names)       # everyone's names, which no model's line may open with as a label

    def narrate(self, req: NarrationRequest) -> str:
        return self.fallback.narrate(req)

    def tome_say(self, req: TomeSpeechRequest) -> str:
        return self.fallback.tome_say(req)

    def speak(self, req: DialogueRequest) -> str:
        # the authored line is chosen first, whatever the model then says, so the offline voice's dice (and with them
        # every later narration) run exactly as they would without a model
        offline = self.fallback.speak(req)
        user = ("(The player has said nothing to you. Act on your own agenda.)" if req.unprompted
                else f"{req.player_name} says: {req.player_text}")
        if req.history:
            user = f"Recent narration (context only): {' '.join(req.history[-2:])}\n\n{user}"
        messages = [{"role": "system", "content": req.system_prompt}, {"role": "user", "content": user}]
        try:
            said = self.llm.chat(messages, temperature=self.temperature, max_tokens=90)
        except GENERATION_ERRORS as e:
            log.warning("dialogue fell back to offline voice: %s", e)
            return offline
        out = npc_line(said, (*self.names, req.npc.name, *req.npc.aliases))
        if said.strip() and not out:
            log.warning("dialogue fell back to offline voice: not one line of plain sentences, or it opened like a "
                        "speaker's label, or named an outcome, dice or luck")
        return out or offline
