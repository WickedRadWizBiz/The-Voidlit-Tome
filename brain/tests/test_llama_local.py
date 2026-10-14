"""The language model must be a local file, used in-process, with no network pathway."""

import ast
import asyncio
from pathlib import Path

import pytest

from voidlit.senses import Intent, SensoryParseError, parse_sensory_json
from voidlit.server import ensure_loopback, run_server
from voidlit.voice import LocalModelError, LocalTransducer, LocalVoice, OfflineVoice, resolve_model_path

PACKAGE = Path(__file__).resolve().parents[1] / "voidlit"
NETWORK_MODULES = {"urllib", "http", "requests", "httpx", "aiohttp", "huggingface_hub", "socket", "ftplib"}


@pytest.mark.parametrize("ref", [
    "http://127.0.0.1:8080", "https://example.com/model.gguf", "hf://TheBloke/x.gguf",
    "hf:org/model", "huggingface.co/org/model/model.gguf", "file:///tmp/model.gguf",
])
def test_remote_model_references_are_refused(ref):
    with pytest.raises(LocalModelError, match="local"):
        resolve_model_path(ref)


def test_model_must_be_an_existing_gguf_file(tmp_path):
    with pytest.raises(LocalModelError, match="no model file"):
        resolve_model_path(tmp_path / "missing.gguf")
    other = tmp_path / "model.bin"
    other.write_bytes(b"x")
    with pytest.raises(LocalModelError, match="not a .gguf"):
        resolve_model_path(other)
    good = tmp_path / "Model.GGUF"
    good.write_bytes(b"GGUF")
    assert resolve_model_path(good) == good.resolve()


def test_the_brain_imports_no_network_client():
    """Static guard: no module in the package may import an HTTP/socket client or a model hub."""
    offenders = []
    for py in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            offenders += [f"{py.name}: {n}" for n in names if n.split(".")[0] in NETWORK_MODULES]
            if isinstance(node, ast.Attribute) and node.attr == "from_pretrained":  # hub download helper
                offenders.append(f"{py.name}: .from_pretrained")
    assert offenders == []


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost", "127.0.0.2"])
def test_loopback_hosts_are_accepted(host):
    ensure_loopback(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "::", "example.com", ""])
def test_the_server_never_listens_beyond_this_machine(host):
    with pytest.raises(ValueError, match="only serves this machine"):
        ensure_loopback(host)
    with pytest.raises(ValueError):
        asyncio.run(run_server(host, 0))


# ---------------------------------------------------------------------------------------- real llama.cpp

@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    pytest.importorskip("llama_cpp", reason="llama-cpp-python not installed (pip install -e '.[llm]')")
    pytest.importorskip("gguf", reason="gguf not installed (pip install gguf)")
    from voidlit.voice import LocalLlama

    from .tiny_gguf import write_tiny_llama

    path = write_tiny_llama(tmp_path_factory.mktemp("models") / "tiny.gguf")
    return LocalLlama.load(path, n_ctx=4096)


class NoFallback:
    def extract(self, text, targets=None):
        raise AssertionError("the grammar-locked pass should not need the heuristic fallback")


def test_llama_cpp_accepts_our_grammars(tiny_model):
    """llama.cpp's own GBNF parser must accept every grammar we build, and the reference file."""
    from voidlit.senses import build_sensory_gbnf

    for targets in ([], ["elara"], ["elara", "marcus", "the_sleepers", "jammed_iris"]):
        tiny_model.check_grammar(build_sensory_gbnf(targets))
    tiny_model.check_grammar((Path(__file__).resolve().parents[2] / "grammars" / "sensory.gbnf").read_text())


def test_a_bad_grammar_falls_back_instead_of_crashing(tiny_model):
    broken = 'root ::= "a"\n  "b"'  # a rule continued on a new line without parentheses
    with pytest.raises(ValueError, match="could not parse"):
        tiny_model.chat([{"role": "user", "content": "x"}], max_tokens=4, grammar=broken)


def test_local_llama_grammar_pass_always_parses(tiny_model):
    """Even a random-weight model must emit schema-valid JSON when llama.cpp enforces the GBNF grammar."""
    targets = {"elara": ["elara"], "marcus": ["marcus"]}
    transducer = LocalTransducer(tiny_model, NoFallback())
    for text in ("Elara, I trust you.", "I smash the door!", "look around"):
        s = transducer.extract(text, targets)
        assert isinstance(s.intent, Intent) and s.target in ("none", "elara", "marcus")
        assert 0.0 <= s.aggression <= 1.0 and -1.0 <= s.rapport <= 1.0


def test_local_llama_loads_once_and_speaks(tiny_model, scenario):
    from voidlit.voice import LocalLlama

    assert LocalLlama.load(tiny_model.path, n_ctx=4096) is tiny_model
    out = tiny_model.chat([{"role": "user", "content": "Say something."}], max_tokens=8, seed=1)
    assert isinstance(out, str)
    voice = LocalVoice(tiny_model, OfflineVoice(scenario.narration))
    from voidlit.voice import NarrationRequest

    text = voice.narrate(NarrationRequest(player_text="look", intent="look", approach="none",
                                          chapter_title="T", chapter_text="A room.", entering_chapter=True,
                                          facts=["The player looks."]))
    assert isinstance(text, str) and text


def test_model_text_is_cleaned_of_control_characters():
    from voidlit.voice.llama_local import clean_text

    assert clean_text("  The\x00 dark\x1b[31m\nlistens.\x07 ") == "The dark[31m\nlistens."


def test_parse_rejects_truncated_output():
    with pytest.raises(SensoryParseError):
        parse_sensory_json('{"intent": "look", "approach": "none", "target": "none", "aggression": 0.')
