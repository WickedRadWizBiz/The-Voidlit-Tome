"""The language model must be a local file, used in-process, with no network pathway."""

import ast
import asyncio
from pathlib import Path

import pytest

from voidlit.senses import Intent, SensoryParseError, parse_sensory_json
from voidlit.server import ensure_loopback, run_server
from voidlit.voice import LocalModelError, LocalTransducer, LocalVoice, OfflineVoice, resolve_model_path

PACKAGE = Path(__file__).resolve().parents[1] / "voidlit"
# Common outbound clients and model hubs. This is a regression tripwire, not a sandbox: the brain's one socket is the
# loopback-only `websockets` server for Unreal, which is checked separately below.
NETWORK_MODULES = {"urllib", "urllib3", "http", "requests", "httpx", "aiohttp", "huggingface_hub", "socket", "ssl",
                   "ftplib", "smtplib", "xmlrpc", "telnetlib"}


@pytest.mark.parametrize("ref", [
    "http://127.0.0.1:8080", "https://example.com/model.gguf", "hf://TheBloke/x.gguf",
    "hf:org/model", "huggingface.co/org/model/model.gguf", "file:///tmp/model.gguf",
])
def test_remote_model_references_are_refused(ref):
    with pytest.raises(LocalModelError, match="local"):
        resolve_model_path(ref)


def test_local_paths_that_merely_mention_hubs_are_fine(tmp_path):
    # A Hugging Face cache snapshot: snapshots/<rev>/model.gguf -> ../../blobs/<sha> (no extension on the blob).
    blob = tmp_path / "hub" / "blobs" / "3f9a1c"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"GGUF")
    snap = tmp_path / "hub" / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (snap / "model.gguf").symlink_to(Path("../../blobs/3f9a1c"))
    assert resolve_model_path(snap / "model.gguf") == blob.resolve()
    # A local mirror whose folder is literally named after the hub.
    mirror = tmp_path / "mirror" / "huggingface.co" / "org" / "model.gguf"
    mirror.parent.mkdir(parents=True)
    mirror.write_bytes(b"GGUF")
    assert resolve_model_path(mirror) == mirror.resolve()


def test_unloadable_model_is_a_clean_error(tmp_path):
    pytest.importorskip("llama_cpp", reason="llama-cpp-python not installed")
    from voidlit.voice import LocalLlama

    bad = tmp_path / "corrupt.gguf"
    bad.write_bytes(b"this is not a gguf file at all")
    with pytest.raises(LocalModelError, match="could not load"):
        LocalLlama(bad)


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
    """Static guard: no module in the package may import an HTTP/socket client, a model hub, or JAX's multi-host
    runtime (the Tome's network runs in this process on this machine; gate G22)."""
    offenders = []
    for py in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            offenders += [f"{py.name}: {n}" for n in names
                          if n.split(".")[0] in NETWORK_MODULES or n.startswith(("llama_cpp.server", "jax.distributed",
                                                                                     "jax.experimental.multihost"))]
            if isinstance(node, ast.Attribute) and node.attr == "from_pretrained":  # hub download helper
                offenders.append(f"{py.name}: .from_pretrained")
            if isinstance(node, ast.Attribute) and node.attr == "distributed":     # jax.distributed: multi-host RPC
                offenders.append(f"{py.name}: .distributed")
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


@pytest.mark.parametrize("template_name", ["GEMMA2_TEMPLATE", "STRICT_TEMPLATE"])
def test_strict_chat_templates_still_reach_the_model(tmp_path, template_name, scenario, monkeypatch):
    """Templates that reject a system role or a leading assistant turn must not push us onto the fallbacks."""
    pytest.importorskip("llama_cpp", reason="llama-cpp-python not installed")
    # a random tiny model writes noise (digits included); the mechanics guard is tested on its own below
    monkeypatch.setattr("voidlit.voice.llama_local.leaks_mechanics", lambda text, allow_digits=False: False)
    pytest.importorskip("gguf", reason="gguf not installed")
    from voidlit.limbic.npc_brain import BrainDecision
    from voidlit.voice import DialogueRequest, LocalLlama, NarrationRequest

    from . import tiny_gguf

    path = tiny_gguf.write_tiny_llama(tmp_path / f"{template_name}.gguf",
                                      chat_template=getattr(tiny_gguf, template_name))
    llm = LocalLlama.load(path, n_ctx=4096)
    if template_name == "GEMMA2_TEMPLATE":
        assert llm.system_role is False

    class Refuse:
        def narrate(self, req):
            raise AssertionError("narration fell back to the offline voice")

        def speak(self, req):
            raise AssertionError("dialogue fell back to the offline voice")

    LocalTransducer(llm, NoFallback()).extract("I smash the door", {"elara": ["elara"]})
    voice = LocalVoice(llm, Refuse())
    history = ["The iris grinds open.", "Frost creeps across a viewport."]
    voice.narrate(NarrationRequest(player_text="look", intent="look", approach="none", chapter_title="T",
                                   chapter_text="A room.", entering_chapter=False, facts=["x"], history=history))
    spec = scenario.npcs["elara"]
    decision = BrainDecision(dominant_plot=0, secondary_plot=1, tactic=0, branch_id=1, branch_origin="ally",
                             branch_source_npc=0, branch_parent=0, plot_spikes=[0, 0, 0], tactic_spikes=[0, 0, 0, 0],
                             momentum=[0.5, 0.3, 0.1])
    voice.speak(DialogueRequest(npc=spec, decision=decision, system_prompt="You are Elara.", player_text="hi",
                                player_name="Ash", history=history))


@pytest.mark.skipif(not Path("/proc/self/maps").exists(), reason="needs Linux /proc")
def test_llama_cpp_download_library_is_never_loaded(tiny_model):
    """llama-cpp-python installs libllama-common (llama.cpp's HTTP/TLS model downloader) but must never load it:
    the bindings use only libllama and libggml, which contain no network code."""
    tiny_model.chat([{"role": "user", "content": "x"}], max_tokens=2)
    maps = Path("/proc/self/maps").read_text()
    assert "libllama-common" not in maps


def test_model_text_is_cleaned_of_control_characters():
    from voidlit.voice.llama_local import clean_text

    assert clean_text("  The\x00 dark\x1b[31m\nlistens.\x07 ") == "The dark[31m\nlistens."


def test_parse_rejects_truncated_output():
    with pytest.raises(SensoryParseError):
        parse_sensory_json('{"intent": "look", "approach": "none", "target": "none", "aggression": 0.')


def test_voice_guard_sends_dice_talk_to_the_offline_voice(scenario):
    from voidlit.voice import LocalVoice, NarrationRequest
    from voidlit.voice.guard import leaks_mechanics

    assert leaks_mechanics("You rolled a 17.") and leaks_mechanics("Make a DC 12 check.")
    assert leaks_mechanics("A natural twenty!") and leaks_mechanics("Three dice clatter.")
    assert leaks_mechanics("The door opens after 3 tries.")
    assert not leaks_mechanics("The air rolls out of the iris, stale and cold.")
    assert not leaks_mechanics("It waited 40 days.", allow_digits=True)

    class Says:
        def __init__(self, text):
            self.text = text

        def chat(self, messages, **kw):
            return self.text

    class Offline:
        def narrate(self, req):
            return "offline"

    req = NarrationRequest(player_text="what did I roll?", intent="other", approach="none", chapter_title="T",
                           chapter_text="A room.", entering_chapter=False, facts=[])
    assert LocalVoice(Says("You rolled a natural 20."), Offline()).narrate(req) == "offline"
    assert LocalVoice(Says("The dark does not answer."), Offline()).narrate(req) == "The dark does not answer."
