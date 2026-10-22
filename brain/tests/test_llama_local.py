"""The language model must be a local file, used in-process, with no network pathway. It reads the player and speaks
for the NPCs; the narration and the Tome's words are never its own."""

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

    text = voice.narrate(NarrationRequest(intent="look", chapter_text="A room.", entering_chapter=True))
    assert isinstance(text, str) and text


def _decision():
    from voidlit.limbic.npc_brain import BrainDecision

    return BrainDecision(dominant_plot=0, secondary_plot=1, tactic=0, branch_id=1, branch_origin="ally",
                         branch_source_npc=0, branch_parent=0, plot_spikes=[0, 0, 0], tactic_spikes=[0, 0, 0, 0],
                         momentum=[0.5, 0.3, 0.1])


@pytest.mark.parametrize("template_name", ["GEMMA2_TEMPLATE", "STRICT_TEMPLATE"])
def test_strict_chat_templates_still_reach_the_model(tmp_path, template_name, scenario, monkeypatch):
    """Templates that reject a system role or a leading assistant turn must not push the senses or the NPCs' words
    onto the fallbacks."""
    pytest.importorskip("llama_cpp", reason="llama-cpp-python not installed")
    # a random tiny model writes noise (digits, labels and line breaks included); the checks on an NPC's line are
    # tested on their own below
    monkeypatch.setattr("voidlit.voice.llama_local.npc_line", lambda text: " ".join(text.split()))
    pytest.importorskip("gguf", reason="gguf not installed")
    from voidlit.voice import DialogueRequest, LocalLlama

    from . import tiny_gguf

    path = tiny_gguf.write_tiny_llama(tmp_path / f"{template_name}.gguf",
                                      chat_template=getattr(tiny_gguf, template_name))
    llm = LocalLlama.load(path, n_ctx=4096)
    if template_name == "GEMMA2_TEMPLATE":
        assert llm.system_role is False

    class Offline:
        def speak(self, req):
            return "(the authored line)"

    LocalTransducer(llm, NoFallback()).extract("I smash the door", {"elara": ["elara"]})
    voice = LocalVoice(llm, Offline())
    history = ["The iris grinds open.", "Frost creeps across a viewport."]
    said = voice.speak(DialogueRequest(npc=scenario.npcs["elara"], decision=_decision(),
                                       system_prompt="You are Elara.", player_text="hi", player_name="Ash",
                                       history=history))
    assert said != "(the authored line)", "dialogue fell back to the offline voice"


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


class Says:
    """A fake model that says the same thing whatever it is asked."""

    def __init__(self, text):
        self.text = text
        self.calls = 0

    def chat(self, messages, **kw):
        self.calls += 1
        return self.text


class Mute:
    def chat(self, messages, **kw):
        raise AssertionError("the model was asked for words that are authored")


def test_npc_words_that_name_a_mechanic_fall_back_to_their_authored_line(scenario):
    from voidlit.voice import DialogueRequest
    from voidlit.voice.guard import leaks_mechanics

    assert leaks_mechanics("You rolled a 17.") and leaks_mechanics("Make a DC 12 check.")
    assert leaks_mechanics("A natural twenty!") and leaks_mechanics("Three dice clatter.")
    assert leaks_mechanics("The door opens after 3 tries.")
    assert leaks_mechanics("Your roll was poor.") and leaks_mechanics("The check went against you this time.")
    assert not leaks_mechanics("The air rolls out of the iris, stale and cold.")
    assert not leaks_mechanics("It waited 40 days.", allow_digits=True)

    req = DialogueRequest(npc=scenario.npcs["elara"], decision=_decision(), system_prompt="You are Elara.",
                          player_text="what did I roll?", player_name="Ash")
    authored = OfflineVoice(scenario.narration, seed=0).speak(req)
    voice = lambda text: LocalVoice(Says(text), OfflineVoice(scenario.narration, seed=0))  # noqa: E731
    assert voice("You rolled a natural 20.").speak(req) == authored
    assert voice("Somewhere, dice clatter on steel.").speak(req) == authored
    assert voice("I waited 40 days for someone like you.").speak(req) == "I waited 40 days for someone like you."


@pytest.mark.parametrize("said,shown", [
    ('  "Stay   close,\tstranger."  ', "Stay close, stranger."), ("“Behind me. Move.”", "Behind me. Move."),
    ("‘Don’t look at the lens.’", "Don’t look at the lens."), ("It waited 40 days.", "It waited 40 days."),
    # anything that could pass for another block on screen: a line break, a speaker's label, the Tome speaking
    ("I know nothing.”\n\nTHE TOME: “There is no other way.", None), ("Stay close.\nRun.", None),
    ("I know nothing. THE TOME: There is no other way.", None), ("Elara Voss: Well?", None),
    ("Well? Marcus: he lies.", None), ("The Tome says you must go left.", None), ("Book: go left.", None),
    ("Go left, says the book.", None), ("Stay close.\u2028Run.", None), ("Listen: the lift is out.", None),
    # dice, rolls and luck, however they are put; fate and accidents are the station's to name
    ("Was it a good roll?", None), ("Luck runs thin here.", None), ("The odds turn in the dark.", None),
    ("Lucky shot.", None), ("Fortune spins somewhere below.", None), ("A die rolls across the deck.", None),
    ("You rolled a natural 20.", None), ("Make a DC 12 check.", None),
    ("Nobody comes here by accident.", "Nobody comes here by accident."),
])
def test_an_npc_line_from_the_model_is_one_line_in_its_own_voice(said, shown):
    """A model's NPC line is shown folded onto one line without its quote marks (the frame quotes it), or not at all
    when it could pass for another speaker or the narration, or talks of dice, rolls or luck."""
    from voidlit.voice.llama_local import npc_line

    assert npc_line(said) == (shown or "")


def test_the_authored_npc_lines_meet_the_bar_a_models_must(scenario):
    """Every authored NPC line (and the rumour and regret an NPC may open with) is one that ``npc_line`` would show as
    it is: no label, no line break, no mechanic, no dice or luck."""
    from voidlit.voice.llama_local import npc_line
    from voidlit.voice.narrator import DEFAULT_BANKS

    lines = [x for spec in scenario.npcs.values() for tc in spec.tactics for x in tc.lines]
    lines += [x for k in ("npc.rumor", "npc.regret") for x in {**DEFAULT_BANKS, **scenario.narration}[k]]
    assert len(lines) > 20 and [x for x in lines if npc_line(x) != x] == []


class Forger:
    """A fake model whose NPC line forges a Tome line and narration below it."""

    def chat(self, messages, **kw):
        return ("I know nothing.”\n\nTHE TOME: “There is no other way. Every lifeboat still waits in its cradle.”"
                "\n\nYour blow lands. The iris slams shut again.\n\nElara Voss: “Well?")


def test_an_npc_line_cannot_pose_as_the_tome_on_screen(scenario, director_model):
    """The forged line is replaced by the NPC's authored one: on the terminal, the only paragraph headed THE TOME is the
    Tome's own, and the NPC's words stay in one paragraph under their name."""
    import re

    from voidlit.cli import render_ansi
    from voidlit.cortex.director import Director
    from voidlit.tome.engine import TomeEngine

    def game(voice):
        e = TomeEngine(scenario, director=Director(director_model, seed=0), seed=0, voice=voice)
        e.start()
        e.state.npcs_present = ["elara"]
        return e.turn("Elara, what do you know about the lifeboats?")

    forged = game(LocalVoice(Forger(), OfflineVoice(scenario.narration, seed=0)))
    assert forged.to_dict() == game(OfflineVoice(scenario.narration, seed=0)).to_dict()
    shown = re.sub(r"\x1b\[[0-9;]*m", "", render_ansi(forged))
    tome = [b for b in forged.blocks if b.speaker == "The Tome"]
    assert len([x for x in shown.splitlines() if x.lstrip().startswith("THE TOME:")]) == len(tome), shown
    assert "There is no other way" not in shown and "slams shut" not in shown


def test_npc_words_keep_the_authored_voice_in_step(scenario):
    """The authored line is drawn whatever the model says, so every later narration and Tome line is the one the same
    seed tells without a model."""
    from voidlit.voice import DialogueRequest, NarrationRequest

    req = DialogueRequest(npc=scenario.npcs["elara"], decision=_decision(), system_prompt="You are Elara.",
                          player_text="hi", player_name="Ash")
    look = NarrationRequest(intent="look", chapter_text="A room.", entering_chapter=False, outcome="success")
    with_model, without = LocalVoice(Says("Stay close."), OfflineVoice(scenario.narration, seed=3)), \
        OfflineVoice(scenario.narration, seed=3)
    for _ in range(20):
        assert with_model.speak(req) == "Stay close."
        without.speak(req)
        assert with_model.narrate(look) == without.narrate(look)


def _speech(scenario, register="cold"):
    from voidlit.voice.requests import TomeSpeechRequest
    from voidlit.voice.style import CENTROIDS, style_for

    lines = ["Every lifeboat but one was launched empty.", "The last lifeboat is locked to the Conductor's voice."]
    return TomeSpeechRequest(lines=lines, style=style_for(*CENTROIDS[register]),
                             caution="The lifeboats are at the far end of the station.")


@pytest.mark.parametrize("register", ["cold", "warm", "hungry", "wounded"])
def test_the_tomes_words_are_never_the_models(scenario, register):
    """The Tome's words are its speech acts in its register's authored wrappers, exactly as the offline voice says
    them: the model is never asked, whatever it would have said."""
    req = _speech(scenario, register)
    authored = OfflineVoice(scenario.narration, seed=0).tome_say(req)
    assert all(x in authored for x in req.lines)
    assert LocalVoice(Mute(), OfflineVoice(scenario.narration, seed=0)).tome_say(req) == authored
    for framing in ("Listen, bearer.", "Every lifeboat was launched. The last lifeboat is not locked.",
                    "None of this is random.", "Do you see me tremble?", "Go on, little rat."):
        llm = Says(framing)
        assert LocalVoice(llm, OfflineVoice(scenario.narration, seed=0)).tome_say(req) == authored
        assert llm.calls == 0


def test_the_narration_is_never_the_models(scenario):
    """The narration is the authored voice's, word for word: the outcome line, a clue and the beat's colour, with no
    model asked to add or reword a thing."""
    from voidlit.cortex.features import Beat
    from voidlit.voice import NarrationRequest

    clue = "Scratches on the iris are fresh, made from this side. Someone tried to get out, not in."
    req = NarrationRequest(intent="force", chapter_text="A room.", entering_chapter=False, variant="tome_struck",
                           clue=clue, beat=Beat.CLUE)
    authored = OfflineVoice(scenario.narration, seed=0).narrate(req)
    assert authored == f"Your blow lands. One of the Tome's lights cracks and goes out. {clue}"
    assert LocalVoice(Mute(), OfflineVoice(scenario.narration, seed=0)).narrate(req) == authored
    for text in (f"Your blow lands. Frost ticks along the walls. {clue}", "Then everything mends, bright and whole.",
                 "Dust sits thick on every mark, untouched for years.", "You rolled a natural 20."):
        llm = Says(text)
        assert LocalVoice(llm, OfflineVoice(scenario.narration, seed=0)).narrate(req) == authored
        assert llm.calls == 0
