import json

import pytest

from voidlit.app import build_engine
from voidlit.playtest import bot_command
from voidlit.cortex.director import Director
from voidlit.senses import HeuristicTransducer
from voidlit.tome.engine import TomeEngine
from voidlit.voice import LocalTransducer, LocalVoice, OfflineVoice

def make(scenario, director_model, seed=0, **kw):
    return TomeEngine(scenario, director=Director(director_model, seed=seed), seed=seed,
                      voice=OfflineVoice(scenario.narration, seed=seed), **kw)


def test_start_frame(scenario, director_model):
    e = make(scenario, director_model, seed=1, background="engineer", player_name="Ash")
    f = e.start()
    roles = [b.role for b in f.blocks]
    assert roles[:3] == ["title", "title", "narration"] and roles[-1] == "prompt"
    assert set(f.status) == {"hull", "stress", "lumen", "chapter", "won", "scene"}
    assert f.status["lumen"] == [6, 6]
    assert set(f.mood) == {"lumen", "stress", "dark", "pulse"}
    sheet = e.state.character.sheet()
    assert sheet["background"] == "Engineer" and sheet["resolve"] == 1
    json.dumps(f.to_dict(include_debug=True))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_bot_can_finish_a_run(scenario, director_model, seed):
    e = make(scenario, director_model, seed=seed, background="marine")
    e.start()
    for _ in range(150):
        f = e.turn(bot_command(e))
        json.dumps(f.to_dict(include_debug=True))
        assert f.blocks[0].role == "echo" and any(b.role == "narration" and b.text for b in f.blocks)
        if f.over:
            break
    assert e.state.over, "a run should end (escape or death) well within 150 turns"
    assert e.state.won or e.state.character.dead


def test_runs_are_deterministic(scenario, director_model):
    script = ["look around", "Elara, are you okay?", "I force the iris", "I analyze the iris", "press on",
              "I search the room", "I shoot it", "I rest", "go on"]
    a, b = make(scenario, director_model, seed=5), make(scenario, director_model, seed=5)
    assert a.start().text == b.start().text
    for cmd in script:
        assert a.turn(cmd).to_dict(include_debug=True) == b.turn(cmd).to_dict(include_debug=True)


def test_save_and_load_resume_identically(scenario, director_model, tmp_path):
    a = make(scenario, director_model, seed=8)
    a.start()
    for cmd in ["Elara, can we trust each other?", "I force the iris", "I analyze the iris"]:
        a.turn(cmd)
    a.save(tmp_path / "slot")
    b = make(scenario, director_model, seed=8)
    b.load(tmp_path / "slot")
    b.director.gen.set_state(a.director.gen.get_state())
    for cmd in ["press on", "I search", "Elara, what are you hiding?"]:
        assert a.turn(cmd).to_dict() == b.turn(cmd).to_dict()


def test_talking_reaches_the_npc_brain(scenario, director_model):
    e = make(scenario, director_model, seed=3)
    e.start()
    f = e.turn('"Elara, I promise we will get out of here together."')
    assert "elara" in f.debug["decisions"]
    assert any(b.role == "npc" and b.speaker == "Elara Voss" for b in f.blocks)


def test_attacking_an_npc_drives_them_off(scenario, director_model):
    e = make(scenario, director_model, seed=4)
    e.start()
    e.turn("I shoot Elara")
    assert "elara" not in e.state.npcs_present


def test_blocked_advance_does_not_move(scenario, director_model):
    e = make(scenario, director_model, seed=6)
    e.start()
    f = e.turn("press on")
    assert e.state.deck.index == 0 and "Not yet" in f.text


class FakeLlm:
    """Stands in for a loaded local model (anything with LocalLlama.chat's signature)."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append((messages, kw))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_local_transducer_uses_grammar_and_falls_back():
    good = FakeLlm('{"intent": "fight", "approach": "grit", "target": "elara", "aggression": 0.9, '
                   '"deception": 0.0, "leverage": 0.0, "exposure": 0.0, "rapport": -0.8, "boldness": 0.7}')
    s = LocalTransducer(good).extract("whatever", {"elara": ["elara"]})
    assert s.intent.value == "fight" and s.target == "elara"
    assert 'target ::= "none" | "elara"' in good.calls[0][1]["grammar"]
    assert good.calls[0][1]["temperature"] <= 0.2
    bad = FakeLlm("I refuse to output JSON")
    s = LocalTransducer(bad, HeuristicTransducer()).extract("I pry the door open", {})
    assert s.intent.value == "force"
    broken = FakeLlm(RuntimeError("llama_decode returned -1"))
    assert LocalTransducer(broken).extract("I pry the door open", {}).intent.value == "force"


def test_local_voice_falls_back_offline(scenario, director_model):
    for failure in (ValueError("Requested tokens exceed context window"), RuntimeError("decode"), ""):
        voice = LocalVoice(FakeLlm(failure), OfflineVoice(scenario.narration, seed=0))
        e = make(scenario, director_model, seed=2)
        e.voice = voice
        e.start()
        f = e.turn("I force the iris")
        assert any(b.role == "narration" and b.text for b in f.blocks)


def test_build_engine_without_models_is_offline():
    engine, llm = build_engine(seed=1)
    assert llm is False and isinstance(engine.voice, OfflineVoice)
    engine, llm = build_engine(seed=1, voice_llm=FakeLlm("The dark listens."))
    assert llm is True and isinstance(engine.voice, LocalVoice)
