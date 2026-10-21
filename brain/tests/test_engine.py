import json

import pytest

from voidlit.app import build_engine
from voidlit.playtest import bot_command
from voidlit.cortex.director import Director
from voidlit.senses import HeuristicTransducer
from voidlit.tome.engine import TOME_LINES, TomeEngine, Turn
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
    for cmd in ["press on", "I search", "Elara, what are you hiding?"]:
        assert a.turn(cmd).to_dict(include_debug=True) == b.turn(cmd).to_dict(include_debug=True)


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


def test_a_model_given_to_the_app_speaks_for_the_npcs_and_nothing_else(scenario):
    """With a model, the app's voice is a LocalVoice: the model is asked for the NPCs' words only, and every narration
    and Tome line is the one the same seed tells without a model."""
    llm = FakeLlm("Stay close to me.")
    a, _ = build_engine(seed=4, voice_llm=llm, brain="director")
    b, _ = build_engine(seed=4, brain="director")
    a.start(), b.start()
    is_npc = lambda x: x.get("data", {}).get("npc") not in (None, "tome")  # noqa: E731
    npc_words = []
    for text in ["Elara, can we trust each other?", "Tome, what does Elara want?", "I force the iris",
                 "Elara, what are you hiding?", "Where is the lifeboat?", "look around"]:
        fa, fb = a.turn(text).to_dict(), b.turn(text).to_dict()
        npc_words += [x["spans"][1]["text"] for x in fa["blocks"] if is_npc(x)]
        assert [x for x in fa["blocks"] if not is_npc(x)] == [x for x in fb["blocks"] if not is_npc(x)]
    assert llm.calls and all("[DOMINANT PLOT]" in m[0]["content"] for m, _ in llm.calls)
    assert "“Stay close to me.”" in npc_words


class Ears(OfflineVoice):
    """The offline voice, keeping every request for an NPC's words."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.asked = []

    def speak(self, req):
        self.asked.append(req)
        return super().speak(req)


def test_words_lost_to_panic_reach_no_one(scenario, director_model):
    """A turn lost to panic is narrated as lost, and no NPC answers words the player never said."""
    e = make(scenario, director_model, seed=2)
    e.voice = Ears(scenario.narration, seed=2)
    e.start()
    e.state.npcs_present = ["elara"]
    e.state.lose_turn = True
    f = e.turn("Elara, I trust you, please, we will stay together")
    assert "The moment passes without you." in f.text and "Elara studies you" not in f.text
    assert not [r for r in e.voice.asked if not r.unprompted]


def test_words_said_to_the_tome_are_not_the_lone_npcs_to_answer(scenario, director_model):
    for text in ("Tome, what does Elara want?", "Hey Tome, what does Elara want?", "Where is Elara, Tome?"):
        e = make(scenario, director_model, seed=2)
        e.voice = Ears(scenario.narration, seed=2)
        e.start()
        e.state.npcs_present = ["elara"]
        f = e.turn(text)
        assert not [r for r in e.voice.asked if not r.unprompted], text
        assert "Elara has lived forty days by rationing light." in f.text, text


def test_a_search_that_finds_nothing_new_is_not_told_as_a_find(scenario, director_model):
    for seed in range(20):
        e = make(scenario, director_model, seed=seed)
        e.start()
        e.state.obstacles.clear()
        e.state.clues_revealed = list(e.chapter.clues) + list(scenario.lore)
        e.state.light_taken = True
        f = e.turn("I search the room")
        rec = next(r for r in e.checks.records if r.action == "player.search")
        if rec.degree.value != "fail":
            assert "nothing new" in f.text, f.text
            assert "find what your eyes missed" not in f.text and "You find something" not in f.text
            return
    pytest.fail("no search got past a fail")


def test_light_fed_to_a_full_tome_does_not_burn_brighter(scenario, director_model):
    e = make(scenario, director_model, seed=1, background="salvager")
    e.start()
    assert e.state.lumen.dice == 6
    f = e.turn("I light the flare")
    assert "burns no brighter" in f.text and "The words burn brighter" not in f.text
    e.state.lumen.dice = 3
    f = e.turn("I light the flare")
    assert "The words burn brighter" in f.text and e.state.lumen.dice == 4


def test_the_blow_that_puts_out_the_last_light_is_answered_as_such(scenario, director_model):
    for seed in range(30):
        e = make(scenario, director_model, seed=seed)
        e.start()
        e.state.lumen.dice = 1
        f = e.turn("I smash the tome")
        if e.state.lumen.dice == 0:
            tome = " ".join(b.text for b in f.blocks if b.speaker == "The Tome")
            assert any(x in tome for x in TOME_LINES["struck_out"][1]), tome
            assert not any(x in tome for x in TOME_LINES["struck"][1]), tome
            return
    pytest.fail("no blow landed")


def test_a_power_lurch_in_the_dark_drags_no_light_down(scenario, director_model):
    e = make(scenario, director_model, seed=0)
    e.start()
    clock = next(c for c in e.state.clocks if c.consequence == "lumen")
    for c in e.state.clocks:
        c.filled = 0
    clock.filled = clock.segments - 1
    e.state.lumen.dice = 0
    t = Turn(text="")
    e._tick_worst_clock(t, 1)
    assert "the dark deepens" in t.beat_fact and "light" not in t.beat_fact, t.beat_fact
