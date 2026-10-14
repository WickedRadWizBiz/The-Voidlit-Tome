import json

import pytest

from voidlit.app import build_engine
from voidlit.cortex.director import Director
from voidlit.playtest import bot_command, play, play_campaign
from voidlit.senses import HeuristicTransducer
from voidlit.tome.engine import TomeEngine
from voidlit.voice import LocalTransducer, LocalVoice, OfflineVoice

CLASSES = ["breacher", "archivist", "vessel", "hauler", "pilgrim", "fence"]


def make(codex, lantern, director_model, seed=0, **kw):
    return TomeEngine(codex, director=Director(director_model, seed=seed), seed=seed,
                      voice=OfflineVoice(lantern.narration, seed=seed), **kw)


def test_codex_matches_the_forge(codex):
    codex.validate()
    assert sorted(codex.classes) == sorted(CLASSES)
    for cid in CLASSES:
        for stat in ("might", "grace", "wits", "nerve"):
            assert codex.opening(cid, stat), f"{cid} has no {stat} Opening"
    assert codex.register_for(0) == "plain" and codex.register_for(8) == "eldritch"
    assert codex.register_for(2, panicked=True) == "breakthrough"


def test_start_frame(codex, lantern, director_model):
    e = make(codex, lantern, director_model, seed=1, class_id="hauler", player_name="Ash")
    f = e.start()
    st = f.status
    assert st["bearer"]["class"] == "Hauler" and st["bearer"]["name"] == "Ash"
    assert st["grit"] == [5, 5] and st["lumen"][0] == st["lumen"][1]
    assert st["pending"]["type"] == "draw"
    assert f.blocks[-1].role == "draw"
    json.dumps(f.to_dict(include_debug=True))


def test_opening_roll_moves_two_stats(codex, lantern, director_model):
    for seed in range(12):
        e = make(codex, lantern, director_model, seed=seed, class_id="breacher")
        base = codex.classes["breacher"]
        op = e.state.opening
        diff = {k: v - int(base[k]) for k, v in e.bearer.stats().items() if v != int(base[k])}
        if op["raised"]:
            assert diff.get(op["stat"]) == 1
            if op["lowered"]:
                assert diff[op["lowered"]] == -1 and len(diff) == 2
        else:
            assert diff == {}


@pytest.mark.parametrize("cls", CLASSES)
def test_a_bot_can_finish_a_trial(codex, lantern, director_model, cls):
    e = make(codex, lantern, director_model, seed=CLASSES.index(cls), class_id=cls)
    e.start()
    for _ in range(300):
        f = e.turn(bot_command(e))
        json.dumps(f.to_dict(include_debug=True))
        assert f.blocks[0].role == "echo"
        if f.over or e.state.stage in ("haven", "fallen"):
            break
    assert e.state.stage in ("haven", "fallen", "over"), "a trial should end well within 300 turns"


def test_a_campaign_outlives_its_bearers(codex, lantern, director_model):
    e = make(codex, lantern, director_model, seed=2)
    r = play_campaign(e, trials=8)
    assert r["trials"] >= 8 or r["stage"] == "over"
    c = e.state.campaign
    assert c.agenda and 0 <= c.fragments <= 6 and c.hunger >= 0


def test_runs_are_deterministic(codex, lantern, director_model):
    a = make(codex, lantern, director_model, seed=5, place="hollow_lantern")
    b = make(codex, lantern, director_model, seed=5, place="hollow_lantern")
    assert a.start().to_dict() == b.start().to_dict()
    for _ in range(25):
        cmd = bot_command(a)
        assert a.turn(cmd).to_dict(include_debug=True) == b.turn(cmd).to_dict(include_debug=True)


def test_save_and_load_resume_identically(codex, lantern, director_model, tmp_path):
    a = make(codex, lantern, director_model, seed=8, class_id="pilgrim", place="hollow_lantern")
    a.start()
    for _ in range(10):
        a.turn(bot_command(a))
    a.save(tmp_path / "slot")
    b = make(codex, lantern, director_model, seed=99)
    b.load(tmp_path / "slot")
    b.director.gen.set_state(a.director.gen.get_state())
    for _ in range(10):
        cmd = bot_command(a)
        assert a.turn(cmd).to_dict() == b.turn(cmd).to_dict()


def _until_npc(e, limit=60):
    e.start()
    for _ in range(limit):
        if e.state.npcs_present:
            return e.state.npcs_present[0]
        e.turn(bot_command(e))
    pytest.skip("no one in this place showed up for this seed")


def test_talking_reaches_the_npc_brain(codex, lantern, director_model):
    e = make(codex, lantern, director_model, seed=3, place="hollow_lantern")
    nid = _until_npc(e)
    name = e.npcs[nid].name
    f = e.turn(f'"{name.split()[0]}, I promise we will get out of here together."')
    assert nid in f.debug["decisions"]
    assert any(b.role == "npc" and b.speaker == name for b in f.blocks)


def test_attacking_an_npc_drives_them_off(codex, lantern, director_model):
    e = make(codex, lantern, director_model, seed=3, place="hollow_lantern")
    nid = _until_npc(e)
    e.turn(f"I attack {e.npcs[nid].name.split()[0]}")
    assert nid not in e.state.npcs_present


def test_the_oracle_answers(codex, lantern, director_model):
    e = make(codex, lantern, director_model, seed=4)
    e.start()
    f = e.turn("? is the air breathable")
    assert any(b.data.get("oracle") for b in f.blocks)


def test_stats_and_kinds_are_honoured(codex, lantern, director_model):
    e = make(codex, lantern, director_model, seed=6, class_id="archivist")
    e.start()
    while e.trial.pending:
        e.turn(bot_command(e))
    accepts = {s.value for s in e.trial.chapter.accepts}
    f = e.turn("I hold steady and face it")
    assert f.debug["events"], "a cast should happen"
    cast = next(b for b in f.blocks if b.role == "dice")
    assert cast.data["stat"] in accepts


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
    good = FakeLlm('{"intent": "fight", "stat": "might", "target": "elara", "aggression": 0.9, '
                   '"deception": 0.0, "leverage": 0.0, "exposure": 0.0, "rapport": -0.8, "boldness": 0.7}')
    s = LocalTransducer(good).extract("whatever", {"elara": ["elara"]})
    assert s.intent.value == "fight" and s.target == "elara" and s.stat.value == "might"
    assert 'target ::= "none" | "elara"' in good.calls[0][1]["grammar"]
    assert good.calls[0][1]["temperature"] <= 0.2
    bad = FakeLlm("I refuse to output JSON")
    s = LocalTransducer(bad, HeuristicTransducer()).extract("I pry the door open", {})
    assert s.intent.value == "force"
    broken = FakeLlm(RuntimeError("llama_decode returned -1"))
    assert LocalTransducer(broken).extract("I pry the door open", {}).intent.value == "force"


def test_local_voice_falls_back_offline(codex, lantern, director_model):
    for failure in (ValueError("Requested tokens exceed context window"), RuntimeError("decode"), ""):
        e = make(codex, lantern, director_model, seed=2)
        e.voice = LocalVoice(FakeLlm(failure), OfflineVoice(lantern.narration, seed=0))
        e.start()
        f = e.turn(bot_command(e))
        assert any(b.role == "narration" and b.text for b in f.blocks)


def test_build_engine_without_models_is_offline():
    engine, llm = build_engine(seed=1)
    assert llm is False and isinstance(engine.voice, OfflineVoice)
    engine, llm = build_engine(seed=1, voice_llm=FakeLlm("The dark listens."))
    assert llm is True and isinstance(engine.voice, LocalVoice)


def test_play_reports_a_result():
    engine, _ = build_engine(seed=11, class_id="fence")
    r = play(engine)
    assert r["stage"] in ("haven", "fallen", "over")
