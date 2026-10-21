"""Saves reload and continue bit-exactly, and a refused save changes nothing.

Also: the network's choices reach the game as it made them. The fact it volunteers is the one whose P slot won,
and a stimulus the rules set after the race reaches the amygdala at the next Phase A.
"""

import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")

from voidlit.playtest import bot_command  # noqa: E402
from voidlit.tome.engine import TomeEngine, Turn  # noqa: E402
from voidlit.tome.net_engine import BRAIN_FILE, TomeNetEngine  # noqa: E402
from voidlit.tomebrain.brain import Commit, PhaseBOut  # noqa: E402


def _snapshot(e: TomeNetEngine, path) -> tuple:
    """Everything a load replaces, as a save writes it: the campaign, and the brain with its turn and mind."""
    e.save(path)
    return ((path / "campaign.json").read_text(), e.brain.digest(),
            torch.load(path / BRAIN_FILE, weights_only=True)["extra"])


def test_a_save_from_before_the_first_step_loads_into_a_used_amygdala_as_into_a_fresh_one(scenario, tmp_path):
    """BindsNET creates its learning rule's traces at the first step, so a save made right after the intro has none.
    Loading it into an engine that has played on must drop that timeline's traces, not keep them."""
    a = TomeNetEngine(scenario, seed=4)
    a.start()
    a.save(tmp_path / "intro")
    for _ in range(2):
        a.turn(bot_command(a))
    a.load(tmp_path / "intro")
    b = TomeNetEngine(scenario, seed=9)
    b.load(tmp_path / "intro")
    assert a.brain.digest() == b.brain.digest()
    rule = a.brain.bn.net.connections[("ctx", "bla")].update_rule
    assert not hasattr(rule, "eligibility_trace")
    for _ in range(2):
        cmd = bot_command(a)
        assert a.turn(cmd).to_dict(include_debug=True) == b.turn(cmd).to_dict(include_debug=True)
    assert a.brain.digest() == b.brain.digest()


def test_a_refused_load_leaves_the_game_exactly_as_it_was(scenario, tmp_path):
    """A slot whose brain does not fit (a Director game's, which has none; another window size), a slot from an
    older version, or one missing an NPC's brain is refused before anything changes."""
    d = TomeEngine(scenario, seed=3)
    d.start()
    d.turn("I search the room")
    d.save(tmp_path / "director")
    a = TomeNetEngine(scenario, seed=4)
    a.start()
    a.turn(bot_command(a))
    a.save(tmp_path / "net")
    shutil.copytree(tmp_path / "net", tmp_path / "w1")
    blob = torch.load(tmp_path / "w1" / BRAIN_FILE, weights_only=True)
    blob["W"] = 1
    torch.save(blob, tmp_path / "w1" / BRAIN_FILE)
    shutil.copytree(tmp_path / "net", tmp_path / "old")
    camp = json.loads((tmp_path / "old" / "campaign.json").read_text())
    camp["version"] = 2                       # an older slot (one from before M4 holds another brain)
    (tmp_path / "old" / "campaign.json").write_text(json.dumps(camp))
    shutil.copytree(tmp_path / "net", tmp_path / "torn")
    (tmp_path / "torn" / "brains" / "elara.pt").unlink()
    a.turn(bot_command(a))                    # play on, so a half-done load would show
    before = _snapshot(a, tmp_path / "before")
    for slot, err in (("director", FileNotFoundError), ("w1", ValueError), ("old", ValueError),
                      ("torn", FileNotFoundError)):
        with pytest.raises(err):
            a.load(tmp_path / slot)
        assert _snapshot(a, tmp_path / f"after_{slot}") == before, slot


def test_the_director_engine_continues_after_a_load_into_another_seed(scenario, tmp_path):
    """The Director's encoder draws from its own generator: the save carries it."""
    a = TomeEngine(scenario, seed=2)
    a.start()
    for _ in range(4):
        a.turn(bot_command(a))
    a.save(tmp_path / "slot")
    b = TomeEngine(scenario, seed=102)
    b.load(tmp_path / "slot")
    for _ in range(6):
        cmd = bot_command(a)
        assert a.turn(cmd).to_dict(include_debug=True) == b.turn(cmd).to_dict(include_debug=True)


def _commit(facts: np.ndarray) -> PhaseBOut:
    return PhaseBOut(Commit("PACE", "calm", 120, False), Commit("RHETORIC", "disclose", 120, False),
                     Commit("FRAME", "help", 120, False), np.full(8, 0.125), 120, facts)


def test_the_volunteered_fact_is_the_one_whose_slot_won(scenario):
    """The P slots are driven at Phase B; the turn's clue may tell the player one of those facts before the Tome
    speaks. The winning slot still means the fact it was driven with, wherever that fact sits now."""
    e = TomeNetEngine(scenario, seed=0, brain=SimpleNamespace())     # no network needed to map a commit
    e._fact_salience()
    driven = [f.id for f in e.ledger.slots(e.chapter.id)]
    assert len(driven) >= 5
    e.ledger.disclose([driven[1]])                   # revealed between the race and the Tome's words
    e.last_commit = _commit(np.eye(8)[3] * 9)        # the network chose the fact in slot 3
    now = [f.id for f in e.ledger.slots(e.chapter.id)]
    rhetoric, _, slot = e._choose_rhetoric(Turn(text=""), len(now))
    assert rhetoric == "disclose" and now[slot] == driven[3]

    e._fact_salience()
    driven = [f.id for f in e.ledger.slots(e.chapter.id)]
    e.ledger.disclose([driven[2]])                   # the chosen fact itself is told meanwhile: nothing to volunteer
    e.last_commit = _commit(np.eye(8)[2] * 9)
    assert e._choose_rhetoric(Turn(text=""), len(e.ledger.slots(e.chapter.id)))[2] is None


def test_a_stimulus_after_the_race_reaches_the_amygdala_next_turn_and_survives_a_save(scenario, tmp_path,
                                                                                     monkeypatch):
    """An NPC beat runs after Phase B; its extract tactic marks a loss. That stimulus arrives at the next Phase A
    (it is not dropped), and a save made in between carries it."""
    a = TomeNetEngine(scenario, seed=4)
    a.start()
    beat = a._beat

    def npc_beat(t):
        beat(t)
        t.us["loss"] = 1.0                   # what _tactic_check does for an extract tactic, after the race

    monkeypatch.setattr(a, "_beat", npc_beat)
    a.turn(bot_command(a))
    monkeypatch.setattr(a, "_beat", beat)
    a.save(tmp_path / "slot")
    b = TomeNetEngine(scenario, seed=9)
    b.load(tmp_path / "slot")
    sent = {"a": [], "b": []}
    for k, e in (("a", a), ("b", b)):
        def spy(us, k=k, set_us=e.brain.set_us):
            sent[k].append(dict(us))
            set_us(us)
        monkeypatch.setattr(e.brain, "set_us", spy)
    cmd = bot_command(a)
    assert a.turn(cmd).to_dict(include_debug=True) == b.turn(cmd).to_dict(include_debug=True)
    assert sent["a"][0] == sent["b"][0] == {"loss": 1.0}         # Phase A, before the rules resolve this turn
    assert a.brain.digest() == b.brain.digest()
