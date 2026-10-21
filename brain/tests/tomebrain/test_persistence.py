"""Saves reload and continue bit-exactly, and a refused save changes nothing.

Also: the network's choices reach the game as it made them. The fact it volunteers is the one whose P slot won,
and a stimulus the rules set after the race reaches the amygdala at the next Phase A.
"""

import asyncio
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")

from voidlit import cli  # noqa: E402
from voidlit.limbic.npc_brain import NPCBrain  # noqa: E402
from voidlit.playtest import bot_command  # noqa: E402
from voidlit.senses import SensoryState  # noqa: E402
from voidlit.server import TomeSession  # noqa: E402
from voidlit.tome.engine import SaveError, TomeEngine, Turn  # noqa: E402
from voidlit.tome.net_engine import BRAIN_FILE, TomeNetEngine  # noqa: E402
from voidlit.tomebrain.brain import Commit, PhaseBOut  # noqa: E402

WARM = SensoryState(rapport=0.8, leverage=0.3)
HOSTILE = SensoryState(aggression=0.9, exposure=0.8, rapport=-0.7)
RULE_STATE = ("p_plus", "p_minus", "eligibility", "eligibility_trace")   # what MSTDPET makes at its first step
DROP = object()


def _flat(obj, key: str = "") -> dict:
    """A nested save blob as {path: leaf}, tensors as arrays, so two can be compared leaf by leaf."""
    if isinstance(obj, dict):
        return {k2: v2 for k, v in obj.items() for k2, v2 in _flat(v, f"{key}/{k}").items()}
    if isinstance(obj, (list, tuple)):
        return {k2: v2 for i, v in enumerate(obj) for k2, v2 in _flat(v, f"{key}/{i}").items()}
    return {key: obj.numpy() if isinstance(obj, torch.Tensor) else obj}


def _held(e: TomeEngine, path) -> dict:
    """Everything a load replaces, as a save writes it: the campaign, every NPC brain, and the Tome's brain (all
    partitions, the bus and the clock, with its turn and mind)."""
    e.save(path)
    out = {"campaign": (path / "campaign.json").read_text()}
    for nid in e.brains:
        out.update(_flat(torch.load(path / "brains" / f"{nid}.pt", weights_only=False), f"npc.{nid}"))
    if (path / BRAIN_FILE).exists():
        out.update(_flat(torch.load(path / BRAIN_FILE, weights_only=True), "brain"))
    return out


def _assert_same(now: dict, before: dict, what: str = "") -> None:
    assert now.keys() == before.keys(), (what, sorted(now.keys() ^ before.keys()))
    for k, v in before.items():
        same = np.array_equal(now[k], v) if isinstance(v, np.ndarray) else now[k] == v
        assert same, (what, k)


def _edit(*path, value=DROP):
    """Set (or drop) one entry of a nested save blob."""
    def fn(blob):
        *parents, last = path
        for k in parents:
            blob = blob[k]
        if value is DROP:
            del blob[last]
        else:
            blob[last] = value
    return fn


def _variant(root, name: str, *, campaign=None, brain=None, npc=None, npc_bytes=None) -> str:
    """A copy of the slot ``root/net`` with its campaign, its brain file or the last NPC's brain file changed."""
    d = root / name
    shutil.copytree(root / "net", d)
    if campaign is not None:
        blob = json.loads((d / "campaign.json").read_text())
        campaign(blob)
        (d / "campaign.json").write_text(json.dumps(blob))
    if brain is not None:
        blob = torch.load(d / BRAIN_FILE, weights_only=True)
        brain(blob)
        torch.save(blob, d / BRAIN_FILE)
    last = max((d / "brains").iterdir())
    if npc is not None:
        state = torch.load(last, weights_only=False)
        npc(state)
        torch.save(state, last)
    if npc_bytes is not None:
        last.write_bytes(npc_bytes)
    return name


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
    """A slot that does not fit or is damaged anywhere (its campaign, an NPC's brain, any partition of the Tome's
    brain, its bus, clock or mind) is refused with SaveError before anything changes."""
    d = TomeEngine(scenario, seed=3)
    d.start()
    d.turn("I search the room")
    d.save(tmp_path / "director")
    a = TomeNetEngine(scenario, seed=4)
    a.start()
    a.turn(bot_command(a))
    a.save(tmp_path / "net")
    shutil.copytree(tmp_path / "net", tmp_path / "torn")
    (tmp_path / "torn" / "brains" / "elara.pt").unlink()
    shutil.copytree(tmp_path / "net", tmp_path / "garbled")
    (tmp_path / "garbled" / "campaign.json").write_text((tmp_path / "net" / "campaign.json").read_text()[:500])
    bn, sj, cortex = ("partitions", "bn"), ("partitions", "sj"), ("partitions", "jax", min(a.brain.jax.state))
    slots = ["director", "torn", "garbled",
             _variant(tmp_path, "old", campaign=_edit("version", value=2)),   # an older slot holds another brain
             _variant(tmp_path, "unnamed", campaign=_edit("player_name")),
             _variant(tmp_path, "stateless", campaign=_edit("state", "lumen")),
             _variant(tmp_path, "dice", campaign=_edit("rng", value=[3, [1, 2], None])),
             _variant(tmp_path, "npc_shape", npc=_edit("w_spines", value=torch.zeros(3, 3))),
             _variant(tmp_path, "npc_choice", npc=_edit("last_decision", value={"tactic": 99})),
             _variant(tmp_path, "npc_junk", npc_bytes=b"not a brain"),
             _variant(tmp_path, "w1", brain=_edit("W", value=1)),
             _variant(tmp_path, "jax_shape", brain=_edit(*cortex, value=torch.zeros(3))),
             _variant(tmp_path, "bn_ghost", brain=_edit(*bn, "tensors", "layer.ghost.v", value=torch.zeros(1, 4))),
             _variant(tmp_path, "bn_shape", brain=_edit(*bn, "tensors", "conn.ctx.bla.w", value=torch.zeros(64, 48))),
             _variant(tmp_path, "bn_cells", brain=_edit(*bn, "tensors", "layer.bla.v", value=torch.zeros(1, 5))),
             _variant(tmp_path, "bn_torn_rule", brain=_edit(*bn, "tensors", "rule.ctx.bla.p_plus")),
             _variant(tmp_path, "bn_scalars", brain=_edit(*bn, "scalars")),
             _variant(tmp_path, "sj_shape", brain=_edit(*sj, "W", "ctx_d1", value=torch.zeros(4, 4))),
             _variant(tmp_path, "sj_keys", brain=_edit(*sj, "elig")),
             _variant(tmp_path, "habit_shape", brain=_edit(*sj, "habit", "v", value=torch.zeros(5))),
             _variant(tmp_path, "clock", brain=_edit("clock", "phase")),
             _variant(tmp_path, "bus", brain=_edit("bus", "buf", "B.cea")),
             _variant(tmp_path, "bus_ahead", brain=_edit("clock", "t_ms", value=10**6)),
             _variant(tmp_path, "mindless", brain=_edit("extra", "mind", "stats")),
             _variant(tmp_path, "late_us", brain=_edit("extra", "late_us", value={"loss": "much"}))]
    a.turn(bot_command(a))                    # play on, so a half-done load would show
    before = _held(a, tmp_path / "before")
    for slot in slots:
        with pytest.raises(SaveError):
            a.load(tmp_path / slot)
        _assert_same(_held(a, tmp_path / f"after_{slot}"), before, slot)
    a.load(tmp_path / "net")                  # the intact slot still loads
    b = TomeNetEngine(scenario, seed=9)
    b.load(tmp_path / "net")
    cmd = bot_command(a)
    assert a.turn(cmd).to_dict(include_debug=True) == b.turn(cmd).to_dict(include_debug=True)


def test_a_load_that_fails_while_the_game_takes_it_in_is_undone(scenario, tmp_path, monkeypatch):
    """Every check passed, but taking the save in still failed halfway (here the basal ganglia, after the cortex
    had loaded): the game takes back exactly what it held."""
    a = TomeNetEngine(scenario, seed=4)
    a.start()
    a.turn(bot_command(a))
    a.save(tmp_path / "slot")
    a.turn(bot_command(a))
    before = _held(a, tmp_path / "before")
    load, calls = a.brain.sj.load_state_dict, []

    def flaky(d):
        load(d)
        calls.append(d)
        if len(calls) == 1:
            raise RuntimeError("the disk hiccuped")

    monkeypatch.setattr(a.brain.sj, "load_state_dict", flaky)
    with pytest.raises(SaveError):
        a.load(tmp_path / "slot")
    assert len(calls) == 2                    # the save's state, then the game's own back
    _assert_same(_held(a, tmp_path / "after"), before)
    b = TomeNetEngine(scenario, seed=9)       # and it plays on as the game it held
    b.load(tmp_path / "before")
    cmd = bot_command(a)
    assert a.turn(cmd).to_dict(include_debug=True) == b.turn(cmd).to_dict(include_debug=True)
    assert a.brain.digest() == b.brain.digest()


def test_the_server_and_the_terminal_report_a_damaged_save_and_play_on(scenario, tmp_path, monkeypatch, capsys):
    """Whatever is wrong with a slot (here an NPC's brain file is not a brain at all), the player is told the load
    failed and the game goes on; nothing crashes."""
    e = TomeEngine(scenario, seed=3)
    e.start()
    e.turn("I search the room")
    e.save(tmp_path / "junk")
    (tmp_path / "junk" / "brains" / "marcus.pt").write_bytes(b"not a brain")
    sent = []
    session = TomeSession(None, None, None, tmp_path, debug=False, brain="director")
    session.engine = e

    async def send(msg):
        sent.append(msg)

    session.send = send
    asyncio.run(session.command("load", {"slot": "junk"}))
    assert sent[-1]["type"] == "error" and sent[-1]["message"].startswith("load failed")
    asyncio.run(session.handle({"type": "input", "text": "I press on"}))
    assert sent[-1]["type"] == "frame"

    lines = iter([f":load {tmp_path / 'junk'}", "I press on", ":q"])
    monkeypatch.setattr("builtins.input", lambda _="": next(lines))
    monkeypatch.setattr("voidlit.app.build_engine", lambda *a, **k: (TomeEngine(scenario, seed=3), False))
    assert cli.main(["play", "--brain", "director"]) == 0
    out = capsys.readouterr().out
    assert "load failed" in out and "Traceback" not in out


def test_an_npc_save_from_before_its_first_step_loads_into_a_used_brain_as_into_a_fresh_one(spec, tmp_path):
    """BindsNET creates the R-STDP traces at an NPC's first step. Loading a save from before it into a brain that
    has perceived since must drop that timeline's traces, or they pay out into its weights at the next reward."""
    a = NPCBrain(spec, seed=6)
    a.save(tmp_path / "fresh.pt")
    for s in (WARM, HOSTILE, WARM):
        a.perceive(s)
    a.load(tmp_path / "fresh.pt")
    b = NPCBrain(spec, seed=99)
    b.load(tmp_path / "fresh.pt")
    for conn in (a.c_spines, a.c_tactics):
        assert not any(hasattr(conn.update_rule, attr) for attr in RULE_STATE)
    for s in (HOSTILE, WARM, HOSTILE):
        assert a.perceive(s).to_dict() == b.perceive(s).to_dict()
    assert torch.equal(a.c_spines.w, b.c_spines.w) and torch.equal(a.c_tactics.w, b.c_tactics.w)


def test_an_npc_brain_keeps_its_last_decision_across_a_save(spec, tmp_path):
    """An NPC beat for an NPC who is not in the scene speaks from its last decision: a load must restore it, or the
    loaded game makes the NPC perceive again (an extra brain step the original game never took)."""
    a = NPCBrain(spec, seed=6)
    a.save(tmp_path / "fresh.pt")
    for s in (WARM, HOSTILE):
        a.perceive(s)
    a.save(tmp_path / "used.pt")
    b = NPCBrain(spec, seed=99)
    b.load(tmp_path / "used.pt")
    assert b.last_decision is not None and b.last_decision.to_dict() == a.last_decision.to_dict()
    assert a.perceive(WARM).to_dict() == b.perceive(WARM).to_dict()
    b.load(tmp_path / "fresh.pt")
    assert b.last_decision is None
    state = torch.load(tmp_path / "used.pt", weights_only=True)
    state["last_decision"]["tactic"] = len(spec.tactics)          # a move this NPC does not have
    with pytest.raises(ValueError):
        b.load_state_dict(state)
    assert b.last_decision is None and b.turn == 0


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
