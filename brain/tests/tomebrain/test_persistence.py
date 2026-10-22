"""Saves reload and continue bit-exactly, a save is written all or nothing, and a refused save changes nothing.

Also: the network's choices reach the game as it made them. The fact it volunteers is the one whose P slot won,
and a stimulus the rules set after the race reaches the amygdala at the next Phase A.
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")

import voidlit  # noqa: E402
from voidlit import cli  # noqa: E402
from voidlit.limbic.npc_brain import NPCBrain  # noqa: E402
from voidlit.playtest import bot_command  # noqa: E402
from voidlit.senses import SensoryState  # noqa: E402
from voidlit.server import TomeSession  # noqa: E402
from voidlit.tome.engine import SaveError, TomeEngine, Turn  # noqa: E402
from voidlit.tome.net_engine import BRAIN_FILE, TomeNetEngine  # noqa: E402
from voidlit.tomebrain import persist  # noqa: E402
from voidlit.tomebrain.brain import Commit, PhaseBOut  # noqa: E402

WARM = SensoryState(rapport=0.8, leverage=0.3)
HOSTILE = SensoryState(aggression=0.9, exposure=0.8, rapport=-0.7)
RULE_STATE = ("p_plus", "p_minus", "eligibility", "eligibility_trace")   # what MSTDPET makes at its first step
DROP = object()
NAN, INF = float("nan"), float("inf")


def _flat(obj, key: str = "") -> dict:
    """A nested save blob as {path: leaf}, tensors as arrays, so two can be compared leaf by leaf."""
    if isinstance(obj, dict):
        return {k2: v2 for k, v in obj.items() for k2, v2 in _flat(v, f"{key}/{k}").items()}
    if isinstance(obj, (list, tuple)):
        return {k2: v2 for i, v in enumerate(obj) for k2, v2 in _flat(v, f"{key}/{i}").items()}
    return {key: obj.numpy() if isinstance(obj, torch.Tensor) else obj}


def _held(e: TomeEngine, path) -> dict:
    """Everything a load replaces, as a save writes it (but the save id each save stamps anew): the campaign, every
    NPC brain, and the Tome's brain (all partitions, the bus and the clock, with its turn and mind)."""
    e.save(path)
    campaign = json.loads((path / "campaign.json").read_text())
    campaign.pop("save_id")
    out = {"campaign": campaign}
    for nid in e.brains:
        out.update(_flat(torch.load(path / "brains" / f"{nid}.pt", weights_only=True)["brain"], f"npc.{nid}"))
    if (path / BRAIN_FILE).exists():
        blob = torch.load(path / BRAIN_FILE, weights_only=True)
        blob["extra"].pop("save_id")
        out.update(_flat(blob, "brain"))
    return out


def _files(d: Path) -> dict[str, bytes]:
    """Every file under ``d``, byte for byte."""
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


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


def _variant(root, name: str, *, campaign=None, brain=None, npc=None, npc_bytes=None, mixed=None) -> str:
    """A copy of the slot ``root/net`` with its campaign, its brain file or Elara's brain state changed, or with the
    file ``mixed`` taken from the slot ``root/later`` (the same game, saved a turn on)."""
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
    elara = d / "brains" / "elara.pt"
    if npc is not None:
        blob = torch.load(elara, weights_only=True)
        npc(blob["brain"])
        torch.save(blob, elara)
    if npc_bytes is not None:
        elara.write_bytes(npc_bytes)
    if mixed is not None:
        shutil.copy(root / "later" / mixed, d / mixed)
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
    """A slot that does not fit, is damaged anywhere (its campaign, an NPC's brain, any partition of the Tome's
    brain, its bus, clock or mind), holds a value of the wrong type, a NaN or an infinity, an obstacle, clock or item
    the scenario does not author, or is put together from files of different saves, is refused with SaveError before
    anything changes."""
    d = TomeEngine(scenario, seed=3)
    d.start()
    d.turn("I search the room")
    d.save(tmp_path / "director")
    a = TomeNetEngine(scenario, seed=4)
    a.start()
    a.turn(bot_command(a))
    elara = a.brains["elara"]
    assert elara.last_decision is not None and elara.meta[0] is not None      # (the variants below edit them)
    a.save(tmp_path / "net")
    a.turn(bot_command(a))                    # play on, so a half-done load would show
    a.save(tmp_path / "later")
    shutil.copytree(tmp_path / "net", tmp_path / "torn")
    (tmp_path / "torn" / "brains" / "elara.pt").unlink()
    shutil.copytree(tmp_path / "net", tmp_path / "garbled")
    (tmp_path / "garbled" / "campaign.json").write_text((tmp_path / "net" / "campaign.json").read_text()[:500])
    bn, sj, cortex = ("partitions", "bn"), ("partitions", "sj"), ("partitions", "jax", min(a.brain.jax.state))
    voltage = min(k for k, v in a.brain.jax.state.items() if np.asarray(v).dtype.kind == "f")
    ghoul = {**scenario.obstacles[min(scenario.obstacles)].to_dict(), "id": "ghoul"}
    luck = {**ghoul, "id": min(scenario.obstacles), "pips": ["luck"]}

    def spent(item_uses):                     # the first item that runs out
        return lambda blob: next(i for i in blob["state"]["character"]["inventory"] if i["uses"] is not None) \
            .update(uses=item_uses)
    slots = ["director", "torn", "garbled",
             _variant(tmp_path, "old", campaign=_edit("version", value=2)),   # an older slot holds another brain
             _variant(tmp_path, "unnamed", campaign=_edit("player_name")),
             _variant(tmp_path, "stateless", campaign=_edit("state", "lumen")),
             # GameState fields with defaults: only the key-set check refuses a campaign without them
             _variant(tmp_path, "historyless", campaign=_edit("state", "history")),
             _variant(tmp_path, "untwisted", campaign=_edit("state", "twisted")),
             _variant(tmp_path, "dice", campaign=_edit("rng", value=[3, [1, 2], None])),
             # values of the wrong type, each of which a turn would crash on or silently misread
             _variant(tmp_path, "turn_text", campaign=_edit("state", "turn", value="21")),
             _variant(tmp_path, "history_null", campaign=_edit("state", "history", value=None)),
             _variant(tmp_path, "tension_text", campaign=_edit("state", "tension", value="high")),
             _variant(tmp_path, "clues_null", campaign=_edit("state", "clues_revealed", value=None)),
             _variant(tmp_path, "pips_text", campaign=_edit("state", "obstacle_pips0", value="abc")),
             _variant(tmp_path, "lose_turn_text", campaign=_edit("state", "lose_turn", value="yes")),
             _variant(tmp_path, "outcome_number", campaign=_edit("state", "last_outcome", value=7)),
             _variant(tmp_path, "item_uses_text", campaign=_edit("state", "character", "inventory", 0, "uses",
                                                                 value="two")),
             _variant(tmp_path, "deck_behind", campaign=_edit("state", "deck", "index", value=-1)),
             _variant(tmp_path, "voiceless", campaign=_edit("voice_rng", value=None)),
             _variant(tmp_path, "unknown_fact", campaign=_edit("ledger", "disclosed", value=["no.such.fact"])),
             # NaN and infinity: of the right type, but no turn can work with them
             _variant(tmp_path, "tension_nan", campaign=_edit("state", "tension", value=NAN)),
             _variant(tmp_path, "gauss_inf", campaign=_edit("rng", 2, value=-INF)),
             # authored records: the scenario's own, found by id, never ones a save makes up
             _variant(tmp_path, "unknown_obstacle", campaign=lambda b: b["state"]["obstacles"].append(ghoul)),
             _variant(tmp_path, "pip_luck", campaign=lambda b: b["state"]["obstacles"].append(luck)),
             _variant(tmp_path, "clock_overfull", campaign=_edit("state", "clocks", 0, "filled", value=99)),
             _variant(tmp_path, "clocks_swapped", campaign=lambda b: b["state"]["clocks"].reverse()),
             _variant(tmp_path, "unknown_item", campaign=_edit("state", "character", "inventory", 0, "name",
                                                               value="Railgun")),
             _variant(tmp_path, "item_refilled", campaign=spent(99)),
             _variant(tmp_path, "item_endless", campaign=spent(None)),
             _variant(tmp_path, "unknown_background", campaign=_edit("state", "character", "background",
                                                                     value="Pirate")),
             _variant(tmp_path, "npc_shape", npc=_edit("w_spines", value=torch.zeros(3, 3))),
             _variant(tmp_path, "npc_choice", npc=_edit("last_decision", value={"tactic": 99})),
             _variant(tmp_path, "npc_volume", npc=_edit("meta", 0, "volume", value="big")),
             _variant(tmp_path, "npc_plot_float", npc=_edit("meta", 0, "target_plot", value=1.5)),
             _variant(tmp_path, "npc_events", npc=_edit("last_decision", "events", value=None)),
             _variant(tmp_path, "npc_baseline_bool", npc=_edit("reward_baseline", value=True)),
             _variant(tmp_path, "npc_baseline_nan", npc=_edit("reward_baseline", value=NAN)),
             _variant(tmp_path, "npc_volume_inf", npc=_edit("meta", 0, "volume", value=INF)),
             _variant(tmp_path, "npc_weights_nan", npc=lambda s: s["w_spines"].fill_(NAN)),
             _variant(tmp_path, "npc_junk", npc_bytes=b"not a brain"),
             _variant(tmp_path, "w1", brain=_edit("W", value=1)),
             _variant(tmp_path, "jax_shape", brain=_edit(*cortex, value=torch.zeros(3))),
             _variant(tmp_path, "jax_nan", brain=lambda b: b["partitions"]["jax"][voltage].fill_(NAN)),
             _variant(tmp_path, "bn_ghost", brain=_edit(*bn, "tensors", "layer.ghost.v", value=torch.zeros(1, 4))),
             _variant(tmp_path, "bn_shape", brain=_edit(*bn, "tensors", "conn.ctx.bla.w", value=torch.zeros(64, 48))),
             _variant(tmp_path, "bn_cells", brain=_edit(*bn, "tensors", "layer.bla.v", value=torch.zeros(1, 5))),
             _variant(tmp_path, "bn_torn_rule", brain=_edit(*bn, "tensors", "rule.ctx.bla.p_plus")),
             _variant(tmp_path, "bn_scalars", brain=_edit(*bn, "scalars")),
             _variant(tmp_path, "bn_seed_float", brain=_edit(*bn, "streams", "seed", value=4.7)),
             _variant(tmp_path, "sj_shape", brain=_edit(*sj, "W", "ctx_d1", value=torch.zeros(4, 4))),
             _variant(tmp_path, "sj_keys", brain=_edit(*sj, "elig")),
             _variant(tmp_path, "sj_scalar", brain=_edit(*sj, "scalars", "kappa_h")),
             _variant(tmp_path, "sj_ticks_float", brain=_edit(*sj, "scalars", "nm_ticks", value=torch.tensor(4.5))),
             _variant(tmp_path, "sj_phase_int", brain=_edit(*sj, "scalars", "in_phase_b", value=torch.tensor(1))),
             _variant(tmp_path, "brain_v1", brain=_edit("version", value=1)),       # an earlier milestone's brain
             _variant(tmp_path, "habit_shape", brain=_edit(*sj, "habit", "v", value=torch.zeros(5))),
             _variant(tmp_path, "clock", brain=_edit("clock", "phase")),
             _variant(tmp_path, "bus", brain=_edit("bus", "buf", "B.cea")),
             _variant(tmp_path, "bus_ahead", brain=_edit("clock", "t_ms", value=10**6)),
             _variant(tmp_path, "mindless", brain=_edit("extra", "mind", "stats")),
             # mind keys only older saves lacked: this version has no defaults for them
             _variant(tmp_path, "mind_affect", brain=_edit("extra", "mind", "affect")),
             _variant(tmp_path, "mind_vs_ema", brain=_edit("extra", "mind", "vs_ema")),
             _variant(tmp_path, "mind_need_text", brain=_edit("extra", "mind", "needs", "survival", value="0.5")),
             _variant(tmp_path, "late_us", brain=_edit("extra", "late_us", value={"loss": "much"})),
             _variant(tmp_path, "fallbacks_int", brain=_edit("extra", "fallbacks", value=3)),   # (one count, as before)
             _variant(tmp_path, "fallbacks_loop", brain=_edit("extra", "fallbacks", "FRAME")),
             _variant(tmp_path, "mind_need_nan", brain=_edit("extra", "mind", "needs", "survival", value=NAN)),
             _variant(tmp_path, "mind_vs_ema_inf", brain=_edit("extra", "mind", "vs_ema", value=INF)),
             _variant(tmp_path, "mind_stats_nan", brain=lambda b: next(iter(b["extra"]["mind"]["stats"].values()))
                      .__setitem__(1, NAN)),
             _variant(tmp_path, "late_us_nan", brain=_edit("extra", "late_us", value={"loss": NAN})),
             # torn or mixed: every file valid on its own, but not all from one save
             _variant(tmp_path, "unstamped", campaign=_edit("save_id")),
             _variant(tmp_path, "mixed_npc", mixed="brains/marcus.pt"),
             _variant(tmp_path, "mixed_brain", mixed=BRAIN_FILE),
             _variant(tmp_path, "mixed_campaign", mixed="campaign.json")]
    # each is refused for its own reason, so no check can hide behind another
    reasons = {
        "director": BRAIN_FILE, "torn": "elara.pt", "garbled": "Expecting property name", "old": "engine version",
        "unnamed": r"campaign: keys differ: \['player_name'\]", "stateless": r"state: keys differ: \['lumen'\]",
        "historyless": r"state: keys differ: \['history'\]", "untwisted": r"state: keys differ: \['twisted'\]",
        "dice": "state vector", "turn_text": r"state\.turn: str, not int",
        "history_null": r"state\.history: NoneType, not a list", "tension_text": r"state\.tension: str, not float",
        "clues_null": r"state\.clues_revealed: NoneType, not a list",
        "pips_text": r"state\.obstacle_pips0: str, not a dict", "lose_turn_text": r"state\.lose_turn: str, not bool",
        "outcome_number": r"state\.last_outcome: int, not str",
        "item_uses_text": r"inventory\[0\]\.uses: str, not int", "deck_behind": "game does not fit this scenario",
        "voiceless": "voice does not fit", "unknown_fact": "game does not fit this scenario",
        "tension_nan": r"state\.tension: not a finite number", "gauss_inf": r"rng\[2\]: not a finite number",
        "unknown_obstacle": "obstacles do not fit", "pip_luck": "obstacles do not fit",
        "clock_overfull": "clocks do not fit", "clocks_swapped": "clocks do not fit",
        "unknown_item": "character does not fit", "item_refilled": "character does not fit",
        "item_endless": "character does not fit", "unknown_background": "character does not fit",
        "npc_baseline_nan": "reward_baseline: not a finite number",
        "npc_volume_inf": r"meta\[0\]\.volume: not a finite number", "npc_weights_nan": "w_spines is not finite",
        "jax_nan": re.escape(f"JAX state.{voltage}: not finite"),
        "mind_need_nan": r"needs\.survival: not a finite number",
        "mind_vs_ema_inf": "vs_ema: not a finite number", "mind_stats_nan": r"stats\..*\[1\]: not a finite number",
        "late_us_nan": r"late_us\.loss: not a finite number",
        "npc_shape": "w_spines is not a float tensor", "npc_choice": "last_decision: keys differ",
        "npc_volume": r"meta\[0\]\.volume: str, not float", "npc_plot_float": r"meta\[0\]\.target_plot: float, not int",
        "npc_events": r"last_decision\.events: NoneType, not a list",
        "npc_baseline_bool": "reward_baseline: bool, not float", "npc_junk": "",     # (in torch's own words)
        "w1": "1-ms window", "jax_shape": r"of shape \(3,\), not numbers",
        "bn_ghost": r"amygdala\.tensors: keys differ: \['layer\.ghost\.v'\]",
        "bn_shape": r"conn\.ctx\.bla\.w: float32 of shape \(64, 48\)",
        "bn_cells": r"layer\.bla\.v: float32 of shape \(1, 5\)", "bn_torn_rule": "learning traces are missing",
        "bn_scalars": r"amygdala: keys differ: \['scalars'\]", "bn_seed_float": "noise streams",
        "sj_shape": r"W\.ctx_d1: float32 of shape \(4, 4\)", "sj_keys": r"basal ganglia: keys differ: \['elig'\]",
        "sj_scalar": r"scalars: keys differ: \['kappa_h'\]", "sj_ticks_float": r"scalars\.nm_ticks: float32, not int64",
        "sj_phase_int": r"scalars\.in_phase_b: int64, not bool", "brain_v1": "saved by an earlier version of the Tome",
        "habit_shape": r"habit\.v: float32 of shape \(5,\)", "clock": r"clock: keys differ: \['phase'\]",
        "bus": r"bus\.buf: keys differ: \['B\.cea'\]", "bus_ahead": "the bus and the clock disagree",
        "mindless": r"mind: keys differ: \['stats'\]", "mind_affect": r"mind: keys differ: \['affect'\]",
        "mind_vs_ema": r"mind: keys differ: \['vs_ema'\]", "mind_need_text": r"needs\.survival: str, not float",
        "late_us": r"late_us\.loss: str, not float", "unstamped": "save id: NoneType",
        "fallbacks_int": r"fallbacks: int, not a record", "fallbacks_loop": r"fallbacks: keys differ: \['FRAME'\]",
        "mixed_npc": "marcus.pt belongs to another save", "mixed_brain": f"{BRAIN_FILE} belongs to another save",
        "mixed_campaign": "belongs to another save than campaign.json",
    }
    assert reasons.keys() == set(slots)
    before = _held(a, tmp_path / "before")
    for slot in slots:
        with pytest.raises(SaveError, match=reasons[slot]):
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
    assert len(calls) == 3                    # the save's state, then the brain's own back, then the game's
    _assert_same(_held(a, tmp_path / "after"), before)
    b = TomeNetEngine(scenario, seed=9)       # and it plays on as the game it held
    b.load(tmp_path / "before")
    cmd = bot_command(a)
    assert a.turn(cmd).to_dict(include_debug=True) == b.turn(cmd).to_dict(include_debug=True)
    assert a.brain.digest() == b.brain.digest()


def test_a_load_whose_undo_fails_too_is_a_save_error_and_closes_the_run(scenario, tmp_path, monkeypatch):
    """Should even taking back what the game held fail (a bug, not a bad save), the player is still told the load
    failed, and the run, which can no longer go on safely, is closed."""
    a = TomeNetEngine(scenario, seed=4)
    a.start()
    a.turn(bot_command(a))
    a.save(tmp_path / "slot")

    def broken(d):
        raise RuntimeError("the basal ganglia are broken")

    monkeypatch.setattr(a.brain.sj, "load_state_dict", broken)
    with pytest.raises(SaveError, match="could not be restored"):
        a.load(tmp_path / "slot")
    frame = a.turn("I press on")
    assert frame.over and frame.text == "The Tome is closed. Begin a new run."


def test_a_brain_load_that_fails_halfway_is_undone(scenario, tmp_path, monkeypatch):
    """``persist.load``, which the G7 and G24 gates call on the brain alone, is all or nothing too: should the
    amygdala fail to take a save after the cortex and the basal ganglia took theirs, the brain takes back what it
    held, and the error is raised."""
    a = TomeNetEngine(scenario, seed=4)
    a.start()
    a.turn(bot_command(a))
    persist.save(a.brain.cosim, tmp_path / "brain.pt")
    a.turn(bot_command(a))
    digest = a.brain.digest()
    load, calls = a.brain.bn.load_state_dict, []

    def flaky(d):
        calls.append(d)
        if len(calls) == 1:
            raise RuntimeError("the disk hiccuped")
        load(d)

    monkeypatch.setattr(a.brain.bn, "load_state_dict", flaky)
    with pytest.raises(RuntimeError, match="hiccuped"):
        persist.load(a.brain.cosim, tmp_path / "brain.pt")
    assert len(calls) == 2 and a.brain.digest() == digest       # the save's state, then the brain's own back


def test_a_save_that_fails_leaves_the_slot_as_it_was(scenario, tmp_path, monkeypatch):
    """A save is written whole beside the old slot and only then takes its place. One that fails (the disk fills up
    while the Tome's brain is written, after the campaign and the NPCs; or the last rename fails) raises SaveError and
    leaves the old slot exactly as it was, and nothing else behind. It still loads, as the game it was."""
    a = TomeNetEngine(scenario, seed=4)
    a.start()
    a.turn(bot_command(a))
    a.save(tmp_path / "slot")
    kept = _files(tmp_path / "slot")
    b = TomeNetEngine(scenario, seed=9)
    b.load(tmp_path / "slot")
    a.turn(bot_command(a))

    def disk_full(*args, **kw):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(persist, "save", disk_full)
    with pytest.raises(SaveError, match="No space left"):
        a.save(tmp_path / "slot")
    monkeypatch.undo()
    assert _files(tmp_path / "slot") == kept and os.listdir(tmp_path) == ["slot"]

    replace, into_slot = os.replace, []

    def flaky(src, dst):                    # the new slot cannot take the old one's place; putting it back can
        if Path(dst).name == "slot":
            into_slot.append(src)
            if len(into_slot) == 1:
                raise OSError("the disk hiccuped")
        replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises(SaveError, match="hiccuped"):
        a.save(tmp_path / "slot")
    monkeypatch.undo()
    assert _files(tmp_path / "slot") == kept and os.listdir(tmp_path) == ["slot"]

    c = TomeNetEngine(scenario, seed=7)
    c.load(tmp_path / "slot")
    cmd = bot_command(b)
    assert b.turn(cmd).to_dict(include_debug=True) == c.turn(cmd).to_dict(include_debug=True)


def test_a_save_never_replaces_a_directory_that_is_not_a_slot(scenario, tmp_path):
    """Saving replaces a whole slot, so it refuses to replace a directory without campaign.json that is not empty,
    even one holding only files a slot also holds."""
    e = TomeEngine(scenario, seed=3)
    e.start()
    (tmp_path / "letters").mkdir()
    (tmp_path / "letters" / "dear.txt").write_text("keep me")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / BRAIN_FILE).write_bytes(b"weights")
    for folder, kept in (("letters", {"dear.txt": b"keep me"}), ("models", {BRAIN_FILE: b"weights"})):
        with pytest.raises(SaveError, match="not a save slot"):
            e.save(tmp_path / folder)
        assert _files(tmp_path / folder) == kept
    e.save(tmp_path / "letters" / "slot")         # a slot inside it is fine, and so is saving over that slot
    e.save(tmp_path / "letters" / "slot")
    assert sorted(os.listdir(tmp_path / "letters")) == ["dear.txt", "slot"]
    # made as any folder is, not private
    assert (tmp_path / "letters" / "slot").stat().st_mode == (tmp_path / "letters").stat().st_mode


def test_a_save_replaces_a_slot_whole_with_whatever_else_lies_in_it(scenario, tmp_path):
    """A directory holding campaign.json is a slot, and so is an empty one: a save replaces it whole, with anything the
    system or an editor left in it (a .DS_Store, an editor's backup), and does not refuse it."""
    e = TomeEngine(scenario, seed=3)
    e.start()
    (tmp_path / "empty").mkdir()
    e.save(tmp_path / "empty")
    e.save(tmp_path / "slot")
    files = set(_files(tmp_path / "slot"))
    for stray in (".DS_Store", "campaign.json~"):
        (tmp_path / "slot" / stray).write_text("left behind")
    e.load(tmp_path / "slot")
    e.turn("I search the room")
    e.save(tmp_path / "slot")
    assert set(_files(tmp_path / "slot")) == files and sorted(os.listdir(tmp_path)) == ["empty", "slot"]
    b = TomeEngine(scenario, seed=30)
    b.load(tmp_path / "slot")
    cmd = bot_command(e)
    assert e.turn(cmd).to_dict(include_debug=True) == b.turn(cmd).to_dict(include_debug=True)


# A save cut off by the process dying (os._exit: no handler runs), while the new slot is written or between setting
# the old slot aside and putting the new one, copied from the slot ``src``, in its place.
CRASH = """
import os, shutil, sys
from pathlib import Path
from voidlit import saves

slot, src, cut = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), sys.argv[3]
replace = os.replace

def write(d, save_id):
    shutil.copy(src / "campaign.json", d / "campaign.json")
    if cut == "writing":
        os._exit(9)
    shutil.copytree(src / "brains", d / "brains")

def die_into_slot(a, b):
    if Path(b) == slot and Path(a).name.endswith(".new"):
        os._exit(9)
    replace(a, b)

os.replace = die_into_slot
saves.write_slot(slot, write)
"""


def _crash(slot: Path, src: Path, cut: str) -> None:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(Path(voidlit.__file__).parents[1]),
                                                        os.environ.get("PYTHONPATH", "")])}
    assert subprocess.run([sys.executable, "-c", CRASH, str(slot), str(src), cut], env=env, check=False).returncode == 9


@pytest.mark.parametrize("cut", ["writing", "renaming"])
def test_a_save_cut_off_by_a_crash_never_loses_the_slot(scenario, tmp_path, cut):
    """The process dies in the middle of a save. While the new slot was still being written, the old one is the
    slot; once the old slot was set aside (the new one is whole on the disk by then), the new one is. The next load
    takes that slot, and clears what the save left behind."""
    a = TomeEngine(scenario, seed=6)
    a.start()
    a.turn(bot_command(a))
    a.save(tmp_path / "slot")
    a.save(tmp_path / "before")
    a.turn(bot_command(a))
    a.save(tmp_path / "after")
    _crash(tmp_path / "slot", tmp_path / "after", cut)
    assert (tmp_path / "slot").exists() == (cut == "writing") and len(os.listdir(tmp_path)) > 3
    b, want = TomeEngine(scenario, seed=60), TomeEngine(scenario, seed=61)
    b.load(tmp_path / "slot")
    want.load(tmp_path / ("before" if cut == "writing" else "after"))
    assert sorted(os.listdir(tmp_path)) == ["after", "before", "slot"]
    cmd = bot_command(want)
    assert b.turn(cmd).to_dict(include_debug=True) == want.turn(cmd).to_dict(include_debug=True)


def test_a_save_after_a_crash_puts_the_slot_back_first(scenario, tmp_path, monkeypatch):
    """A save after a crash between the two renames takes the slot back before anything else, so even should it fail
    itself, the slot is there, and nothing else is left behind."""
    a = TomeEngine(scenario, seed=6)
    a.start()
    a.save(tmp_path / "slot")
    a.turn(bot_command(a))
    a.save(tmp_path / "after")
    _crash(tmp_path / "slot", tmp_path / "after", "renaming")
    kept = _files(tmp_path / "after")

    def disk_full(*args, **kw):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(torch, "save", disk_full)
    with pytest.raises(SaveError, match="No space left"):
        a.save(tmp_path / "slot")
    monkeypatch.undo()
    assert sorted(os.listdir(tmp_path)) == ["after", "slot"] and _files(tmp_path / "slot") == kept
    a.save(tmp_path / "slot")
    assert sorted(os.listdir(tmp_path)) == ["after", "slot"]


def test_text_in_a_save_never_reaches_the_narration(scenario, tmp_path):
    """A save holds its obstacles, clocks and items whole, but the game takes each one's names, texts and other
    authored values from the scenario, by id ("authored only"): a save whose copies were rewritten by hand plays on
    exactly as the one it was made from, and none of what was written into it is ever shown."""
    a = TomeEngine(scenario, seed=35)
    a.start()
    while not a.state.obstacles:
        a.turn(bot_command(a))
    a.save(tmp_path / "slot")
    shutil.copytree(tmp_path / "slot", tmp_path / "rewritten")
    blob = json.loads((tmp_path / "rewritten" / "campaign.json").read_text())
    st, said = blob["state"], "Elara is the one who killed them all."
    for o in st["obstacles"]:
        o.update(name=said, text=said, defeat_text=said, kind="horror", harm=9, blocking=False)
    for c in st["clocks"]:
        c.update(name=said, segments=99, consequence="lumen")
    for i in st["character"]["inventory"]:
        i.update(tags=["light"], bonus={"grit": 9}, light=9)
    (tmp_path / "rewritten" / "campaign.json").write_text(json.dumps(blob))
    b, c = TomeEngine(scenario, seed=77), TomeEngine(scenario, seed=78)
    b.load(tmp_path / "slot")
    c.load(tmp_path / "rewritten")
    assert c._campaign() == b._campaign()
    for _ in range(12):
        cmd = bot_command(b)
        fb, fc = b.turn(cmd).to_dict(include_debug=True), c.turn(cmd).to_dict(include_debug=True)
        assert fc == fb and said not in json.dumps(fc)


def test_the_server_and_the_terminal_report_a_failed_load_or_save_and_play_on(scenario, tmp_path, monkeypatch,
                                                                              capsys):
    """Whatever is wrong with a slot (here an NPC's brain file is not a brain at all), or whatever stops a save (here
    a file where the save folder should be), the player is told the load or the save failed and the game goes on;
    nothing crashes."""
    e = TomeEngine(scenario, seed=3)
    e.start()
    e.turn("I search the room")
    e.save(tmp_path / "junk")
    (tmp_path / "junk" / "brains" / "marcus.pt").write_bytes(b"not a brain")
    (tmp_path / "blocked").write_text("a file, not a folder")
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
    session.save_root = tmp_path / "blocked"
    asyncio.run(session.command("save", {"slot": "quicksave"}))
    assert sent[-1]["type"] == "error" and sent[-1]["message"].startswith("save failed")
    asyncio.run(session.handle({"type": "input", "text": "I press on"}))
    assert sent[-1]["type"] == "frame"

    lines = iter([f":load {tmp_path / 'junk'}", "I press on", f":save {tmp_path / 'blocked' / 'slot'}", "I press on",
                  ":q"])
    monkeypatch.setattr("builtins.input", lambda _="": next(lines))
    monkeypatch.setattr("voidlit.app.build_engine", lambda *a, **k: (TomeEngine(scenario, seed=3), False))
    assert cli.main(["play", "--brain", "director"]) == 0
    out = capsys.readouterr().out
    assert "load failed" in out and "save failed" in out and "Traceback" not in out


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


def test_an_npc_brain_state_is_checked_whole_before_any_of_it_loads(spec):
    """The check refuses what loading would only trip over later, or not at all: a generator state no generator
    takes, and branch or decision fields of the wrong type (which crash a later turn or silently misread)."""
    a = NPCBrain(spec, seed=6)
    a.perceive(WARM)
    edits = (lambda s: s.update(gen_state=torch.zeros_like(s["gen_state"])),
             lambda s: s["meta"][0].update(volume="big"),
             lambda s: s["meta"][0].update(trauma=1),
             lambda s: s["last_decision"].update(momentum="high"),
             lambda s: s["last_decision"].update(reward="0.5"))
    for edit in edits:
        state = a.state_dict()
        a.check_state_dict(state)
        edit(state)
        with pytest.raises(ValueError):
            a.check_state_dict(state)


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
