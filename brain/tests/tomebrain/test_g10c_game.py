"""G10c: pacing survives in play. The network-run Tome and the classic Director, same rules and the same smart bot,
40 seeds: escape rates within 10 points and every beat's share within 0.04. (Long: run with -m balance.)"""

import collections

import pytest

pytest.importorskip("jax")

from voidlit.cortex.director import Director, load_director  # noqa: E402
from voidlit.cortex.features import BEATS  # noqa: E402
from voidlit.playtest import bot_command  # noqa: E402
from voidlit.tome import TomeEngine, load_scenario  # noqa: E402
from voidlit.tome.net_engine import TomeNetEngine  # noqa: E402


def arm(net: bool, seeds: int = 40, max_turns: int = 150):
    sc, model = load_scenario(), load_director()
    escapes, beats = 0, collections.Counter()
    for seed in range(seeds):
        cls = TomeNetEngine if net else TomeEngine
        e = cls(sc, director=Director(model, seed=seed), seed=seed)
        e.start()
        for _ in range(max_turns):
            f = e.turn(bot_command(e, smart=True))
            beats[f.debug["beat"]] += 1
            if f.over:
                break
        escapes += e.state.won
    total = sum(beats.values())
    return escapes / seeds, {b.value: beats[b.value] / total for b in BEATS}


@pytest.mark.balance
def test_g10c_pacing_survives_in_play():
    net_rate, net_beats = arm(True)
    dir_rate, dir_beats = arm(False)
    assert abs(net_rate - dir_rate) <= 0.10, (net_rate, dir_rate)
    for b in net_beats:
        assert abs(net_beats[b] - dir_beats[b]) <= 0.04, (b, net_beats, dir_beats)
