"""Balance report: play N seeded runs with each scripted player and summarise outcomes.

    python tools/balance.py --runs 40
"""

from __future__ import annotations

import argparse
import collections
import warnings

warnings.filterwarnings("ignore")

from voidlit.cortex.director import Director, load_director  # noqa: E402
from voidlit.playtest import bot_command  # noqa: E402
from voidlit.tome import TomeEngine, load_scenario  # noqa: E402


def run(smart: bool, runs: int, max_turns: int = 200, offset: int | None = None) -> str:
    sc, model = load_scenario(), load_director()
    if offset is not None:
        sc.dc_offset = offset
    outcomes, deaths, beats, turns = collections.Counter(), collections.Counter(), collections.Counter(), []
    for seed in range(runs):
        e = TomeEngine(sc, director=Director(model, seed=seed), seed=seed)
        e.start()
        for n in range(1, max_turns + 1):
            f = e.turn(bot_command(e, smart=smart))
            beats[f.debug["beat"]] += 1
            if f.over:
                break
        outcomes["escaped" if e.state.won else "died" if e.state.character.dead else "timeout"] += 1
        if e.state.character.dead:
            deaths[e.state.deck.index] += 1
        turns.append(n)
    rate = outcomes["escaped"] / runs
    return (f"{'smart' if smart else 'naive'}: escaped {rate:.0%} {dict(outcomes)} | avg turns {sum(turns) / runs:.1f}"
            f" | deaths by chapter {dict(sorted(deaths.items()))} | beats {dict(beats.most_common())}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=40)
    p.add_argument("--offset", type=int, default=None, help="override the scenario's global DC offset")
    args = p.parse_args()
    for smart in (True, False):
        print(run(smart, args.runs, offset=args.offset), flush=True)


if __name__ == "__main__":
    main()
