"""Balance report: play N seeded short trials per class with each scripted player and summarise survival.

    python tools/balance.py --runs 40

The project's playtest simulator (tome_sim.py, final_cfg.json) reports 90-93% short-trial survival by class with a
smart bot. This bot plays through the full app (senses, Director, people and places), and may use allies and gear.
"""

from __future__ import annotations

import argparse
import collections
import warnings

warnings.filterwarnings("ignore")

from voidlit.cortex.director import Director, load_director  # noqa: E402
from voidlit.playtest import bot_command  # noqa: E402
from voidlit.tome import Codex, TomeEngine  # noqa: E402


def run(smart: bool, runs: int, max_turns: int = 400) -> str:
    codex, model = Codex.load(), load_director()
    lines = []
    for cid in sorted(codex.classes):
        outcomes, beats, turns = collections.Counter(), collections.Counter(), []
        for seed in range(runs):
            e = TomeEngine(codex, director=Director(model, seed=seed), seed=seed, class_id=cid)
            e.start()
            n = 0
            for n in range(1, max_turns + 1):
                f = e.turn(bot_command(e, smart=smart))
                beats[f.debug["beat"]] += 1
                if f.over or e.state.stage in ("haven", "fallen"):
                    break
            outcomes["survived" if e.state.stage == "haven" else "died" if e.bearer.dead else "timeout"] += 1
            turns.append(n)
        lines.append(f"  {cid:<10} survived {outcomes['survived'] / runs:4.0%}  {dict(outcomes)}  "
                     f"avg turns {sum(turns) / runs:.1f}  beats {dict(beats.most_common(3))}")
    return f"{'smart' if smart else 'naive'} bot, short trials:\n" + "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=40)
    args = p.parse_args()
    for smart in (True, False):
        print(run(smart, args.runs))


if __name__ == "__main__":
    main()
