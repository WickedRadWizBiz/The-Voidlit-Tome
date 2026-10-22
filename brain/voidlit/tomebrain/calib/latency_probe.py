"""Per-turn latency of the network-run game (gate G17: p95 <= 500 ms) and what it is made of (design §17).

    python -m voidlit.tomebrain.calib.latency_probe                     # seeds 1-5, 30 turns each, the smart bot
    python -m voidlit.tomebrain.calib.latency_probe --seeds 1 2 --turns 10

It plays TomeNetEngine games and prints the median, p90 and p95 wall time per turn, the length of Phase B, each loop's
fallback rate and commit times, and each part's wall ms per brain-ms. Wall times mean something only on an otherwise
idle machine: the load average is printed with them, and a warning when it was above 1.5 before the probe started (the
probe itself adds about 1.5: XLA computes beside the torch partitions).
"""

from __future__ import annotations

import argparse
import collections
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from ..brain import CAP_MS
from ..sjpart.basal_ganglia import LOOPS

BUSY_LOAD = 1.5
PARTS = ("jax_dispatch", "jax_join", "sj", "bn", "bus")


def play(seeds=(1, 2, 3, 4, 5), turns: int = 30) -> dict:
    """Play ``turns`` turns of each seed's game with the smart bot. Returns one row per turn that ran Phase B (wall ms,
    the brain's timing, Phase B's length, each loop's commit ms, None for a fallback), each game's cold start, and the
    first game's save. A game's first turn compiles its network (about 2 s), so it counts as the cold start (G17's
    own clause), not as a turn."""
    from ...playtest import bot_command
    from ...tome.net_engine import TomeNetEngine

    rows, cold, save = [], [], None
    for seed in seeds:
        t = time.perf_counter()
        e = TomeNetEngine(seed=seed, timing=True)
        e.start()
        built = time.perf_counter() - t
        for i in range(turns):
            if e.state.over:
                break
            cmd = bot_command(e)
            t = time.perf_counter()
            f = e.turn(cmd)
            wall = (time.perf_counter() - t) * 1000.0
            if i == 0:
                cold.append(built + wall / 1000.0)
            elif "commit_ms" in f.debug:
                rows.append({"seed": seed, "turn": i, "wall": wall, "timing": f.debug["timing"],
                             "phase_b": f.debug["phase_b_ms"], "commit": f.debug["commit_ms"]})
        if save is None:
            with tempfile.TemporaryDirectory() as d:
                t = time.perf_counter()
                e.save(Path(d) / "slot")
                save = {"ms": (time.perf_counter() - t) * 1000.0,
                        "mb": sum(p.stat().st_size for p in (Path(d) / "slot").rglob("*") if p.is_file()) / 1e6}
    return {"rows": rows, "cold_s": cold, "save": save}


def _q(a, qs=(50, 90, 95)) -> str:
    a = np.asarray(a, float)
    return " ".join(f"p{q} {np.percentile(a, q):.0f}" for q in qs) if len(a) else "-"


def report(res: dict, load: tuple[float, float]) -> str:
    rows = res["rows"]
    wall = np.array([r["wall"] for r in rows])
    capped = np.array([r["phase_b"] >= CAP_MS for r in rows])
    lines = [f"load average {load[0]:.2f} before, {load[1]:.2f} after"
             + ("  ** busy machine: these wall times are not a measurement **" if load[0] > BUSY_LOAD else ""),
             f"{len(rows)} turns over {len({r['seed'] for r in rows})} games",
             f"turn wall ms: median {np.median(wall):.0f}  p90 {np.percentile(wall, 90):.0f}  "
             f"p95 {np.percentile(wall, 95):.0f}  max {wall.max():.0f}",
             f"  Phase B at the {CAP_MS}-ms cap: {capped.mean():.0%} of turns, wall {_q(wall[capped], (50,))}; "
             f"the others: wall {_q(wall[~capped], (50,))}"]
    hist = collections.Counter(r["phase_b"] for r in rows)
    lines.append("Phase B ms: " + " ".join(f"{ms}:{n}" for ms, n in sorted(hist.items())))
    for loop in LOOPS:
        at = [r["commit"][loop] for r in rows]
        done = [a for a in at if a is not None]
        lines.append(f"{loop:9s} fallback {sum(a is None for a in at) / len(at):6.1%}   commit ms {_q(done)}")
    brain_ms = np.array([r["timing"]["brain_ms"] for r in rows])
    split = {k: np.array([r["timing"].get(k, 0.0) for r in rows]) for k in ("phase_a", "phase_b", "quiet", *PARTS)}
    rules = wall - split["phase_a"] - split["phase_b"] - split["quiet"]
    lines.append("wall ms per turn (median): " + "  ".join(
        f"{k} {np.median(v):.1f}" for k, v in (*split.items(), ("rules+voice", rules))))
    lines.append("wall ms per brain-ms (median): " + "  ".join(
        f"{k} {np.median(split[k] / brain_ms):.3f}" for k in PARTS))
    lines.append(f"cold start (engine built, first turn played) median {np.median(res['cold_s']):.2f} s, max "
                 f"{max(res['cold_s']):.2f} s; save {res['save']['mb']:.2f} MB in {res['save']['ms']:.0f} ms")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--turns", type=int, default=30)
    args = ap.parse_args(argv)
    before = os.getloadavg()[0]
    res = play(tuple(args.seeds), args.turns)
    print(report(res, (before, os.getloadavg()[0])))


if __name__ == "__main__":
    main()
