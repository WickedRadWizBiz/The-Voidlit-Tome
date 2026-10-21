"""Co-simulation: per-turn latency, partition breakdown, determinism, lockstep==windowed exactness, save/load."""
import os, sys, time, pickle
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
import numpy as np
import cosim as CS

def feats(turn, seed=3):
    r = np.random.default_rng(seed * 100 + turn)
    return r.random(64).astype(np.float32), 1.0 - 0.1 * turn

def play(tome, turns, start=0, record=True):
    recs, times = [], []
    for k in range(start, start + turns):
        f, lum = feats(k)
        tome.set_game_inputs(f, lum)
        t = time.perf_counter()
        r = tome.run_turn(record=record)
        times.append(time.perf_counter() - t)
        recs.append(r)
    return recs, times

def same(ra, rb):
    return all(np.array_equal(a[k], b[k]) for a, b in zip(ra, rb) for k in a)

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    t = time.perf_counter(); A = CS.Tome(seed=0, window=4); build = time.perf_counter() - t
    t = time.perf_counter(); play(A, 1); first = time.perf_counter() - t      # includes JIT compile
    recs, times = play(A, 6, start=1)
    J = np.concatenate([r["jax"] for r in recs]); V = np.concatenate([r["vavl"] for r in recs])
    L = A.L
    rates = {p: round(float(J[:, L.start[p]:L.start[p] + L.size[p]].mean() * 1000), 1) for p in ("TOM.L23", "PLAN.L5ET", "TC", "CA1")}
    print(f"build {build:.2f}s, first turn (incl. JIT) {first:.2f}s, warm 200 ms turn: median {np.median(times)*1e3:.0f} ms, max {max(times)*1e3:.0f} ms")
    print("rates Hz", rates, "VA/VL Hz", round(float(V.mean() * 1000), 1), "decision", A.readout(200)[0])
    if what in ("all", "det"):
        B = CS.Tome(seed=0, window=4); C = CS.Tome(seed=0, window=4)
        rb, _ = play(B, 4); rc, _ = play(C, 4)
        print("determinism (same seed, 4 turns, every raster bitwise):", same(rb, rc))
    if what in ("all", "lockstep"):
        B = CS.Tome(seed=0, window=4); C = CS.Tome(seed=0, window=1)
        rb, _ = play(B, 2); rc, _ = play(C, 2)
        eq = same(rb, rc)
        print("windowed (D=4) == lockstep (D=1), 2 turns, bitwise:", eq)
        if not eq:
            for k in rb[0]:
                a = np.concatenate([r[k] for r in rb]); b = np.concatenate([r[k] for r in rc])
                diff = np.argwhere(a != b)
                print(f"  {k}: {len(diff)} differing entries; first at tick {diff[0][0] if len(diff) else None}; spikes {a.sum():.0f} vs {b.sum():.0f}")
    if what in ("all", "save"):
        B = CS.Tome(seed=0, window=4)
        play(B, 2)
        blob = pickle.dumps(B.state())
        rb, _ = play(B, 2, start=2)
        C = CS.Tome(seed=5, window=4)           # different seed: everything must come from the save
        C.load(pickle.loads(blob))
        rc, _ = play(C, 2, start=2)
        print(f"save/load mid-run resumes bitwise: {same(rb, rc)} (save blob {len(blob)/1e6:.1f} MB)")
