"""Full-turn wall time vs min-delay window D (all cross delays scaled to >= D)."""
import os, sys, time
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
import numpy as np
import cosim as CS
from m3_cosim import feats
for D in (2, 4, 5, 8):
    CS.D_MIN = D
    T = CS.Tome(seed=0, window=D)
    T.delays = {k: max(v, D) for k, v in T.delays.items()}
    T.bn.mon = T.bn.mon  # monitors have time=D_MIN at construction
    f, lum = feats(0); T.set_game_inputs(f, lum)
    T.run_turn(200); T.run_turn(200)
    ts = []
    for k in range(5):
        t = time.perf_counter(); T.run_turn(200); ts.append(time.perf_counter() - t)
    print(f"D={D}: full 200 ms turn median {np.median(ts)*1e3:.0f} ms (min {min(ts)*1e3:.0f})")
