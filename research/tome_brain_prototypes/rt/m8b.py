import os, time
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
import numpy as np, torch
import cosim as CS
W = 4
for rep in range(3):
    T = CS.Tome(seed=rep, window=4)
    ctx = torch.zeros(W, 256); lim = torch.zeros(W, 64); mf = torch.zeros(W, 128); da = torch.zeros(W)
    with torch.no_grad():
        for _ in range(20): T.sj.step_window(ctx, lim, mf, da)
        t = time.perf_counter()
        for _ in range(50): T.sj.step_window(ctx, lim, mf, da)
        ts = time.perf_counter() - t
    z64, z48 = np.zeros((W, 64), np.float32), np.zeros((W, 48), np.float32)
    for _ in range(20): T.bn.step_window(z64, z48, W)
    t = time.perf_counter()
    for _ in range(50): T.bn.step_window(z64, z48, W)
    tb = time.perf_counter() - t
    print(f"rep {rep}: SJ {ts*1e3:.0f} ms, BN {tb*1e3:.0f} ms per turn; grad mode {torch.is_grad_enabled()}")
