"""Per-partition cost per 200 ms turn and whether JAX async dispatch overlaps the torch partitions."""
import os, time
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
import jaxpath  # noqa
import numpy as np, torch, jax, jax.numpy as jnp
import cosim as CS
from m3_cosim import feats
T = CS.Tome(seed=0, window=4)
f, lum = feats(0); T.set_game_inputs(f, lum); T.run_turn(); T.run_turn()
W = 4; n = 50
N = T.L.n
bus_I = jnp.asarray(np.zeros((W, N), np.float32)); da = jnp.zeros(W); lc = jnp.zeros(W)
# JAX alone (blocking each window)
st = T.jst
t = time.perf_counter()
for _ in range(n):
    st, sp = T.jrun(st, bus_I, 1.0, da, lc); np.asarray(sp)
tj = time.perf_counter() - t
# SJ alone
ctx = torch.zeros(W, 256); lim = torch.zeros(W, 64); mf = torch.zeros(W, 128); dat = torch.zeros(W)
t = time.perf_counter()
with torch.no_grad():
    for _ in range(n):
        T.sj.step_window(ctx, lim, mf, dat)
ts = time.perf_counter() - t
# BN alone
hyp = np.zeros((W, 48), np.float32); c64 = np.zeros((W, 64), np.float32)
t = time.perf_counter()
for _ in range(n):
    T.bn.step_window(c64, hyp, W)
tb = time.perf_counter() - t
# overlap: dispatch JAX then run SJ+BN, then block
t = time.perf_counter()
with torch.no_grad():
    for _ in range(n):
        st, sp = T.jrun(st, bus_I, 1.0, da, lc)
        T.sj.step_window(ctx, lim, mf, dat); T.bn.step_window(c64, hyp, W)
        np.asarray(sp)
to = time.perf_counter() - t
# full turn
ts_full = []
for _ in range(3):
    t = time.perf_counter(); T.run_turn(); ts_full.append(time.perf_counter() - t)
print(f"per 200 ms turn: JAX {tj*1e3:.0f} ms | SJ {ts*1e3:.0f} ms | BN {tb*1e3:.0f} ms | sum {(tj+ts+tb)*1e3:.0f} | overlapped JAX||(SJ+BN) {to*1e3:.0f} ms | full run_turn {min(ts_full)*1e3:.0f} ms")
