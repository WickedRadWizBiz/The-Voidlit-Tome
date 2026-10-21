import os, time
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
import jaxpath  # noqa
import numpy as np, torch, jax, jax.numpy as jnp
from concurrent.futures import ThreadPoolExecutor
import cosim as CS
T = CS.Tome(seed=0, window=4)
N = T.L.n; W = 4
bus_I = jnp.zeros((W, N)); z = jnp.zeros(W)
st = T.jst
st, sp = T.jrun(st, bus_I, 1.0, z, z); np.asarray(sp)
t = time.perf_counter(); st, sp = T.jrun(st, bus_I, 1.0, z, z); t_disp = time.perf_counter() - t
np.asarray(sp); t_all = time.perf_counter() - t
print(f"JAX dispatch returns after {t_disp*1e3:.2f} ms; result ready after {t_all*1e3:.2f} ms")
def torch_work():
    ctx = torch.zeros(W, 256); lim = torch.zeros(W, 64); mf = torch.zeros(W, 128); da = torch.zeros(W)
    with torch.no_grad():
        T.sj.step_window(ctx, lim, mf, da)
    T.bn.step_window(np.zeros((W, 64), np.float32), np.zeros((W, 48), np.float32), W)
for mode in ("async", "thread"):
    ex = ThreadPoolExecutor(1)
    t = time.perf_counter()
    for _ in range(50):
        if mode == "async":
            st, sp = T.jrun(st, bus_I, 1.0, z, z)
            torch_work(); np.asarray(sp)
        else:
            fut = ex.submit(lambda s: (lambda r: (r[0], np.asarray(r[1])))(T.jrun(s, bus_I, 1.0, z, z)), st)
            torch_work(); st, sp = fut.result()
    print(f"{mode}: 50 windows (one 200 ms turn) {(time.perf_counter()-t)*1e3:.0f} ms")
t = time.perf_counter()
for _ in range(50): torch_work()
print(f"torch partitions alone {(time.perf_counter()-t)*1e3:.0f} ms")
