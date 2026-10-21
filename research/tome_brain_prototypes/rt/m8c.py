import os, sys, time
os.environ["XLA_FLAGS"] = os.environ.get("XF", "--xla_cpu_multi_thread_eigen=false")
import jaxpath  # noqa
import numpy as np, torch, jax, jax.numpy as jnp
import cosim as CS
T = CS.Tome(seed=0, window=4)
N = T.L.n; W = 4
bus_I = jnp.zeros((W, N)); z = jnp.zeros(W)
st = T.jst
def torch_work():
    with torch.no_grad():
        T.sj.step_window(torch.zeros(W, 256), torch.zeros(W, 64), torch.zeros(W, 128), torch.zeros(W))
    T.bn.step_window(np.zeros((W, 64), np.float32), np.zeros((W, 48), np.float32), W)
for _ in range(10): torch_work()
t = time.perf_counter()
for _ in range(50): torch_work()
a = time.perf_counter() - t
for _ in range(20):
    st, sp = T.jrun(st, bus_I, 1.0, z, z); np.asarray(sp)
t = time.perf_counter()
for _ in range(50): torch_work()
b = time.perf_counter() - t
t = time.perf_counter()
for _ in range(50):
    st, sp = T.jrun(st, bus_I, 1.0, z, z)
    torch_work(); np.asarray(sp)
c = time.perf_counter() - t
print(f"XLA_FLAGS={os.environ['XLA_FLAGS']!r}: torch before any JAX {a*1e3:.0f} ms | torch right after JAX {b*1e3:.0f} ms | overlapped turn {c*1e3:.0f} ms | threads {len(os.listdir('/proc/self/task'))}")
