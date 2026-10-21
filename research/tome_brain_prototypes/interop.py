import time, torch, numpy as np
import jax, jax.numpy as jnp
print("jax", jax.__version__, jax.devices())
t = torch.arange(12, dtype=torch.float32).reshape(3,4)
j = jax.dlpack.from_dlpack(t)          # torch -> jax
t2 = torch.from_dlpack(j)               # jax -> torch
print("zero-copy torch->jax->torch shares memory:", t2.data_ptr() == t.data_ptr())
x = jnp.ones((4096,), jnp.float32)
@jax.jit
def f(a): return a*2.0
y = f(x); y.block_until_ready()
n=2000; t0=time.perf_counter()
for _ in range(n):
    yt = torch.from_dlpack(f(x))
print("jit call + dlpack round trip us:", round((time.perf_counter()-t0)/n*1e6,1))
t0=time.perf_counter()
for _ in range(n):
    jj = jax.dlpack.from_dlpack(torch.zeros(4096))
print("torch->jax dlpack us:", round((time.perf_counter()-t0)/n*1e6,1))
