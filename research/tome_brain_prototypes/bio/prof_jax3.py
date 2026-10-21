import time, sys
import jaxpath  # noqa
import jax, jax.numpy as jnp
import tome_jax as T
def timeit(label, abl=()):
    T.ABL = set(abl)
    st, static = T.build(0); win = T.make_window(); drive = jnp.zeros(T.N)
    for _ in range(3): st, sp = win(static, st, drive)
    jax.block_until_ready(sp); t0 = time.perf_counter()
    for _ in range(250 // T.W_SYNC): st, sp = win(static, st, drive)
    jax.block_until_ready(sp); print(f"{label:30s} {1e3*(time.perf_counter()-t0):7.1f} ms / 250 ms bio", flush=True)
timeit("full")
timeit("no rng", ["rng"])
timeit("no long gather", ["long"])
