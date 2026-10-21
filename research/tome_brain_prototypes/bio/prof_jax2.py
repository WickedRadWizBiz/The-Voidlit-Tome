import time, numpy as np
import jaxpath  # noqa
import jax, jax.numpy as jnp
import tome_jax as T
def timeit(label):
    st, static = T.build(0); win = T.make_window(); drive = jnp.zeros(T.N)
    for _ in range(3): st, sp = win(static, st, drive)
    jax.block_until_ready(sp); t0 = time.perf_counter()
    for _ in range(250 // T.W_SYNC): st, sp = win(static, st, drive)
    jax.block_until_ready(sp); print(f"{label:40s} {1e3*(time.perf_counter()-t0):7.1f} ms / 250 ms bio")
g, gi, v, vi = T.GB_BLOCKS, T.GB_IDX, T.VG_BLOCKS, T.VG_IDX
T.GB_BLOCKS, T.GB_IDX = [], []; timeit("no GB blocks"); T.GB_BLOCKS, T.GB_IDX = g, gi
T.VG_BLOCKS, T.VG_IDX = [], []; timeit("no Vogels blocks"); T.VG_BLOCKS, T.VG_IDX = v, vi
T.K_E, T.K_I = 48, 24; timeit("gather K_E=48, K_I=24"); T.K_E, T.K_I = 96, 48
T.GB_BLOCKS, T.GB_IDX, T.VG_BLOCKS, T.VG_IDX = [], [], [], []; T.SUB = 1; timeit("no plastic blocks, SUB=1 (floor)")
