import time, numpy as np
import jaxpath  # noqa
import jax, jax.numpy as jnp
import tome_jax as T

def timeit(label, W=T.W_SYNC):
    T.W_SYNC = W
    st, static = T.build(0)
    win = T.make_window()
    drive = jnp.zeros(T.N)
    for _ in range(3):
        st, sp = win(static, st, drive)
    jax.block_until_ready(sp)
    t0 = time.perf_counter()
    for _ in range(250 // W):
        st, sp = win(static, st, drive)
    jax.block_until_ready(sp)
    print(f"{label:40s} {1e3*(time.perf_counter()-t0):7.1f} ms / 250 ms bio")

timeit("full, W=4")
timeit("full, W=10", 10)
orig_sub = T.SUB
T.SUB = 1; timeit("membrane substeps 1 (dt=1 ms)"); T.SUB = orig_sub
orig = T.GB_IDX; T.GB_IDX = []; timeit("no GB blocks"); T.GB_IDX = orig
origv = T.VG_IDX; T.VG_IDX = []; timeit("no Vogels blocks"); T.VG_IDX = origv
T.K_E, T.K_I = 48, 24; timeit("gather K_E=48, K_I=24")
