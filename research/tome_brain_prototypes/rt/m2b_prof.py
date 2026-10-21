import os, sys, time
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
import jaxpath  # noqa
import numpy as np, jax, jax.numpy as jnp
import tome_jax as TJ
from m2_jax import drive
P, meta = TJ.build(seed=0, dt_sub=0.25)
st0 = TJ.init_state(P, meta, jax.random.PRNGKey(0))
ext = jnp.asarray(drive(meta, 200, 1))
for sub, smax in ((4, 192), (4, 32), (1, 192), (2, 64), (1, 32)):
    run = TJ.make_window(P, meta, 200, sub, s_max=smax)
    st, sp = run(st0, ext, 1.0, 0.0, 0.0); jax.block_until_ready(sp)
    ts = []
    for _ in range(5):
        t = time.perf_counter(); st, sp = run(st0, ext, 1.0, 0.0, 0.0); jax.block_until_ready(sp); ts.append(time.perf_counter() - t)
    print(f"sub={sub} s_max={smax}: {min(ts)*1e3:.1f} ms / 200 ms")
