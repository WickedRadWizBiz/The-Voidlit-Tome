import os, time
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
import jaxpath  # noqa
import numpy as np, jax, jax.numpy as jnp
import tome_jax as TJ
from m2_jax import drive
P, meta = TJ.build(seed=0, dt_sub=1.0)
st0 = TJ.init_state(P, meta, jax.random.PRNGKey(0))
ext = jnp.asarray(drive(meta, 200, 1))
def t(flags, sub=1, smax=32):
    run = TJ.make_window(P, meta, 200, sub, s_max=smax, flags=flags)
    st, sp = run(st0, ext, 1.0, 0.0, 0.0); jax.block_until_ready(sp)
    ts = []
    for _ in range(5):
        a = time.perf_counter(); st, sp = run(st0, ext, 1.0, 0.0, 0.0); jax.block_until_ready(sp); ts.append(time.perf_counter() - a)
    return min(ts) * 1e3
allon = {}
print("all on", round(t({}), 1))
for k in ("gb", "astro", "events", "membrane"):
    print("without", k, round(t({k: False}), 1))
print("only membrane", round(t(dict(gb=False, astro=False, events=False)), 1))
print("nothing", round(t(dict(gb=False, astro=False, events=False, membrane=False)), 1))
