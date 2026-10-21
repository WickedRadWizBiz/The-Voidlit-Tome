import time
import jaxpath  # noqa
import jax, jax.numpy as jnp, numpy as np
import column2 as C
run = C.make(0.1, 10)
st, W, idx = C.init(jax.random.PRNGKey(0), 0.1)
ext2 = jnp.zeros((2, C.N)) + 0.0005
ext1 = jnp.zeros((1, C.N)) + 0.0005
for ext, name in ((ext1, "1 tick"), (ext2, "2 ticks")):
    s, sp = run(st, W, idx, ext); jax.block_until_ready(sp)
    ts = []
    for _ in range(20):
        t0 = time.perf_counter(); s, sp = run(s, W, idx, ext); jax.block_until_ready(sp); ts.append(time.perf_counter() - t0)
    print(name, "per call ms: min %.2f median %.2f" % (min(ts) * 1e3, sorted(ts)[10] * 1e3))
# Does a scan of 200 cost the same per tick?
ext200 = jnp.zeros((200, C.N)) + 0.0005
s, sp = run(st, W, idx, ext200); jax.block_until_ready(sp)
t0 = time.perf_counter(); s, sp = run(s, W, idx, ext200); jax.block_until_ready(sp)
print("200 ticks per call ms: %.1f" % ((time.perf_counter() - t0) * 1e3))
# closure variant: W, idx baked in as constants
run_c = jax.jit(lambda st, ext: run(st, W, idx, ext))
s, sp = run_c(st, ext2); jax.block_until_ready(sp)
ts = []
for _ in range(20):
    t0 = time.perf_counter(); s, sp = run_c(s, ext2); jax.block_until_ready(sp); ts.append(time.perf_counter() - t0)
print("closure, 2 ticks per call ms: min %.2f" % (min(ts) * 1e3))
