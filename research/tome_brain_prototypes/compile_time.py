import time
t0 = time.perf_counter()
import jaxpath  # noqa
import jax, jax.numpy as jnp
import column2 as C
t1 = time.perf_counter()
run = C.make(0.1, 10)
st, W, idx = C.init(jax.random.PRNGKey(0), 0.1)
ext = jnp.zeros((2, C.N))
t2 = time.perf_counter()
s, sp = run(st, W, idx, ext); jax.block_until_ready(sp)
t3 = time.perf_counter()
print(f"import jax {t1-t0:.2f}s, init {t2-t1:.2f}s, first call (trace+compile+run) {t3-t2:.2f}s")
