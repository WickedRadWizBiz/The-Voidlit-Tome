import time, os
import jaxpath  # noqa
import jax, jax.numpy as jnp, numpy as np
import column2 as C
run = C.make(0.1, 10)
st, W, idx = C.init(jax.random.PRNGKey(0), 0.1)
def turn(epoch):
    ext = jnp.zeros((epoch, C.N)) + 0.0005
    s, sp = run(st, W, idx, ext); jax.block_until_ready(sp)
    out = []
    for rep in range(3):
        s = st
        t0 = time.perf_counter()
        for _ in range(200 // epoch):
            s, sp = run(s, W, idx, ext)
        jax.block_until_ready(sp)
        out.append((time.perf_counter() - t0) * 1e3)
    return out
print(os.environ.get("XLA_FLAGS"))
for e in (1, 2, 4, 10, 50, 200):
    print(f"epoch {e:3d} ticks: turn ms", [round(x, 1) for x in turn(e)])
