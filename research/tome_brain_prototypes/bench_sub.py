import time, sys
import jaxpath  # noqa
import jax, jax.numpy as jnp
import l23cell as L

def bench(n, steps=2000, dt=0.05):
    st = L.init(n)
    ge = jnp.full((n, L.NC), 0.001); gi = jnp.zeros((n, L.NC))
    @jax.jit
    def run(st):
        def body(s, _):
            return L.step(s, ge, gi, jnp.zeros(n), dt), None
        return jax.lax.scan(body, st, None, length=steps)[0]
    jax.block_until_ready(run(st))
    t0 = time.perf_counter(); jax.block_until_ready(run(st)); t = time.perf_counter() - t0
    return t / steps * 1e6

for n in (64, 352, 1056, 4096):
    print(f"n={n:5d} 9-comp cells: {bench(n):7.1f} us per substep")
