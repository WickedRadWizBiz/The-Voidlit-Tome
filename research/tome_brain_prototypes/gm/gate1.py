"""Honest Gate 1: XOR emerges from dynamics; a monotone dCaAP (mutation) must fail it; stable across dt."""
import time
import jaxpath  # noqa
import jax, jax.numpy as jnp, numpy as np
import l23gm as L
from test_l23 import xor, run_step_current

def table():
    out = {}
    for dt in (0.025, 0.05, 0.1):
        out[dt] = [(g, xor(0, 0, dt)[0], xor(g, 0, dt)[0], xor(0, g, dt)[0], xor(g, g, dt)[0]) for g in (7, 8)]
    return out
ok = table()
print("calibrated (g, 00, 10, 01, 11 somatic spikes):", ok)
G0, SIG = L.DCA_G0, L.DCA_SIG
L.DCA_SIG = 1e9                       # mutation: amplitude no longer falls with drive (monotone dCaAP)
print("MUTANT monotone dCaAP:", table())
L.DCA_SIG = SIG
L.DCA_GBAR = 0.0                      # mutation: no dCaAP at all
print("MUTANT no dCaAP:", table())
L.DCA_GBAR = 3.0
for dt in (0.025, 0.05, 0.1, 0.2):
    t, vmax, nan = run_step_current(dt, i_amp=150.0)
    print(f"step 150 pA dt={dt}: rate {len(t)*5} Hz, first spike {t[0]:.2f} ms, peak {vmax:.1f} mV, NaN {nan}")

def bench(n, ms=250, dt=0.1):
    s = L.init(n); z = jnp.zeros((n, 4)); steps = int(ms / dt)
    key = jax.random.PRNGKey(0)
    @jax.jit
    def run(s, key):
        def body(c, k):
            s, key = c
            key, sub = jax.random.split(key)
            wa = (jax.random.uniform(sub, (n, 4)) < 0.002) * 2.0
            s, sp = L.step(s, wa, 0.5 * wa, z, jnp.zeros(n), dt)
            return (s, key), sp
        return jax.lax.scan(body, (s, key), jnp.arange(steps))
    jax.block_until_ready(run(s, key))
    t0 = time.perf_counter(); r = run(s, key); jax.block_until_ready(r); t = time.perf_counter() - t0
    return t, float(r[1].sum()) / n / (ms / 1000)
for n in (96, 192, 384, 768):
    t, rate = bench(n)
    print(f"N={n:4d} cells x 4 compartments, 250 ms at dt=0.1 ms (2500 substeps): {t*1000:7.1f} ms wall, mean rate {rate:.1f} Hz")
