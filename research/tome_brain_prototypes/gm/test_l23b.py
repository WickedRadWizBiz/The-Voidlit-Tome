import jaxpath  # noqa
import jax, jax.numpy as jnp, numpy as np
import l23gm as L
from test_l23 import run_step_current, xor
for i in (25, 50, 75, 100, 150):
    t_sp, vmax, nan = run_step_current(0.05, i_amp=i)
    isi = np.diff(t_sp)
    print(f"I={i:4d} pA: {len(t_sp)} spikes in 200 ms; ISIs(ms) {np.round(isi[:6],1)}")
for g in (6, 8, 10, 12, 14, 16, 20, 24):
    r10 = xor(g, 0.0, 0.05); r11 = xor(g, g, 0.05)
    r10b = xor(g, 0.0, 0.1); r11b = xor(g, g, 0.1)
    print(f" g={g:2d}: X alone spikes={r10[0]} Vtr={r10[2]:6.1f} dCa={r10[3]:.2f} | X+Y spikes={r11[0]} Vtr={r11[2]:6.1f} dCa={r11[3]:.2f} | dt0.1: {r10b[0]}/{r11b[0]}")
