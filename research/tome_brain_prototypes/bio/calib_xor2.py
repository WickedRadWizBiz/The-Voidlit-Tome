import numpy as np, itertools
import jaxpath  # noqa
import jax, jax.numpy as jnp
import cells
from cells import build_params, init_state, substep
dt = 0.25
def make(**over):
    cells.KINDS["pc23h"].update(over); return build_params(["pc23h"], g_katp=0.0)
def run(p, i_tuft, T=120.0):
    s = init_state(p); step = jax.jit(lambda s, i: substep(p, s, i, dt))
    n, vap, g = 0, -1e9, 0.0
    for k in range(int(T/dt)):
        t = k*dt
        s, spk = step(s, jnp.zeros((3, 1)).at[2, 0].set(i_tuft if 10 <= t < 110 else 0.0))
        n += int(spk[0]); vap = max(vap, float(s["va"][0])); g = max(g, float(s["dca_gain"][0]))
    return n, vap, g
best = []
for gK, A, g_sa in itertools.product((4.0, 8.0, 12.0), (1000.0, 1600.0, 2400.0), (3.0, 5.0)):
    p = make(g_sa=g_sa, A_dca=A, D_th=1e9, g_Ka=gK)
    lo, hi = 0.0, 3000.0
    for _ in range(16):
        mid = 0.5*(lo+hi); _, _, g = run(p, mid, T=50.0)
        hi, lo = (mid, lo) if g > 0 else (hi, mid)
    rheo = hi
    p = make(g_sa=g_sa, A_dca=A, D_th=rheo, g_Ka=gK)
    r = {k: run(p, sum(k)*1.05*rheo) for k in [(0,0),(1,0),(1,1)]}
    xor = r[(1,0)][0] > 0 and r[(0,0)][0] == 0 and r[(1,1)][0] == 0
    print(f"gK={gK:4.0f} A={A:5.0f} g_sa={g_sa} rheo={rheo:4.0f} | X:{r[(1,0)][0]}sp/{r[(1,0)][1]:.0f}mV/K{r[(1,0)][2]:.2f}  "
          f"X+Y:{r[(1,1)][0]}sp/{r[(1,1)][1]:.0f}mV/K{r[(1,1)][2]:.2f} | XOR {xor}")
