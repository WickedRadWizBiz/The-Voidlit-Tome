import numpy as np, itertools
import jaxpath  # noqa
import jax, jax.numpy as jnp
import cells
from cells import build_params, init_state, substep
dt = 0.25

def make(**over):
    cells.KINDS["pc23h"].update(over)
    return build_params(["pc23h"], g_katp=0.0)

def run(p, i_tuft, T=120.0, nodca=False):
    s = init_state(p); step = jax.jit(lambda s, i: substep(p, s, i, dt))
    n, vap, g = 0, -1e9, 0.0
    for k in range(int(T/dt)):
        t = k*dt
        i = jnp.zeros((3, 1)).at[2, 0].set(i_tuft if 10 <= t < 110 else 0.0)
        s, spk = step(s, i)
        if nodca: s = dict(s, dca_r=s["dca_r"]*0, dca_d=s["dca_d"]*0)
        n += int(spk[0]); vap = max(vap, float(s["va"][0])); g = max(g, float(s["dca_gain"][0]))
    return n, vap, g

for g_sa, A in itertools.product((3.0, 4.0, 5.0), (700.0, 1000.0)):
    p = make(g_sa=g_sa, A_dca=A, D_th=1e9)   # D_th huge -> gain 1 at any drive: find rheobase of the tuft
    lo, hi = 0.0, 2000.0
    for _ in range(18):
        mid = 0.5*(lo+hi)
        _, vap, g = run(p, mid, T=60.0)
        if g > 0: hi = mid
        else: lo = mid
    rheo = hi
    p = make(g_sa=g_sa, A_dca=A, D_th=rheo)
    res = {k: run(p, sum(k)*1.15*rheo) for k in [(0,0),(1,0),(0,1),(1,1)]}
    passive = run(p, 2.3*rheo, nodca=True)
    xor = res[(1,0)][0] > 0 and res[(0,1)][0] > 0 and res[(0,0)][0] == 0 and res[(1,1)][0] == 0
    print(f"g_sa={g_sa} A={A} rheo={rheo:.0f}pA | " + " ".join(f"{k}:{v[0]}sp/{v[1]:.0f}mV/K{v[2]:.2f}" for k,v in res.items())
          + f" | passive 2.3x: {passive[0]}sp/{passive[1]:.0f}mV | XOR {xor}")
