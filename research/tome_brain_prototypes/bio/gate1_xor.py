"""Gate 1 (honest): human L2/3 PC, exponential Euler. Apical current steps X, Y (Gidon 2020 protocol) and
synaptic AMPA+NMDA trains; soma spike counts must implement XOR; no NaN at any dt; tuft peak stays physiological."""
import sys, numpy as np
import jaxpath  # noqa
import jax, jax.numpy as jnp
from cells import build_params, init_state, substep

p = build_params(["pc23h"], g_katp=0.0)

def run(ix, iy, dt, T=120.0, mode="current", gsyn=None):
    s = init_state(p)
    steps = int(T / dt)
    spikes, va_peak, vs_peak, dca = 0, -1e9, -1e9, 0
    step = jax.jit(lambda s, i: substep(p, s, i, dt))
    for k in range(steps):
        t = k * dt
        on = 10.0 <= t < 110.0
        if mode == "current":
            i = jnp.zeros((3, 1)).at[2, 0].set((ix + iy) if on else 0.0)
        else:
            i = jnp.zeros((3, 1))
            if on:   # tonic synaptic conductance: X and Y each add gsyn nS AMPA (+ NMDA 0.5x) onto the tuft
                s = dict(s, gA_a=jnp.full(1, (ix + iy) * gsyn), gN_a=jnp.full(1, 0.5 * (ix + iy) * gsyn))
        s, spk = step(s, i)
        spikes += int(spk[0]); va_peak = max(va_peak, float(s["va"][0])); vs_peak = max(vs_peak, float(s["vs"][0]))
        dca = max(dca, float(s["dca_gain"][0]))
    return spikes, va_peak, dca, any(np.isnan([va_peak, vs_peak]))

for dt in (0.05, 0.25, 1.0):
    print(f"--- dt={dt} ms, apical current steps (pA)")
    for th in (300.0,):
        res = {k: run(*(np.array(k) * th), dt) for k in [(0,0),(1,0),(0,1),(1,1)]}
        for k, (n, vap, g, nan) in res.items():
            print(f"  X,Y={k}: soma spikes={n:2d}  tuft peak={vap:7.2f} mV  dCaAP gain={g:.3f}  nan={nan}")
        xor = res[(1,0)][0] > 0 and res[(0,1)][0] > 0 and res[(0,0)][0] == 0 and res[(1,1)][0] == 0
        print("  XOR:", xor)
print("--- synaptic version, dt=0.25: X and Y = tonic AMPA(+NMDA) conductance on the tuft")
for g in (2.0, 3.0, 4.0, 6.0):
    res = {k: run(*k, 0.25, mode="syn", gsyn=g) for k in [(0,0),(1,0),(0,1),(1,1)]}
    xor = res[(1,0)][0] > 0 and res[(0,1)][0] > 0 and res[(0,0)][0] == 0 and res[(1,1)][0] == 0
    print(f"  g={g} nS: " + "  ".join(f"{k}:{v[0]}sp/{v[1]:.0f}mV/K{v[2]:.2f}" for k, v in res.items()), " XOR:", xor)
