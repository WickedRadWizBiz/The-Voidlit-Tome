import time, sys
import jaxpath  # noqa
import jax, jax.numpy as jnp, numpy as np
import l23gm as L

def run_step_current(dt, i_amp=200.0, T=300.0):
    n = 1
    s = L.init(n)
    steps = int(T / dt)
    z = jnp.zeros((n, 4))
    @jax.jit
    def go(s):
        def body(s, k):
            t = k * dt
            i = jnp.where((t >= 20.0) & (t < 220.0), i_amp, 0.0) * jnp.ones(n)
            s, sp = L.step(s, z, z, z, i, dt)
            return s, (s["v"][0, 0], sp[0])
        return jax.lax.scan(body, s, jnp.arange(steps))
    s, (v, sp) = go(s)
    v = np.asarray(v); sp = np.asarray(sp)
    t_sp = np.nonzero(sp)[0] * dt
    return t_sp, np.nanmax(v), np.isnan(v).any()

ref = None
for dt in (0.01, 0.025, 0.05, 0.1, 0.2):
    t_sp, vmax, nan = run_step_current(dt)
    if ref is None: ref = t_sp
    k = min(len(ref), len(t_sp))
    err = np.abs(ref[:k] - t_sp[:k]).max() if k else float('nan')
    print(f"dt={dt:5.3f} spikes={len(t_sp):3d} peak={vmax:6.1f} mV nan={nan} first={t_sp[:3]} max|dt_spike - ref|={err:.3f} ms")

def xor(gx, gy, dt=0.05, T=60.0, gn_ratio=0.5):
    n = 1
    s = L.init(n)
    steps = int(T / dt)
    k_on = int(10.0 / dt)
    def body(s, k):
        g = jnp.where(k == k_on, gx + gy, 0.0)
        wa = jnp.zeros((n, 4)).at[:, 2].set(g)
        wn = jnp.zeros((n, 4)).at[:, 2].set(gn_ratio * g)
        s, sp = L.step(s, wa, wn, jnp.zeros((n, 4)), jnp.zeros(n), dt)
        return s, (s["v"][0, 0], s["v"][0, 2], sp[0], s["ca_amp"][0])
    s, (vs, vt, sp, amp) = jax.jit(lambda s: jax.lax.scan(body, s, jnp.arange(steps)))(s)
    return int(np.asarray(sp).sum()), float(np.max(vs)), float(np.max(vt)), float(np.max(amp))

print("XOR sweep (per-pathway AMPA conductance, NMDA = 0.5x):")
for g in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
    for dt in (0.05, 0.1):
        r10 = xor(g, 0.0, dt); r11 = xor(g, g, dt); r00 = xor(0.0, 0.0, dt)
        print(f" g={g:.1f} dt={dt}: 00 spikes={r00[0]} | 10 spikes={r10[0]} Vs={r10[1]:6.1f} Vtrunk={r10[2]:6.1f} dCa={r10[3]:.2f} | 11 spikes={r11[0]} Vs={r11[1]:6.1f} Vtrunk={r11[2]:6.1f} dCa={r11[3]:.2f}")
