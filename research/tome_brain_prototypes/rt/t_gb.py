"""Graupner-Brunel gate on the network's own update function (tome_jax.gb_drho), with the network's calcium model
(pre transient delayed D, C_pre = 1, C_post = 2, tau_Ca = 20 ms, 1 ms Euler as in ms_step)."""
import jaxpath  # noqa
import numpy as np, jax, jax.numpy as jnp
from jax import lax
import tome_jax as TJ
P = TJ.GB
k = np.exp(-1 / P["tauCa"])
def pairing(freq, dt_pp, n_pairs=60, rho0=0.5):
    """Pairs at `freq` Hz; dt_pp = t_post - t_pre (ms). Then 10 s of silence. Returns final rho."""
    T = int(n_pairs * 1000 / freq) + 10000
    pre = np.zeros(T, np.float32); post = np.zeros(T, np.float32)
    for i in range(n_pairs):
        t = int(i * 1000 / freq) + 50
        pre[t] = 1; post[t + dt_pp] = 1
    D = int(round(P["D"]))
    def step(carry, x):
        rho, cpre, cpost = carry
        pre_d, po = x
        cpre = cpre * k + P["Cpre"] * pre_d
        cpost = cpost * k + P["Cpost"] * po
        rho = jnp.clip(rho + TJ.gb_drho(rho, cpre + cpost), 0, 1)
        return (rho, cpre, cpost), None
    pre_d = np.concatenate([np.zeros(D, np.float32), pre[:-D]])
    (rho, _, _), _ = lax.scan(step, (jnp.float32(rho0), 0.0, 0.0), (pre_d, post))
    return float(rho)
def drift(rho0, secs=600):
    def step(r, _):
        return jnp.clip(r + 1000.0 * TJ.gb_drho(r, 0.0), 0, 1), None   # 1 s steps, no calcium: bistable drift
    r, _ = lax.scan(step, jnp.float32(rho0), None, length=secs)
    return float(r)
print("bistability (no calcium, 10 min):", {r0: round(drift(r0), 3) for r0 in (0.45, 0.48, 0.52, 0.55)})
for f in (1, 5, 10, 20, 30, 50):
    print(f"{f:2d} Hz x60 pairs: pre-post +10 ms -> rho {pairing(f, 10):.3f} | post-pre -10 ms -> rho {pairing(f, -10):.3f}")
