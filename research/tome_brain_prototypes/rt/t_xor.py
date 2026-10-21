"""Honest dCaAP XOR probe: the network's own integrator, one MC cell, apical pathway volleys X, Y (20 ms each)."""
import os, sys
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
import jaxpath  # noqa
import numpy as np, jax, jax.numpy as jnp
import tome_jax as TJ

def probe(P, meta, run1, gX, gY, cell=0, T=60, on=20):
    st = TJ.init_state(P, meta, jax.random.PRNGKey(0))
    N = meta["L"].n
    st = dict(st, V=jnp.full(N, -72.0, jnp.float32))
    base = (cell * 3 + 2) * 4
    ext = jnp.full((1, N), -400.0, jnp.float32).at[:, cell].set(0.0)
    spikes, vmax = 0, -100.0
    for t in range(T):
        if 5 <= t < 5 + on:
            g = (gX + gY) * 0.4     # per-ms increment of a pathway's AMPA conductance (tau 2 ms -> mean ~ g)
            st["G"] = st["G"].at[base + 0].add(g).at[base + 1].add(0.1 * g)
        st, sp = run1(st, ext, 1.0, 0.0, 0.0)
        spikes += int(np.asarray(sp)[0, cell])
        vmax = max(vmax, float(st["Vd"][cell, 1]))
    return spikes, round(vmax, 1)

if __name__ == "__main__":
    sub = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    uh = float(sys.argv[2]) if len(sys.argv) > 2 else 26.0
    gsa = float(os.environ.get('GSA', '5')); tj = float(os.environ.get('TJ', '12'))
    units = [float(x) for x in sys.argv[3:]] or [8.0, 12.0, 16.0, 22.0]
    P, meta = TJ.build(seed=0, dt_sub=1.0 / sub)
    for gd in (20.0, 0.0):
        run1 = TJ.make_window(P, meta, 1, sub, s_max=8, flags=dict(events=False), dca=dict(gd=gd, u_half=uh, u_k=1.5, g_sa=gsa, tau_j=tj, tau_u=float(os.environ.get('TU','1')), tau_h=float(os.environ.get('TH','0.5'))))
        for unit in units:
            r = [probe(P, meta, run1, x * unit, y * unit) for x, y in ((0, 0), (1, 0), (0, 1), (1, 1))]
            s = [x[0] for x in r]
            ok = s[0] == 0 and s[1] > 0 and s[2] > 0 and s[3] == 0
            print(f"gd={gd:4.1f} u_half={uh} unit={unit:4.1f}  spikes {s} Va_peak {[x[1] for x in r]}  XOR={'yes' if ok else 'no'}")
