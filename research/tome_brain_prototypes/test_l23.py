import time, sys
import jaxpath  # noqa
import jax, jax.numpy as jnp, numpy as np
import l23cell as L

def run(I_x, I_y, T=60.0, dt=0.025, dcaap=True, monotone=False, n=1, g_unit=0.05, onset=5.0, dur=30.0):
    """Two apical pathways X, Y as conductance steps on the tuft (and the matching apical drive)."""
    st = L.init(n)
    steps = int(round(T / dt))
    def body(st, k):
        t = k * dt
        on = (t >= onset) & (t < onset + dur)
        drive = (I_x + I_y) * on
        g_exc = jnp.zeros((n, L.NC)).at[:, 8].set(g_unit * drive)
        st = L.step(st, g_exc, jnp.zeros((n, L.NC)), jnp.full(n, drive), dt, dcaap=dcaap, monotone=monotone)
        return st, (st["V"][:, 0], st["V"][:, 8])
    st, (vs, vt) = jax.lax.scan(body, st, jnp.arange(steps))
    vs, vt = np.asarray(vs[:, 0]), np.asarray(vt[:, 0])
    spikes = int(np.sum((vs[1:] >= 0.0) & (vs[:-1] < 0.0)))
    return vt.max(), vs.max(), spikes, vs

if __name__ == "__main__":
    print("XOR at the somatic output (spike counts) and tuft peak; X, Y each 1.1 x threshold drive")
    for label, kw in (("dCaAP (non-monotone)", {}), ("control: monotone Ca spike", {"monotone": True}),
                      ("control: no dCaAP", {"dcaap": False})):
        res = {}
        for x, y in ((0, 0), (1, 0), (0, 1), (1, 1)):
            vt, vs, sp, _ = run(1.1 * x, 1.1 * y, **kw)
            res[(x, y)] = (round(float(vt), 1), sp)
        xor = (res[(1, 0)][1] > 0 and res[(0, 1)][1] > 0 and res[(0, 0)][1] == 0 and res[(1, 1)][1] == 0)
        print(f"  {label:28s} {res}  XOR={xor}")
    print("\nIntegrator convergence vs dt (single strong somatic-ish drive via basal AMPA), spike counts/peak:")
    def basal(dt, T=200.0):
        st = L.init(1); steps = int(round(T / dt))
        def body(st, k):
            t = k * dt
            g = jnp.zeros((1, L.NC)).at[:, 1:7].set(jnp.where((t > 10) & (t < 190), 0.012, 0.0))
            st = L.step(st, g, jnp.zeros((1, L.NC)), jnp.zeros(1), dt, dcaap=False)
            return st, st["V"][0, 0]
        _, vs = jax.lax.scan(body, st, jnp.arange(steps))
        vs = np.asarray(vs)
        idx = np.nonzero((vs[1:] >= 0) & (vs[:-1] < 0))[0]
        return len(idx), (idx * dt)[:6], np.isfinite(vs).all(), vs.max()
    for dt in (0.005, 0.0125, 0.025, 0.05, 0.1, 0.2):
        n, times, fin, pk = basal(dt)
        print(f"  dt={dt:<6} spikes={n:3d} finite={fin} peak={pk:6.1f} mV first times={np.round(times, 2)}")
