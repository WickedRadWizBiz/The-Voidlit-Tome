import jaxpath  # noqa
import jax, jax.numpy as jnp, numpy as np
import l23cell as L
from functools import partial

@partial(jax.jit, static_argnames=("steps", "dcaap", "monotone"))
def sim(drive_amp, g_unit, g_dcaap, steps, dt=0.025, dcaap=True, monotone=False, onset=5.0, dur=30.0):
    st = L.init(1)
    def body(st, k):
        t = k * dt
        on = (t >= onset) & (t < onset + dur)
        drive = drive_amp * on
        g_exc = jnp.zeros((1, L.NC)).at[:, 8].set(g_unit * drive)
        st = L.step(st, g_exc, jnp.zeros((1, L.NC)), jnp.full(1, drive), dt, dcaap=dcaap, monotone=monotone,
                    g_dcaap=g_dcaap)
        return st, (st["V"][0, 0], st["V"][0, 8])
    _, (vs, vt) = jax.lax.scan(body, st, jnp.arange(steps))
    return vs, vt

def summary(vs, vt):
    vs, vt = np.asarray(vs), np.asarray(vt)
    return float(vt.max()), int(np.sum((vs[1:] >= 0) & (vs[:-1] < 0))), float(vs.max())

steps = int(60 / 0.025)
print("passive tuft drive (no dCaAP): g_unit -> (tuft peak, soma spikes, soma peak) for drive 1.1 and 2.2")
for gu in (0.002, 0.004, 0.006, 0.008, 0.012):
    a = summary(*sim(1.1, gu, 0.0, steps, dcaap=False)); b = summary(*sim(2.2, gu, 0.0, steps, dcaap=False))
    print(f"  g_unit={gu}: single {np.round(a,1)}  both {np.round(b,1)}")
print("dCaAP gain sweep at g_unit=0.008: (tuft peak, soma spikes, soma peak)")
for gd in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
    out = {}
    for x, y in ((0, 0), (1, 0), (1, 1)):
        out[(x, y)] = np.round(summary(*sim(1.1 * (x + y), 0.008, gd, steps)), 1).tolist()
    mono = np.round(summary(*sim(2.2, 0.008, gd, steps, monotone=True)), 1).tolist()
    print(f"  g_dCaAP={gd}: {out}  monotone(1,1)={mono}")
