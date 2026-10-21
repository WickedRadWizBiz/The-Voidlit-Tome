"""Gate 1 (honest): the human L2/3 cell computes XOR in its apical tuft (Gidon 2020), through the real integrator.
Checks: XOR on soma spike counts for current steps AND for synaptic AMPA+NMDA conductance input; identical verdict at
dt = 0.05, 0.25 and 1.0 ms; no NaN; tuft peak below +40 mV; and a mutation check (dCaAP gain forced to 1 = monotonic
Ca2+ spike, as in rodent L5) must BREAK the XOR, proving the test depends on the dCaAP nonlinearity."""
import numpy as np
import jaxpath  # noqa
import jax, jax.numpy as jnp
import cells
from cells import build_params, init_state, substep

def sim(p, amp, dt, mode, T=120.0):
    steps = int(T / dt)
    t = jnp.arange(steps) * dt
    on = ((t >= 10) & (t < 110)).astype(jnp.float32)
    def body(s, on_k):
        if mode == "current":
            i = jnp.zeros((3, 1)).at[2, 0].set(amp * on_k)
        else:
            i = jnp.zeros((3, 1))
            s = dict(s, gA_a=jnp.where(on_k > 0, jnp.full(1, amp), s["gA_a"]),
                     gN_a=jnp.where(on_k > 0, jnp.full(1, 0.5 * amp), s["gN_a"]))
        s, spk = substep(p, s, i, dt)
        return s, (spk[0], s["va"][0])
    s, (spk, va) = jax.lax.scan(body, init_state(p), on)
    return int(spk.sum()), float(va.max()), bool(jnp.isnan(va).any())

def verdict(p, unit, dt, mode):
    r = {k: sim(p, sum(k) * unit, dt, mode) for k in [(0, 0), (1, 0), (0, 1), (1, 1)]}
    ok = r[(1, 0)][0] > 0 and r[(0, 1)][0] > 0 and r[(0, 0)][0] == 0 and r[(1, 1)][0] == 0
    return ok, r

p = build_params(["pc23h"], g_katp=0.0)
for mode, unit in (("current", 1.05 * 270.0), ("synaptic", None)):
    if mode == "synaptic":   # find the smallest tonic tuft conductance that triggers a dCaAP, use 1.05x
        lo, hi = 0.0, 40.0
        for _ in range(20):
            mid = (lo + hi) / 2
            s = sim(p, mid, 0.25, "synaptic"); hi, lo = (mid, lo) if s[1] > -38 else (hi, mid)
        unit = 1.05 * hi
    for dt in (0.05, 0.25, 1.0):
        ok, r = verdict(p, unit, dt, mode)
        print(f"{mode:9s} unit={unit:6.1f} dt={dt:4.2f}: " + "  ".join(f"{k}:{v[0]}sp,{v[1]:.0f}mV" for k, v in r.items())
              + f"  nan={any(v[2] for v in r.values())}  XOR={ok}")
# mutation: monotonic Ca2+ spike (no dCaAP decay with drive) must break XOR
cells.KINDS["pc23h"]["tau_k"] = 1e9
pm = build_params(["pc23h"], g_katp=0.0)
ok, r = verdict(pm, 1.05 * 270.0, 0.25, "current")
print("MUTANT (monotonic dendritic spike):", "  ".join(f"{k}:{v[0]}sp" for k, v in r.items()), " XOR=", ok, "(must be False)")
