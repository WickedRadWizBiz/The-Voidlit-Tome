"""Honest stability gate on the real JAX partition: perturb one neuron by 0.01 mV, compare 200 ms rasters.
Balanced spiking networks are microscopically chaotic (London et al. 2010): expect spike-time divergence, but
population rates must stay within a few percent, and nothing may run away or go NaN."""
import os
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
import jaxpath  # noqa
import numpy as np, jax, jax.numpy as jnp
import tome_jax as TJ
P, meta = TJ.build(seed=0, dt_sub=0.5)
L = meta["L"]
run = TJ.make_window(P, meta, 200, 2, s_max=64, flags=dict(bg_inside=True))
st = TJ.init_state(P, meta, jax.random.PRNGKey(0))
z = jnp.zeros((200, L.n), jnp.float32)
for _ in range(3):
    st, _ = run(st, z, 1.0, 0.0, 0.0)                     # settle 600 ms
import sys
kick = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
st2 = dict(st, V=st["V"].at[L.start["TOM.L23"]:L.start["TOM.L23"] + (8 if kick > 5 else 1)].add(kick))
a_st, A = run(st, z, 1.0, 0.0, 0.0); b_st, B = run(st2, z, 1.0, 0.0, 0.0)
A, B = np.asarray(A), np.asarray(B)
c_st, C = run(st, z, 1.0, 0.0, 0.0); C = np.asarray(C)
first = np.argmax((A != B).any(1)) if (A != B).any() else None
pops = ["TOM.L23", "PLAN.L5ET", "SENSE.L4", "PV", "TC", "CA3T"]
def rate(X, p):
    if "." not in p and p not in L.start:
        idx = np.concatenate([np.arange(L.start[q], L.start[q] + L.size[q]) for q in L.names if q.endswith("." + p)])
    else:
        idx = np.arange(L.start[p], L.start[p] + L.size[p])
    return X[:, idx].mean() * 1000
print(f"unperturbed rerun identical: {np.array_equal(A, C)}")
print(f"{kick} mV perturbation: rasters diverge from tick {first}; differing spikes {int((A != B).sum())} of {int(A.sum())} "
      f"({(A != B).sum() / max(1, A.sum()):.0%})")
print("population rates Hz (unperturbed vs perturbed):", {p: (round(rate(A, p), 2), round(rate(B, p), 2)) for p in pops})
print("max |V|", float(jnp.abs(b_st["V"]).max()), "NaN:", bool(jnp.isnan(b_st["V"]).any()))
