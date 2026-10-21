import time, numpy as np, jaxenv, jax, jax.numpy as jnp
from core_var import build, make_window, init_state, WIN_MS, DMAX, NCOL
t0 = time.perf_counter(); net = build(0); print("N =", net["N"], "build s", round(time.perf_counter()-t0, 2))
print("sparse groups:", len(net["sparse"]), "nnz:", sum(int((W != 0).sum()) for *_, W in net["sparse"]),
      "local nnz:", int((net["Wl"] != 0).sum()), "intercol nnz:", int((net["Wc"] != 0).sum()))
window, N = make_window(net)
s = init_state(N); ring = jnp.zeros((DMAX, N)); stp = (jnp.full(N, 0.2), jnp.ones(N))
delays = jnp.asarray(np.clip(np.round(net["L"] / 15.0), 3, 12).astype(np.int32))  # L/v, v=15 mm/ms
ext = jnp.zeros((WIN_MS, 5, N)); key = jax.random.PRNGKey(0)
t0 = time.perf_counter(); out = window(s, ring, stp, 0, delays, key, ext); jax.block_until_ready(out); print("compile s", round(time.perf_counter()-t0, 1))
def epoch(s, ring, stp, m0, n_win=50):
    tot = 0
    for k in range(n_win):
        s, ring, stp, cnts = window(s, ring, stp, m0 + k*WIN_MS, delays, key, ext)
        tot = tot + cnts.sum(0)
    jax.block_until_ready(tot)
    return s, ring, stp, tot
for rep in range(4):
    t0 = time.perf_counter(); s, ring, stp, tot = epoch(s, ring, stp, rep*250); dt = time.perf_counter() - t0
    rates = np.asarray(tot) / 0.25
    g = net["groups"]
    def r(name): a, b = g[name]; return round(float(rates[a:b].mean()), 1)
    print(f"epoch {rep}: {dt*1e3:.0f} ms wall for 250 ms sim | Hz L23_IT {r('c0.L23_IT')} L4 {r('c0.L4_IT')} L5ET {r('c0.L5_ET')} "
          f"PV {r('c0.L23_PV')} SST {r('c0.L23_SST')} TC {r('c0.TC')} CA3 {r('CA3')} DG {r('DG')} | NaN: {bool(np.isnan(np.asarray(s[0])).any())}")
