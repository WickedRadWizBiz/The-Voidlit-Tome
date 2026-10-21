import warnings; warnings.filterwarnings("ignore")
import numpy as np
import jaxpath  # noqa
import jax, jax.numpy as jnp
import tome_jax as T
ex = np.flatnonzero(T.IS_E); pc = np.concatenate([T.sl(f"{c}.L23_PC") for c in T.COLS])
d = np.zeros(T.N, np.float32); d[ex] = 100.0; d[pc] = 220.0
for c in T.COLS: d[T.sl(f"{c}.TC_core")] += 120.0
st, static = T.build(0); win = T.make_window(); base = jnp.asarray(d)
for _ in range(50): st, _ = win(static, st, base)
c0 = np.asarray(st["count"]).copy()
for _ in range(75): st, _ = win(static, st, base)
c = (np.asarray(st["count"]) - c0) / 0.3
names = ["P.L23_PC", "P.L23_PV", "P.L23_SST", "P.L23_VIP", "P.L1_NGF", "P.L4_SS", "P.L4_PV", "P.L5_ET", "P.L5_IT",
         "P.L5_PV", "P.L6_CT", "P.TC_core", "P.TC_mtx", "P.TRN", "DG", "CA3", "CA3_PV", "CA1", "VTA_DA", "LC", "DRN", "NBM"]
print("  ".join(f"{n}:{c[T.sl(n)].mean():.1f}" for n in names))
print("mean gGA_s on L2/3 PCs (nS):", float(np.asarray(st["gGA_s"])[pc].mean()), " mean gA_s+gA_b:",
      float((np.asarray(st["gA_s"]) + np.asarray(st["gA_b"]))[pc].mean()))
