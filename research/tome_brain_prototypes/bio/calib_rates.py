import warnings, sys; warnings.filterwarnings("ignore")
import numpy as np
import jaxpath  # noqa
import jax, jax.numpy as jnp
import tome_jax as T
pops = ["S.L1_NGF", "S.L23_VIP", "S.L4_SS", "S.L23_PC", "S.L23_PV", "S.L23_SST", "S.L5_ET", "S.L5_IT", "S.L6_CT", "S.TC_core", "S.TRN", "P.L23_PC", "P.L5_ET", "CA3", "CA1"]
for ge, gi in [(float(a), float(b)) for a, b in (x.split(",") for x in sys.argv[1:])]:
    T.GAIN.update(e=ge, i=gi)
    ex = np.flatnonzero(T.IS_E); pc = np.concatenate([T.sl(f"{c}.L23_PC") for c in T.COLS])
    d = np.zeros(T.N, np.float32); d[ex] = 100.0; d[pc] = 200.0; d[np.flatnonzero(T.IS_I)] = 60.0
    for c in T.COLS: d[T.sl(f"{c}.TC_core")] += 100.0
    st, static = T.build(0); win = T.make_window(); base = jnp.asarray(d)
    for _ in range(50): st, _ = win(static, st, base)
    c0 = np.asarray(st["count"]).copy(); ov0 = int(st["overflow"])
    for _ in range(75): st, _ = win(static, st, base)
    c = (np.asarray(st["count"]) - c0) / 0.3
    print(f"gain e={ge} i={gi}: " + " ".join(f"{n}:{c[T.sl(n)].mean():.1f}" for n in pops) + f" | overflow {int(st['overflow'])-ov0}", flush=True)
