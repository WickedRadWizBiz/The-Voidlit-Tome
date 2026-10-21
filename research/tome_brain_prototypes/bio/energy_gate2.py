import time, warnings, sys; warnings.filterwarnings("ignore")
import numpy as np
import jaxpath  # noqa
import jax, jax.numpy as jnp
import tome_jax as T
pc = np.concatenate([T.sl(f"{c}.L23_PC") for c in T.COLS]); arc = T.sl("ARC"); vta = T.sl("VTA_DA")
ex = np.flatnonzero(T.IS_E)
def make_drive(bg_pc):
    d = np.zeros(T.N, np.float32); d[ex] = 100.0; d[pc] = bg_pc
    for c in T.COLS: d[T.sl(f"{c}.TC_core")] += 120.0
    return jnp.asarray(d)
st0, static = T.build(0)
win, quiet = T.make_window(), T.make_quiet()
for bg in [float(x) for x in sys.argv[1:]]:
    drive = make_drive(bg)
    for dice in (6, 0):
        st = dict(st0, glucose=jnp.float32(T.glucose_of_lumen(dice)))
        for _ in range(3):
            st = quiet(static, st, 20.0, jnp.full(T.N, 3.0, jnp.float32))
        for _ in range(25): st, _ = win(static, st, drive)       # settle 100 ms
        c0 = np.asarray(st["count"]).copy()
        for _ in range(125): st, _ = win(static, st, drive)
        c = (np.asarray(st["count"]) - c0) / 0.5
        print(f"bg {bg:5.0f} pA, Lumen {dice}/6: L2/3 ATP {np.asarray(st['atp'])[pc].mean():.2f}  L2/3 {c[pc].mean():5.2f} Hz  "
              f"all-E {c[ex].mean():5.2f} Hz  ARC {c[arc].mean():5.1f} Hz  VTA {c[vta].mean():4.1f} Hz  "
              f"overflow {int(st['overflow'])}", flush=True)
