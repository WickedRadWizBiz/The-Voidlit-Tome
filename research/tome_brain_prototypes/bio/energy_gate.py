"""Needs from physiology: the Lumen sets the Tome's glucose; ATP, K_ATP, cortical activity and the glucose-
inhibited ARC 'hunger' cells follow. Also times the between-turn quiet integrator."""
import time, warnings; warnings.filterwarnings("ignore")
import numpy as np
import jaxpath  # noqa
import jax, jax.numpy as jnp
import tome_jax as T

st, static = T.build(0)
win, quiet = T.make_window(), T.make_quiet()
drive = jnp.zeros(T.N).at[T.sl("S.TC_core")].set(120.0).at[T.sl("I.TC_core")].set(120.0)  # ongoing input
st, _ = win(static, st, drive)
pc = np.concatenate([T.sl(f"{c}.L23_PC") for c in T.COLS]); arc = T.sl("ARC"); vta = T.sl("VTA_DA")
for dice in (6, 3, 0):
    st = dict(st, glucose=jnp.float32(T.glucose_of_lumen(dice)))
    rate = jnp.asarray(np.asarray(st["count"]) * 0 + 3.0, jnp.float32)
    t0 = time.perf_counter()
    for _ in range(3):                                   # three turns' worth of quiet (3 x 20 s)
        st = quiet(static, st, 20.0, rate); jax.block_until_ready(st["atp"])
    tq = (time.perf_counter() - t0) / 3
    c0 = np.asarray(st["count"]).copy()
    for _ in range(125):                                 # 500 ms of fast dynamics
        st, _ = win(static, st, drive)
    c = (np.asarray(st["count"]) - c0) / 0.5
    atp = np.asarray(st["atp"])
    print(f"Lumen {dice}/6 (glucose {T.glucose_of_lumen(dice):.2f}): cortical L2/3 ATP {atp[pc].mean():.2f}, "
          f"L2/3 rate {c[pc].mean():.2f} Hz, ARC hunger {c[arc].mean():.1f} Hz, VTA DA {c[vta].mean():.1f} Hz | "
          f"quiet step {1e3*tq:.1f} ms", flush=True)
