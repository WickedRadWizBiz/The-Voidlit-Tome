"""Gate 4 (honest): macroscopic stability of the REAL loop (tome_jax.ms_step), not of an uncalled function.
Baseline 300 ms -> +60% drive pulse 100 ms -> baseline 300 ms. PASS if the excitatory rate in the last 200 ms is
within 25% of the pre-pulse rate, the pulse response is bounded (< 40 Hz) and no gather overflow occurs after the
warm-up. Mutant: PV and SST inhibition onto pyramidal cells removed (rows zeroed) -> must FAIL (runaway)."""
import warnings, sys; warnings.filterwarnings("ignore")
import numpy as np
import jaxpath  # noqa
import jax, jax.numpy as jnp
import tome_jax as T
ex = np.flatnonzero(T.IS_E)
pc = np.concatenate([T.sl(f"{c}.L23_PC") for c in T.COLS])
d = np.zeros(T.N, np.float32); d[ex] = 100.0; d[pc] = 220.0
for c in T.COLS: d[T.sl(f"{c}.TC_core")] += 120.0
base = jnp.asarray(d)
win = T.make_window()

def trial(mutant):
    st, static = T.build(0)
    if mutant:
        w_i = np.asarray(static["w_i"]).copy()
        ipos = {g: i for i, g in enumerate(T.I_IDX)}
        for c in T.COLS:
            for pop in ("L23_PV", "L23_SST", "L5_PV", "L5_SST"):
                for g in T.sl(f"{c}.{pop}"):
                    w_i[ipos[g]] = 0.0
        static = dict(static, w_i=jnp.asarray(w_i))
        st = dict(st, vg_w=[v * 0 for v in st["vg_w"]])
    for _ in range(50): st, _ = win(static, st, base)        # 200 ms warm-up
    ov0 = int(st["overflow"])
    def rate(n, drive):
        nonlocal st
        c0 = np.asarray(st["count"]).copy()
        for _ in range(n): st, _ = win(static, st, drive)
        return (np.asarray(st["count"]) - c0)[ex].mean() / (n * T.W_SYNC * 1e-3)
    r_pre = rate(75, base); r_pulse = rate(25, base * 1.6); _ = rate(25, base); r_post = rate(50, base)
    ov = int(st["overflow"]) - ov0
    ok = abs(r_post - r_pre) <= 0.25 * max(r_pre, 0.5) and r_pulse < 40.0 and ov == 0
    return r_pre, r_pulse, r_post, ov, ok

for mutant in (False, True):
    r = trial(mutant)
    print(f"{'MUTANT (no PV/SST->PC)' if mutant else 'intact network        '}: pre {r[0]:.2f} Hz  pulse {r[1]:.2f} Hz  "
          f"post {r[2]:.2f} Hz  overflow {r[3]}  PASS={r[4]}", flush=True)
