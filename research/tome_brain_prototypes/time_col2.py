import time
import jaxpath  # noqa
import jax, jax.numpy as jnp, numpy as np
import column2 as C
for dt_sub, sub in ((0.1, 10), (0.25, 4)):
    run = C.make(dt_sub, sub)
    st, W, idx = C.init(jax.random.PRNGKey(0), dt_sub)
    TURN = 200
    for epoch_ticks in (2, TURN):
        ext = jnp.zeros((epoch_ticks, C.N)) + 0.0005
        s, sp = run(st, W, idx, ext); jax.block_until_ready(sp)
        t0 = time.perf_counter(); s = st
        for _ in range(TURN // epoch_ticks):
            s, sp = run(s, W, idx, ext)
        jax.block_until_ready(sp)
        print(f"dt_sub={dt_sub} ms ({sub}/tick), epoch={epoch_ticks} ticks: {(time.perf_counter()-t0)*1e3:.1f} ms "
              f"per 200 ms turn; N={C.N} ({C.N_PC} 9-comp PCs)")
    # epoch invariance (bitwise): 2-tick epochs vs 1-tick epochs
    ext = jnp.zeros((TURN, C.N)) + 0.0005
    a = st; sa = []
    for k in range(0, TURN, 2):
        a, sp = run(a, W, idx, ext[k:k + 2]); sa.append(sp)
    b = st; sb = []
    for k in range(TURN):
        b, sp = run(b, W, idx, ext[k:k + 1]); sb.append(sp)
    sa, sb = np.concatenate(sa), np.concatenate(sb)
    print("   epoch invariance 2-tick vs 1-tick: spikes identical =", bool((sa == sb).all()),
          " V max |diff| =", float(jnp.max(jnp.abs(a["pc"]["V"] - b["pc"]["V"]))), " total spikes", int(sa.sum()))
