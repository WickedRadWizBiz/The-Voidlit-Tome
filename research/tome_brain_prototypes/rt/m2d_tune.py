import os, sys, time
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
import jaxpath  # noqa
import numpy as np, jax, jax.numpy as jnp
import tome_jax as TJ
BASE = {"L23": 330, "L5ET": 340, "L4": 175, "L5IT": 175, "L6CT": 170, "PV": 120, "SST": 90, "VIP": 70,
        "LAMP5": 60, "TC": 420, "MTX": 400, "TRN": 140, "CA3T": 170, "CA3A": 170, "CA1": 170}
def drive(meta, T, seed, scale=1.0, sigma=150.0, base=BASE):
    L = meta["L"]; rng = np.random.default_rng(seed); N = L.n
    b = np.zeros(N, np.float32)
    for p in L.names:
        b[L.start[p]:L.start[p] + L.size[p]] = base[p.split(".")[-1]] * scale
    ext = b[None, :] + sigma * rng.standard_normal((T, N)).astype(np.float32)
    ext[:, L.start["TC"]:L.start["TC"] + 64] += 150.0 * (rng.random((T, 64)) < 0.3)
    return ext
if __name__ == "__main__":
    sub = int(sys.argv[1]); smax = int(sys.argv[2])
    P, meta = TJ.build(seed=0, dt_sub=1.0 / sub)
    L = meta["L"]
    st0 = TJ.init_state(P, meta, jax.random.PRNGKey(0))
    run = TJ.make_window(P, meta, 200, sub, s_max=smax)
    for scale in [float(x) for x in sys.argv[3:]]:
        ext = jnp.asarray(drive(meta, 1000, 1, scale))
        st = st0; S = []
        for k in range(5):
            st, sp = run(st, ext[k*200:(k+1)*200], 1.0, 0.0, 0.0); S.append(np.asarray(sp))
        S = np.concatenate(S)[200:]
        t0 = time.perf_counter(); st2, sp = run(st, ext[:200], 1.0, 0.0, 0.0); jax.block_until_ready(sp); dt_ = time.perf_counter() - t0
        r = {p: round(float(S[:, L.start[p]:L.start[p] + L.size[p]].mean() * 1000), 1) for p in L.names}
        keep = {k: v for k, v in r.items() if k.startswith("TOM") or "." not in k}
        print(f"scale {scale}: {dt_*1e3:.1f} ms/turn | spikes/ms mean {S.sum(1).mean():.1f} p99 {np.percentile(S.sum(1),99):.0f} max {S.sum(1).max():.0f} | {keep} | atp {np.asarray(st['atp']).round(3)} nan {bool(np.isnan(np.asarray(st['V'])).any())}")
