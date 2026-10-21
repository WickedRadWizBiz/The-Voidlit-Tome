"""Time the JAX partition per 200 ms turn: window size D, substeps, threads; report firing rates."""
import os, sys, time
threads = int(os.environ.get("XLA_THREADS", "1"))
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + (" --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" if threads == 1 else "")
import numpy as np
import tome_jax as TJ
import jax, jax.numpy as jnp

def drive(meta, T, seed, mu=None):
    L = meta["L"]; rng = np.random.default_rng(seed)
    N = L.n
    base = np.zeros(N, np.float32)
    for p in L.names:
        k = p.split(".")[-1]
        sl = slice(L.start[p], L.start[p] + L.size[p])
        base[sl] = {"L23": 130, "L5ET": 140, "L4": 150, "L5IT": 150, "L6CT": 140, "PV": 60, "SST": 50, "VIP": 40,
                    "LAMP5": 30, "TC": 170, "MTX": 140, "TRN": 60, "CA3T": 150, "CA3A": 150, "CA1": 150}[k]
    ext = base[None, :] + 90.0 * rng.standard_normal((T, N)).astype(np.float32)
    # game-feature channels: half the TC cells get a Poisson-ish boost
    ext[:, L.start["TC"]:L.start["TC"] + 64] += 120.0 * (rng.random((T, 64)) < 0.3)
    return ext

if __name__ == "__main__":
    sub = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    TURN = 200
    P, meta = TJ.build(seed=0, dt_sub=1.0 / sub)
    print(f"N={meta['L'].n} (MC {meta['L'].n_mc}) edges={meta['n_edges']} kmax={meta['kmax']} plastic={meta['n_plastic']} slots={meta['n_slot']} sub={sub} threads={threads}")
    st0 = TJ.init_state(P, meta, jax.random.PRNGKey(0))
    ext = drive(meta, TURN * 6, 1)
    for D in (4, 200):
        run = TJ.make_window(P, meta, D, sub)
        t = time.perf_counter()
        st, sp = run(st0, jnp.asarray(ext[:D]), 1.0, 0.0, 0.0); jax.block_until_ready(sp)
        comp = time.perf_counter() - t
        times = []
        st = st0
        spikes = []
        for turn in range(6):
            t = time.perf_counter()
            for w in range(TURN // D):
                o = turn * TURN + w * D
                st, sp = run(st, jnp.asarray(ext[o:o + D]), 1.0, 0.0, 0.0)
                spikes.append(sp)
            jax.block_until_ready(sp)
            times.append(time.perf_counter() - t)
        S = np.concatenate([np.asarray(x) for x in spikes])
        print(f"D={D:3d}: compile {comp:.2f}s, per 200 ms turn: median {np.median(times)*1e3:.1f} ms (min {min(times)*1e3:.1f})")
    L = meta["L"]
    rates = {p: S[:, L.start[p]:L.start[p] + L.size[p]].mean() * 1000 for p in L.names}
    print("rates Hz:", {k: round(v, 1) for k, v in rates.items() if k.startswith("TOM") or "." not in k})
    print("max spikes/ms", S.sum(1).max(), "mean", S.sum(1).mean(), "rho mean", float(st["rho"].mean()), "atp", np.asarray(st["atp"]).round(3), "astroCa", float(st["astro"][1].mean()), "nan?", bool(np.isnan(np.asarray(st["V"])).any()))
