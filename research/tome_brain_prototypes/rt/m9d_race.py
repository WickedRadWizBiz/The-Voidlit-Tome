"""Race-to-threshold commit in the BG pace loop; fidelity of the chosen-beat distribution to the Director."""
import sys, time, numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, "/home/user/The-Voidlit-Tome/brain")
import cosim as CS
from m9b_bg import make
W = 4
def race(sj, sal, K=6, max_ms=240, seed=0, rate=0.004):
    g = torch.Generator().manual_seed(seed)
    M = sj.sizes["M"]
    het = 0.85 + 0.3 * torch.rand(CS.N_CH * M, generator=torch.Generator().manual_seed(1))
    sj.prior_bias = torch.repeat_interleave(torch.as_tensor(sal, dtype=torch.float32), M) * het
    cnt = torch.zeros(CS.N_CH)
    with torch.no_grad():
        for w in range(max_ms // W):
            ctx = (torch.rand(W, 256, generator=g) < rate).float()
            o = sj.step_window(ctx, torch.zeros(W, 64), torch.zeros(W, 128), torch.zeros(W))
            v = o["vavl"].reshape(W, CS.N_CH, 4).sum(2)
            for t in range(W):
                cnt += v[t]
                if cnt[:8].max() >= K:
                    c = cnt[:8]
                    return int(torch.argmax(c)), (w * W + t + 1)
    return int(torch.argmax(cnt[:8])), max_ms
if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    rng = np.random.default_rng(0)
    from voidlit.cortex.director import load_director
    from voidlit.cortex.teacher import sample_states
    from voidlit.cortex.features import BEATS
    model = load_director(); gen = torch.Generator().manual_seed(3)
    X = sample_states(n, gen)
    with torch.no_grad():
        P = torch.softmax(model(X, gen), 1).numpy()
    lp_net, lp_exact, hits, lat = [], [], 0, []
    hist_net, hist_exp = np.zeros(8), P.sum(0)
    t0 = time.perf_counter()
    for i in range(n):
        p = P[i]
        gum = -np.log(-np.log(rng.random(8)))
        score = np.log(p + 1e-6) + gum
        sal = np.zeros(CS.N_CH); sal[:8] = 0.05 + 0.03 * np.clip(score - score.max() + 4.0, 0, None)
        sj = make(w_d1gpi=0.8, tonic_gpi=0.15); sj.hyper = 1.5
        c, ms = race(sj, sal, seed=i)
        hits += int(c == int(np.argmax(score)))
        lp_net.append(np.log(p[c])); lp_exact.append(float((p * np.log(p + 1e-12)).sum())); lat.append(ms)
        hist_net[c] += 1
    print(f"n={n}: race winner == argmax(log p + Gumbel) in {hits}/{n}; mean log p(chosen) network {np.mean(lp_net):.3f} "
          f"vs exact sampling {np.mean(lp_exact):.3f} (uniform would be {np.mean(np.log(P).mean(1)):.3f})")
    print("beat histogram network", hist_net.astype(int).tolist(), "expected", hist_exp.round(1).tolist())
    print(f"decision latency (simulated ms): median {np.median(lat):.0f}, p90 {np.percentile(lat, 90):.0f}; wall {(time.perf_counter()-t0)/n*1e3:.0f} ms per decision")
