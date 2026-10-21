"""BG winner-take-all reliability and Director-prior fidelity (Gumbel-max via trial-persistent excitability)."""
import sys, time, numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, "/home/user/The-Voidlit-Tome/brain")
import cosim as CS
from m9b_bg import make
W = 4
def decide(sj, sal, ms=120, seed=0, rate=0.004):
    """sal: [N_CH] salience -> D1 bias. Returns the winning channel per loop (argmax VA/VL count) and counts."""
    g = torch.Generator().manual_seed(seed)
    M = sj.sizes["M"]
    het = 0.85 + 0.3 * torch.rand(CS.N_CH * M, generator=torch.Generator().manual_seed(1))
    sj.prior_bias = torch.repeat_interleave(torch.as_tensor(sal, dtype=torch.float32), M) * het
    cnt = torch.zeros(CS.N_CH)
    with torch.no_grad():
        for w in range(ms // W):
            ctx = (torch.rand(W, 256, generator=g) < rate).float()
            o = sj.step_window(ctx, torch.zeros(W, 64), torch.zeros(W, 128), torch.zeros(W))
            if w >= 4:
                cnt += o["vavl"].reshape(W, CS.N_CH, 4).sum((0, 2))
    return cnt
if __name__ == "__main__":
    kw = dict(w_d1gpi=0.8, tonic_gpi=0.15)
    hyper = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
    base, kappa = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05, float(sys.argv[3]) if len(sys.argv) > 3 else 0.03
    rng = np.random.default_rng(0)
    from voidlit.cortex.director import load_director
    from voidlit.cortex.teacher import sample_states
    model = load_director()
    gen = torch.Generator().manual_seed(3)
    X = sample_states(60, gen)
    with torch.no_grad():
        P = torch.softmax(model(X, gen), 1).numpy()
    ok = ties = 0; chosen = []; t0 = time.perf_counter()
    for i in range(len(P)):
        p = P[i]
        gum = -np.log(-np.log(rng.random(8)))                 # trial-persistent excitability ~ Gumbel(0, 1)
        score = np.log(p + 1e-6) + gum
        sal = np.zeros(CS.N_CH)
        z = (score - score.max())                              # <= 0, winner 0
        sal[:8] = base + kappa * np.clip(z + 4.0, 0, None)    # salience: log-prob + Gumbel, clipped floor
        sj = make(**kw); sj.hyper = hyper
        cnt = decide(sj, sal, seed=i)[:8].numpy()
        win = int(np.argmax(cnt)) if cnt.max() > 0 else -1
        ties += int((cnt == cnt.max()).sum() > 1)
        ok += int(win == int(np.argmax(score)))
        chosen.append(win)
    dt = time.perf_counter() - t0
    # distribution fidelity: mean Director probability of the chosen beat vs of a true sample from p
    print(f"hyper {hyper} base {base} kappa {kappa}: network picks argmax(log p + Gumbel) in {ok}/{len(P)} trials "
          f"(ties {ties}, no-release {chosen.count(-1)}); {dt/len(P)*1e3:.0f} ms per 120 ms decision")
