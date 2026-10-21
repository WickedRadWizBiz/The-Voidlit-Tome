"""Tune the BG loops: tonic GPi, D1 disinhibition, VA/VL release; measure selection vs a prior."""
import sys, time, numpy as np, torch
torch.set_num_threads(1)
import cosim as CS
W = 4
def trial(sj, prior8, ms=160, seed=0, rate=0.004, kappa=None):
    g = torch.Generator().manual_seed(seed)
    M = sj.sizes["M"]
    bias = torch.zeros(CS.N_CH * M)
    bias[: 8 * M] = torch.repeat_interleave(torch.as_tensor(prior8, dtype=torch.float32), M)
    sj.prior_bias = bias
    counts = torch.zeros(CS.N_CH)
    gpi = 0.0; d1 = 0.0
    with torch.no_grad():
        for w in range(ms // W):
            ctx = (torch.rand(W, 256, generator=g) < rate).float()
            o = sj.step_window(ctx, torch.zeros(W, 64), torch.zeros(W, 128), torch.zeros(W))
            if w >= 5:
                counts += o["vavl"].reshape(W, CS.N_CH, 4).sum((0, 2))
                gpi += float(o["gpi"].mean()); d1 += float(o["d1"].mean())
    return counts, gpi / (ms // W - 5) * 1000, d1 / (ms // W - 5) * 1000

if __name__ == "__main__":
    sj = CS.SJPart(0)
    for tonic_gpi, w_d1gpi, w_gpivavl, vavl_bias in [(0.42, 0.30, 0.6, 0.08), (0.30, 1.0, 0.8, 0.16), (0.25, 1.5, 1.0, 0.2)]:
        sj = CS.SJPart(0)
        M, I, V = sj.sizes["M"], sj.sizes["I"], sj.sizes["V"]
        o = sj.off
        sj.tonic[-CS.N_CH * I:] = tonic_gpi
        R = sj.R.clone()
        a, b = o["d1"]; c, d = o["gpi"]
        R[a:b, c:d] = (R[a:b, c:d] != 0).float() * -w_d1gpi
        a, b = o["gpi"]; c, d = o["vavl"]
        R[a:b, c:d] = (R[a:b, c:d] != 0).float() * -w_gpivavl
        sj.R = R; sj.RT_csr = R.T.contiguous().to_sparse_csr()
        import types
        old = sj.step_window
        p = np.array([0.05, 0.4, 0.05, 0.05, 0.05, 0.3, 0.05, 0.05])
        cnt, gpi, d1 = trial(sj, 0.25 * p / p.max())
        print(f"tonicGPi {tonic_gpi} D1-|GPi {w_d1gpi} GPi-|VAVL {w_gpivavl}: GPi {gpi:.0f} Hz, D1 {d1:.1f} Hz, VA/VL beat counts {cnt[:8].int().tolist()} other loops {cnt[8:].int().tolist()}")
