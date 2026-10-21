import sys, numpy as np, torch
torch.set_num_threads(1)
import cosim as CS
from m9_bg import trial
def make(tonic_gpi=0.15, tonic_gpe=0.14, tonic_stn=0.13, w_d1gpi=0.5, w_gpivavl=0.5, vavl_bias=0.15, d2gpe=0.4):
    sj = CS.SJPart(0)
    o, n = sj.off, CS.N_CH
    E, S, I = sj.sizes["E"], sj.sizes["S"], sj.sizes["I"]
    sj.tonic[: n * E] = tonic_gpe; sj.tonic[n * E: n * (E + S)] = tonic_stn; sj.tonic[n * (E + S):] = tonic_gpi
    R = sj.R.clone()
    def setb(src, dst, w):
        a, b = o[src]; c, d = o[dst]; R[a:b, c:d] = (R[a:b, c:d] != 0).float() * w
    setb("d1", "gpi", -w_d1gpi); setb("gpi", "vavl", -w_gpivavl); setb("d2", "gpe", -d2gpe)
    sj.R = R; sj.RT_csr = R.T.contiguous().to_sparse_csr()
    sj.vavl_bias = vavl_bias
    return sj
# patch: VA/VL bias configurable
src = open("cosim.py").read()
if "self.vavl_bias" not in src:
    src = src.replace("I_in[:, o[\"vavl\"][0]:o[\"vavl\"][1]] = ctx @ self.w_ctx_vavl + 0.08",
                      "I_in[:, o[\"vavl\"][0]:o[\"vavl\"][1]] = ctx @ self.w_ctx_vavl + self.vavl_bias")
    src = src.replace("        self.fuse()\n", "        self.vavl_bias = 0.15\n        self.fuse()\n")
    open("cosim.py", "w").write(src)
    import importlib; importlib.reload(CS)
p = np.array([0.05, 0.4, 0.05, 0.05, 0.05, 0.3, 0.05, 0.05])
for kw in [dict(), dict(w_d1gpi=0.8), dict(w_d1gpi=0.8, tonic_gpi=0.13), dict(w_d1gpi=1.2, w_gpivavl=0.8)]:
    for kappa in (0.06, 0.12):
        sj = make(**kw)
        cnt, gpi, d1 = trial(sj, kappa * p / p.max())
        print(f"{kw} kappa {kappa}: GPi {gpi:.0f} Hz D1 {d1:.1f} Hz | beats {cnt[:8].int().tolist()} | other loops {cnt[8:].int().tolist()}")
