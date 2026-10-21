import time, torch
torch.set_num_threads(1)
import cosim as CS
sj = CS.SJPart(0)
D = 4
ctx = (torch.rand(D, 256) < 0.01).float(); mf = (torch.rand(D, 128) < 0.03).float()
R = sj.R; n = R.shape[0]
RsT = R.T.contiguous().to_sparse_csr()
print("BG n", n, "R nnz", int((R != 0).sum()), "of", n * n)
x = (torch.rand(1, n) < 0.05).float()
def tm(f, k=2000):
    f(); t = time.perf_counter()
    for _ in range(k): f()
    return (time.perf_counter() - t) / k * 1e6
print(f"dense x@R {tm(lambda: x @ R):.1f} us | sparse CSR RT@x {tm(lambda: (RsT @ x.T)):.1f} us")
xs = x.to_sparse()
print(f"LIF 950 single step {tm(lambda: sj.bg(x)):.1f} us")
idx = (sj.w_mf_gc != 0).T.nonzero()[:, 1].reshape(2048, 4)
print(f"MF->GrC dense {tm(lambda: mf @ sj.w_mf_gc, 500):.1f} us | gather {tm(lambda: mf[:, idx].sum(-1) * 0.45, 500):.1f} us")
print(f"GrC LIF multi-step [4,1,2048] {tm(lambda: sj.gc(torch.zeros(4,1,2048)), 500):.1f} us | PF->PC {tm(lambda: torch.zeros(4,1,2048) @ sj.w_pf_pc, 500):.1f} us")
with torch.no_grad():
    print(f"whole step_window {tm(lambda: sj.step_window(ctx, torch.zeros(D,64), mf, torch.zeros(D)), 300):.0f} us")
