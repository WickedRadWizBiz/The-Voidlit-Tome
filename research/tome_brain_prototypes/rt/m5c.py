import time, sys, torch
torch.set_num_threads(1)
if sys.argv[1] == "ftz":
    print("flush denormal:", torch.set_flush_denormal(True))
import cosim as CS
sj = CS.SJPart(0)
if sys.argv[2] == "csr":
    RT_ = sj.R.T.contiguous().to_sparse_csr()
    CS.SJPart._rec = lambda self, prev: (RT_ @ prev.T).T
else:
    CS.SJPart._rec = lambda self, prev: prev @ self.R
src = open("cosim.py").read()
D = 4
ctx = (torch.rand(D, 256) < 0.01).float(); mf = (torch.rand(D, 128) < 0.03).float()
with torch.no_grad():
    for _ in range(50): sj.step_window(ctx, torch.zeros(D, 64), mf, torch.zeros(D))
    t = time.perf_counter()
    for _ in range(300): sj.step_window(ctx, torch.zeros(D, 64), mf, torch.zeros(D))
print(sys.argv[1:], f"step_window {(time.perf_counter()-t)/300*1e6:.0f} us -> per 200 ms turn {(time.perf_counter()-t)/300*50*1e3:.0f} ms")
