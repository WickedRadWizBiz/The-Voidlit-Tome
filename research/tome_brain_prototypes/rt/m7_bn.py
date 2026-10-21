import time, numpy as np, torch
torch.set_num_threads(1)
import cosim as CS
W = 4
def cost(bn, n=100):
    hyp = np.zeros((W, 48), np.float32); c64 = (np.random.default_rng(0).random((W, 64)) < 0.02).astype(np.float32)
    for _ in range(10): bn.step_window(c64, hyp, W)
    t = time.perf_counter()
    for _ in range(n): bn.step_window(c64, hyp, W)
    return (time.perf_counter() - t) / n * 50 * 1e3
bn = CS.BNPart(0)
print(f"BN as built (6 layers, 6 connections, 1 MSTDPET, 5 monitors): {cost(bn):.0f} ms per 200 ms turn")
for k in ("HYP", "BLA", "CeA"):
    bn.net.monitors.pop(k); bn.mon.pop(k)
print(f"  with 2 monitors (VTA, LC): {cost(bn):.0f} ms")
bn2 = CS.BNPart(0); bn2.net.learning = False
bn2.c_ctx_bla.update_rule = type(bn2.c_ctx_bla.update_rule).__mro__[-2] if False else bn2.c_ctx_bla.update_rule
print(f"  learning off: {cost(bn2):.0f} ms")
# bus overhead: dense numpy projections per window
T = CS.Tome.__new__(CS.Tome)
rng = np.random.default_rng(0)
M = [np.zeros((n, 2080), np.float32) for n in (72, 32, 64, 48)]
xs = [(rng.random((W, n)) < 0.05).astype(np.float32) for n in (72, 32, 64, 48)]
t = time.perf_counter()
for _ in range(500):
    I = sum(x @ m for x, m in zip(xs, M))
print(f"numpy dense bus projections: {(time.perf_counter()-t)/500*50*1e3:.1f} ms per turn")
