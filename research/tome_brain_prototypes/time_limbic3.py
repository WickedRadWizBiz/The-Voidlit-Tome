import time, warnings
warnings.filterwarnings("ignore")
import torch
torch.set_num_threads(1)
from time_limbic2 import build
import cProfile, pstats
for n_ctx, n_lim in ((96, 640), (128, 640), (96, 512)):
    net = build(n_lim, n_ctx=n_ctx)
    inp = {"ctx": (torch.rand(2, 1, n_ctx) < 0.03).float(), "intero": (torch.rand(2, 1, 64) < 0.05).float()}
    net.run(inputs=inp, time=2, reward=0.0)
    t0 = time.perf_counter()
    for _ in range(100): net.run(inputs=inp, time=2, reward=0.0)
    t_l = time.perf_counter() - t0
    net.learning = False
    for c in net.connections.values(): c.update_rule.nu = torch.zeros(2) if hasattr(c.update_rule, "nu") else None
    net2 = build(n_lim, n_ctx=n_ctx); net2.train(False)
    net2.run(inputs=inp, time=2, reward=0.0)
    t0 = time.perf_counter()
    for _ in range(100): net2.run(inputs=inp, time=2, reward=0.0)
    t_n = time.perf_counter() - t0
    print(f"ctx={n_ctx} limbic={n_lim}: learning {t_l*1e3:.1f} ms/turn, no learning {t_n*1e3:.1f} ms/turn")
net = build(640, n_ctx=96)
inp = {"ctx": (torch.rand(2, 1, 96) < 0.03).float(), "intero": (torch.rand(2, 1, 64) < 0.05).float()}
pr = cProfile.Profile(); pr.enable()
for _ in range(100): net.run(inputs=inp, time=2, reward=0.0)
pr.disable(); pstats.Stats(pr).sort_stats("tottime").print_stats(8)
