import cProfile, pstats, torch, time
import tome_bn2 as B
torch.set_num_threads(1)
net, mons, tonic = B.build(0)
g = torch.Generator().manual_seed(1)
def go():
    for k in range(62):
        ctx = (torch.rand(4, B.N_CTX, generator=g) < 0.02).float()
        B.window(net, tonic, ctx, 0.1, 0.5)
        for m in mons.values(): m.reset_state_variables()
go()
pr = cProfile.Profile(); pr.enable(); go(); pr.disable()
pstats.Stats(pr).sort_stats("tottime").print_stats(14)
net.learning = False
t0 = time.perf_counter(); go(); print("learning off:", round(1e3*(time.perf_counter()-t0),1), "ms")
net.learning = True
for k in list(net.monitors): del net.monitors[k]
t0 = time.perf_counter()
for k in range(62):
    ctx = (torch.rand(4, B.N_CTX, generator=g) < 0.02).float(); B.window(net, tonic, ctx, 0.1, 0.5)
print("no monitors:", round(1e3*(time.perf_counter()-t0),1), "ms")
