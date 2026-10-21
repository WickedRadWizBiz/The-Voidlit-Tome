import time, warnings, sys
warnings.filterwarnings("ignore")
import torch
import integrated as I
feats = torch.rand(18)
for threaded in (False, True):
    tome = I.Tome(); tome.threaded = threaded
    tome.turn(feats, seed=0)
    ts = []
    for s in range(1, 6):
        t0 = time.perf_counter(); tome.turn(feats, seed=s); ts.append(time.perf_counter() - t0)
    print("threaded" if threaded else "serial  ", "turn ms median", round(sorted(ts)[2] * 1e3, 1), [round(x * 1e3) for x in ts])
import cProfile, pstats
tome = I.Tome(); tome.turn(feats, seed=0)
pr = cProfile.Profile(); pr.enable(); tome.turn(feats, seed=9); pr.disable()
pstats.Stats(pr).sort_stats("tottime").print_stats(12)
