import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
import hybrid as Hy
tm = Hy.Tome(seed=0)
gr = Hy.game_rates(0, 0)
for e in range(12):
    s, so, bo = tm.epoch(gr)
    rates = {k: round(float(s[:, a:b].mean()) * 1000, 0) for k, (a, b) in tm.G["off"].items()}
    print(e, "J pops Hz:", rates, "| dcn", float(so["dcn"].mean()) * 1000, "pc", float(so["pc"].mean()) * 1000)
