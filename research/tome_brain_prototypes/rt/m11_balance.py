"""Game-level 'preserve the Director' check: escape rate with the network (prior only) vs the original Director."""
import os, sys, time, warnings, collections
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
warnings.filterwarnings("ignore")
import numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, "/home/user/The-Voidlit-Tome/brain")
from net_engine import TomeNetEngine
from voidlit.tome import TomeEngine, load_scenario
from voidlit.cortex.director import Director, load_director
from voidlit.playtest import bot_command
sc, model = load_scenario(), load_director()
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 40
kind = sys.argv[1]
out = collections.Counter(); beats = collections.Counter(); turns = []
t0 = time.perf_counter()
net = None
for seed in range(runs):
    if kind == "net":
        e = TomeNetEngine(sc, director=Director(model, seed=seed), seed=seed)
    else:
        e = TomeEngine(sc, director=Director(model, seed=seed), seed=seed)
    e.start()
    for n in range(1, 201):
        f = e.turn(bot_command(e, smart=True))
        beats[f.debug["beat"]] += 1
        if f.over: break
    out["escaped" if e.state.won else "died" if e.state.character.dead else "timeout"] += 1
    turns.append(n)
tot = sum(beats.values())
print(f"{kind}: smart escaped {out['escaped']}/{runs} = {out['escaped']/runs:.0%} {dict(out)} avg turns {np.mean(turns):.1f} | beat shares "
      + ", ".join(f"{k} {v/tot:.2f}" for k, v in sorted(beats.items())) + f" | {time.perf_counter()-t0:.0f}s")
