import sys, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/The-Voidlit-Tome/brain")
import torch; torch.set_num_threads(1)
from voidlit.cortex.director import Director, load_director
from voidlit.playtest import bot_command
from voidlit.tome import TomeEngine, load_scenario
sc, model = load_scenario(), load_director()
e = TomeEngine(sc, director=Director(model, seed=3), seed=3); e.start()
ts, npc_t = [], []
for n in range(200):
    cmd = bot_command(e, smart=True)
    t0 = time.perf_counter(); f = e.turn(cmd); ts.append(time.perf_counter() - t0)
    if f.over: break
import statistics as S
print(f"current engine (Director + NPC BindsNET brains + offline voice): {len(ts)} turns, median {1e3*S.median(ts):.1f} ms, p90 {1e3*sorted(ts)[int(.9*len(ts))]:.1f} ms, won={e.state.won}")
