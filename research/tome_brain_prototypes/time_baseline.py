import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/The-Voidlit-Tome/brain")
import torch
from voidlit.tome import TomeEngine, load_scenario
from voidlit.cortex.director import Director, load_director
from voidlit.playtest import bot_command
from voidlit.senses.state import SensoryState
from voidlit.cortex.features import DirectorFeatures
sc = load_scenario(); model = load_director()
e = TomeEngine(sc, director=Director(model, seed=1), seed=1)
e.start()
# NPC brain perceive timing
b = e.brains["elara"]
s = SensoryState(aggression=0.3, rapport=0.5, leverage=0.4)
for _ in range(3): b.perceive(s)
t0 = time.perf_counter(); N = 20
for _ in range(N): b.perceive(s)
print("NPC perceive (100 ms BindsNET) ms:", (time.perf_counter()-t0)/N*1e3)
d = e.director; f = DirectorFeatures()
for _ in range(3): d.probabilities(f)
t0 = time.perf_counter()
for _ in range(N): d.probabilities(f)
print("Director probabilities ms:", (time.perf_counter()-t0)/N*1e3)
# full turns
ts = []
for i in range(60):
    cmd = bot_command(e, smart=True)
    t0 = time.perf_counter(); fr = e.turn(cmd); ts.append(time.perf_counter()-t0)
    if fr.over: break
ts.sort()
print("engine.turn ms: median %.1f p90 %.1f max %.1f over %d turns" % (ts[len(ts)//2]*1e3, ts[int(len(ts)*0.9)]*1e3, ts[-1]*1e3, len(ts)))
print("threads", torch.get_num_threads())
