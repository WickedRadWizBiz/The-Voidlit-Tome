import sys, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/The-Voidlit-Tome/brain")
import torch
from voidlit.tome import load_scenario
from voidlit.limbic.npc_brain import NPCBrain
from voidlit.senses.state import SensoryState
sc = load_scenario()
b = NPCBrain(sc.npcs["elara"], seed=1)
b.perceive(SensoryState(rapport=0.5))
t0 = time.perf_counter()
for k in range(10):
    b.perceive(SensoryState(rapport=0.3, deception=0.2))
print(f"NPC brain perceive (100 ms BindsNET): {(time.perf_counter()-t0)/10*1000:.1f} ms each")
import hybrid as Hy
bus = Hy.Bus({"a": 3})
x = torch.zeros(4, 3); x[1, 2] = 1.0            # spike at tick t0+1 = 41
bus.write("a", 40, x)
arr = [(t, float(bus.read("a", t0, 6)[t - t0, 2])) for t0 in (44, 48) for t in range(t0, t0 + 4)]
print("bus: spike emitted at t=41 with delay 6 is seen at t =", [t for t, v in arr if v > 0])
# partition order invariance: run J before S before B vs B before S before J inside each epoch
import numpy as np
def run(order):
    tm = Hy.Tome(seed=3)
    orig = tm.epoch
    return [tm.turn(Hy.game_rates(3, k)) for k in range(2)]
a = run("JSB")
src = open("hybrid.py").read()
assert "st, key, s = run_j(); so = run_s(); bo = run_b()" in src
Hy_src = src.replace("st, key, s = run_j(); so = run_s(); bo = run_b()", "bo = run_b(); so = run_s(); st, key, s = run_j()")
ns = {}
exec(compile(Hy_src, "hybrid_rev", "exec"), ns)
tm = ns["Tome"](seed=3)
b = [tm.turn(ns["game_rates"](3, k)) for k in range(2)]
print("reversed partition order gives identical turns:", a == b)
