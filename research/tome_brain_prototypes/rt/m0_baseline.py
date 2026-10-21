"""Baseline costs in the current game: full engine turn, NPC brain perceive, Director choose."""
import sys, time, random, statistics as S, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/The-Voidlit-Tome/brain")
import torch
from voidlit.tome import TomeEngine, load_scenario
from voidlit.cortex.director import Director, load_director
from voidlit.cortex.features import DirectorFeatures
from voidlit.playtest import bot_command
from voidlit.senses.state import SensoryState

def tm(f, n):
    xs = []
    for _ in range(n):
        t = time.perf_counter(); f(); xs.append(time.perf_counter() - t)
    xs.sort(); return xs[len(xs)//2] * 1e3, xs[int(0.9*len(xs))] * 1e3

for threads in (1, 4):
    torch.set_num_threads(threads)
    sc, model = load_scenario(), load_director()
    e = TomeEngine(sc, director=Director(model, seed=1), seed=1)
    e.start()
    turn_t = []
    for i in range(60):
        t = time.perf_counter(); f = e.turn(bot_command(e)); turn_t.append(time.perf_counter() - t)
        if f.over: break
    turn_t.sort()
    brain = e.brains["elara"]
    s = SensoryState(aggression=0.3, deception=0.2, leverage=0.4, exposure=0.1, rapport=0.5, boldness=0.4)
    npc = tm(lambda: brain.perceive(s), 30)
    d = Director(model, seed=0); feat = DirectorFeatures(); rng = random.Random(0)
    dirc = tm(lambda: d.choose(feat, rng), 200)
    print(f"threads={threads}: engine turn median {turn_t[len(turn_t)//2]*1e3:.1f} ms p90 {turn_t[int(.9*len(turn_t))]*1e3:.1f} ms "
          f"(n={len(turn_t)}) | NPC perceive (100 ms sim) median {npc[0]:.1f} p90 {npc[1]:.1f} ms | Director choose {dirc[0]:.2f} ms")
