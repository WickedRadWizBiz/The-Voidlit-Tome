"""End-to-end: engine turn latency with the network (NPC brains, rules, offline voice), determinism, save/load."""
import os, sys, time, json, tempfile, warnings
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
warnings.filterwarnings("ignore")
import numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, "/home/user/The-Voidlit-Tome/brain")
from net_engine import TomeNetEngine
from voidlit.tome import load_scenario
from voidlit.cortex.director import Director, load_director
from voidlit.playtest import bot_command
from voidlit.voice import OfflineVoice
sc, model = load_scenario(), load_director()
def make(seed):
    return TomeNetEngine(sc, director=Director(model, seed=seed), seed=seed, voice=OfflineVoice(sc.narration, seed=seed))
what = sys.argv[1] if len(sys.argv) > 1 else "all"
if what in ("all", "latency"):
    e = make(1); e.start()
    ts, npcs = [], []
    for i in range(80):
        cmd = bot_command(e)
        t = time.perf_counter(); f = e.turn(cmd); ts.append(time.perf_counter() - t)
        npcs.append(len(e.state.npcs_present))
        if f.over: break
    ts = np.array(ts[1:]) * 1e3
    ms = [x["ms"] for x in e.net_log]
    print(f"engine turns with the network: n={len(ts)} median {np.median(ts):.0f} ms, p90 {np.percentile(ts,90):.0f}, max {ts.max():.0f} "
          f"| simulated ms per turn median {np.median(ms):.0f} | NPCs present mean {np.mean(npcs):.2f} | outcome {'won' if e.state.won else 'dead' if e.state.character.dead else 'running'}")
    from collections import Counter
    print("beats", dict(Counter(x["beat"] for x in e.net_log)), "released", sum(x["released"] for x in e.net_log), "/", len(e.net_log))
if what in ("all", "det"):
    script = ["look around", "Elara, are you okay?", "I force the iris", "I analyze the iris", "press on", "I search the room"]
    a, b = make(5), make(5)
    ok = a.start().text == b.start().text
    for cmd in script:
        ok &= a.turn(cmd).to_dict(include_debug=True) == b.turn(cmd).to_dict(include_debug=True)
    print("engine determinism with the network (6 turns, full frames incl. debug):", ok)
if what in ("all", "save"):
    a = make(8); a.start()
    for cmd in ["Elara, can we trust each other?", "I force the iris", "I analyze the iris"]:
        a.turn(cmd)
    with tempfile.TemporaryDirectory() as d:
        t = time.perf_counter(); a.save(d); ts = time.perf_counter() - t
        size = sum(os.path.getsize(os.path.join(d, x)) for x in os.listdir(d) if os.path.isfile(os.path.join(d, x)))
        b = make(99)
        t = time.perf_counter(); b.load(d); tl = time.perf_counter() - t
    b.director.gen.set_state(a.director.gen.get_state())
    ok = all(a.turn(c).to_dict() == b.turn(c).to_dict() for c in ["press on", "I search", "Elara, what are you hiding?", "I sneak past it"])
    print(f"engine save/load mid-run resumes identically: {ok} (save {ts*1e3:.0f} ms, load incl. re-jit {tl*1e3:.0f} ms, {size/1e6:.1f} MB)")
