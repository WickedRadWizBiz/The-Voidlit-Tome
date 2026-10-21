"""Calibrate the d20 layer for brutal difficulty: smart-bot escape rate vs base DC (repo code monkeypatched here).

DC(act) = BASE + ACT_STEP * act (+2 for horrors). Last Light: d20 + Grit mod vs LL_DC. Panic: d6 + Resolve mod vs
Stress//2 + 2 (passive). Lumen gutter: unchanged d6 depletion (already a passive d6 event).
"""
import os, sys, random, collections, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/The-Voidlit-Tome/brain")
import torch
torch.set_num_threads(1)
import voidlit.rules.character as CH
import voidlit.rules.panic as PN
import voidlit.tome.engine as EN
from voidlit.tome import TomeEngine, load_scenario
from voidlit.cortex.director import Director, load_director
from voidlit.playtest import bot_command
from voidlit.rules.dice import roll_2d6
import d20

BASE = int(os.environ.get("BASE", "12")); STEP = int(os.environ.get("STEP", "1")); LL_DC = int(os.environ.get("LL", "12"))
RESOLVE = {"Salvager": 2, "Ex-Marine": 3, "Engineer": 2, "Medic": 2, "Pilot": 1, "Archivist": 2}
CTX = {"act": 0, "kind": "lock"}

def roll(self, approach, rng, intent="", bonus=0):
    pool, notes = self.pool_for(approach, intent)          # attribute + item + conditions, as before
    attr = self.attribute(approach)
    mod = d20.attr_mod(attr) + 2 * (pool - attr) + 2 * bonus
    dc = BASE + STEP * CTX["act"] + (2 if CTX["kind"] == "horror" else 0)
    return d20.check(rng, mod, dc).as_pool(), notes
CH.Character.roll = roll

def last_light(n, rng):
    # called by take_harm with the Grit score: d20 + Grit mod vs LL_DC; fail -> 0 hits (death)
    c = d20.check(rng, d20.attr_mod(n), LL_DC)
    return c.as_pool() if c.degree > 0 else d20.Check(1, 0, 0, 99).as_pool()
CH.roll_pool = last_light

def panic_check(char, rng):
    eff = char.stress + (1 if "phobia" in char.conditions else 0)
    res = RESOLVE.get(char.background, 2)
    d, held = d20.passive(rng, res, eff // 2 + 2)
    if held:
        return PN.PanicCheck((d, 0), eff, True, None, None)
    row = max(0, min(len(PN.PANIC_TABLE) - 1, rng.randint(1, 6) + eff // 2 - 1))
    return PN.PanicCheck((d, 0), eff, False, row, PN.PANIC_TABLE[row])
EN.panic_check = panic_check

orig_engage = EN.TomeEngine._engage
def engage(self, t, ob, s):
    CTX["act"], CTX["kind"] = self.chapter.act, ob.kind
    return orig_engage(self, t, ob, s)
EN.TomeEngine._engage = engage
orig_turn = EN.TomeEngine.turn
def turn(self, text):
    CTX["act"], CTX["kind"] = self.chapter.act, "lock"
    return orig_turn(self, text)
EN.TomeEngine.turn = turn

sc, model = load_scenario(), load_director()
runs = int(sys.argv[1]) if len(sys.argv) > 1 else 40
out = collections.Counter(); deaths = collections.Counter()
for seed in range(runs):
    e = TomeEngine(sc, director=Director(model, seed=seed), seed=seed)
    e.start()
    for n in range(200):
        f = e.turn(bot_command(e, smart=True))
        if f.over: break
    out["escaped" if e.state.won else "died" if e.state.character.dead else "timeout"] += 1
    if e.state.character.dead:
        deaths[e.state.deck.index] += 1
print(f"BASE {BASE} STEP {STEP} LL_DC {LL_DC}: smart escaped {out['escaped']}/{runs} = {out['escaped']/runs:.0%} {dict(out)} deaths by chapter {dict(sorted(deaths.items()))}")
