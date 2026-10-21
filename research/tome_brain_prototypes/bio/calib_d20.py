"""Brutal-difficulty calibration of the invisible hybrid checks, on the real engine, without touching the repo.
Monkeypatches: direct actions -> d20 + attribute modifier vs DC (margin -> hits, so obstacle pips keep working),
Last Light -> d20 + Grit, panic -> d20 + Resolve vs 10 + Stress (passive d6 table row as before), Lumen stays d6.
Sweeps the DC to find the setting where the sensible bot escapes ~20% (Tome network NOT included: dice only)."""
import sys, collections, warnings, random
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/The-Voidlit-Tome/brain")
import torch; torch.set_num_threads(1)
import voidlit.rules.character as C
import voidlit.rules.panic as P
import voidlit.tome.engine as E
from voidlit.rules.dice import PoolRoll
from voidlit.cortex.director import Director, load_director
from voidlit.playtest import bot_command
from voidlit.tome import TomeEngine, load_scenario

MOD = {0: -1, 1: 1, 2: 2, 3: 4, 4: 5}          # attribute score -> d20 modifier
CFG = dict(dc=12, act_step=1, last_dc=12, act=0)

def d20(mod, dc, rng):
    d = rng.randint(1, 20); m = d + mod - dc
    if d == 20 or m >= 10: return PoolRoll((d,), mod, 4, 2)       # critical
    if d == 1 or m <= -5: return PoolRoll((d,), mod, 0, 0)        # fail
    if m < 0: return PoolRoll((d,), mod, 1, 0)                    # mixed
    return PoolRoll((d,), mod, 2 if m < 5 else 3, 0)              # success (bigger margin strips more pips)

def roll(self, approach, rng, intent="", bonus=0):
    pool, notes = self.pool_for(approach, intent)
    attr = max(0, min(4, self.attribute(approach)))
    mod = MOD[attr] + 2 * (pool - attr) + 2 * bonus               # each kit/ally/position die -> +2
    return d20(mod, CFG["dc"] + CFG["act_step"] * CFG["act"], rng), notes
C.Character.roll = roll
C.roll_pool = lambda n, rng: d20(MOD[max(0, min(4, n))], CFG["last_dc"], rng)   # Last Light save (Grit)

def panic(char, rng):
    eff = char.stress + (1 if "phobia" in char.conditions else 0)
    resolve = char.insight                                         # stand-in Resolve for the calibration
    d = rng.randint(1, 20)
    if d + MOD[max(0, min(4, resolve))] >= 10 + eff:
        return P.PanicCheck((d, 0), eff, True, None, None)
    row = max(0, min(len(P.PANIC_TABLE) - 1, rng.randint(1, 6) + eff // 2 - 1))  # passive d6 row
    return P.PanicCheck((d, 0), eff, False, row, P.PANIC_TABLE[row])
E.panic_check = panic

_turn = TomeEngine.turn
def turn(self, text):
    CFG["act"] = self.chapter.act
    return _turn(self, text)
TomeEngine.turn = turn

def run(dc, runs=40, smart=True):
    CFG["dc"], CFG["last_dc"] = dc, dc
    sc, model = load_scenario(), load_director()
    out, deaths = collections.Counter(), collections.Counter()
    for seed in range(runs):
        e = TomeEngine(sc, director=Director(model, seed=seed), seed=seed); e.start()
        for n in range(200):
            f = e.turn(bot_command(e, smart=smart))
            if f.over: break
        out["escaped" if e.state.won else "died" if e.state.character.dead else "timeout"] += 1
        if e.state.character.dead: deaths[e.state.deck.index] += 1
    return out["escaped"] / runs, dict(out), dict(sorted(deaths.items()))

if __name__ == "__main__":
    for dc in [int(x) for x in sys.argv[1:]]:
        r, o, d = run(dc)
        print(f"DC {dc} (+1/act): smart escapes {r:.0%} {o} deaths by chapter {d}", flush=True)
