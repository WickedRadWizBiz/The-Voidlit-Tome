"""Rules-only balance proxy for the invisible hybrid checks (d20 + modifier for actions, d6 for passive events).

Monkeypatches the existing engine (repo untouched): every Character.roll becomes d20 + mod vs DC with four degrees,
Last Light and Panic become passive d6 checks, Lumen stays a d6 depletion pool. The Director and both scripted bots
are the repo's own. Usage: python d20_balance.py BASE STEP RUNS [interval]
"""
import sys, warnings, collections, random
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/The-Voidlit-Tome/brain")
from voidlit.rules import character as C, dice as D, panic as P
from voidlit.rules.dice import PoolRoll
from voidlit.tome import engine as E
from voidlit.cortex.director import Director, load_director
from voidlit.playtest import bot_command
from voidlit.tome import TomeEngine, load_scenario

CUR = {"dc": 14}
RESOLVE = {"salvager": 2, "marine": 2, "engineer": 1, "medic": 2, "pilot": 1, "archivist": 2}


def mod_of(score):
    return 2 * score - 2          # 0 -> -2, 1 -> 0, 2 -> +2, 3 -> +4, 4 -> +6


def d20_check(mod, dc, rng):
    r = rng.randint(1, 20)
    t = r + mod
    deg = 3 if t >= dc + 10 else 2 if t >= dc else 1 if t >= dc - 5 else 0
    if r == 20:
        deg = min(3, deg + 1)
    if r == 1:
        deg = max(0, deg - 1)
    hits = (0, 1, 2, 2)[deg]
    sixes = 2 if deg == 3 else 0
    return PoolRoll(dice=(r,), pool=mod, hits=hits, sixes=sixes)


def roll(self, approach, rng, intent="", bonus=0):
    pool, notes = self.pool_for(approach, intent)       # attribute + best item die - conditions
    base = self.attribute(approach)
    extra = pool - max(0, base)                          # item/condition delta in "dice"
    mod = mod_of(base) + 2 * extra + 2 * bonus
    return d20_check(mod, CUR["dc"], rng), notes


def take_harm(self, amount, rng):
    if amount <= 0 or self.dead:
        return C.HarmResult(0, 0, None, self.dead)
    soaked = min(self.hull, amount)
    self.hull -= soaked
    overflow = amount - soaked
    if overflow <= 0:
        return C.HarmResult(soaked, 0, None, False)
    d = rng.randint(1, 6)                                # passive d6 + Grit vs 5 + overflow
    ok = d + self.grit >= 5 + overflow
    save = PoolRoll(dice=(d,), pool=self.grit, hits=int(ok), sixes=0)
    if not ok:
        self.dead = True
        return C.HarmResult(soaked, 0, save, True)
    self.grit = max(0, self.grit - 1)
    self.conditions.add("bleeding")
    return C.HarmResult(soaked, 1, save, False)


def panic_check(char, rng):
    eff = char.stress + (1 if "phobia" in char.conditions else 0)
    d = rng.randint(1, 6)
    res = RESOLVE.get(char.background.lower().replace("ex-", ""), 2)
    if d + res + 3 > eff:                                # passive d6 + Resolve
        return P.PanicCheck((d, 0), eff, True, None, None)
    row = max(0, min(len(P.PANIC_TABLE) - 1, rng.randint(1, 6) + eff // 2 - 1))
    return P.PanicCheck((d, 0), eff, False, row, P.PANIC_TABLE[row])


C.Character.roll = roll
C.Character.take_harm = take_harm
E.panic_check = panic_check
_turn = TomeEngine.turn


def make_turn(base, step):
    def turn(self, text):
        act = self.chapter.act
        CUR["dc"] = base + step * act + (1 if self.state.lumen.dark else 0)
        return _turn(self, text)
    return turn


def run(base, step, runs, smart, interval=None, max_turns=200):
    TomeEngine.turn = make_turn(base, step)
    sc, model = load_scenario(), load_director()
    if interval:
        sc.lumen_interval = interval
    out, deaths, turns = collections.Counter(), collections.Counter(), []
    for seed in range(runs):
        e = TomeEngine(sc, director=Director(model, seed=seed), seed=seed)
        e.start()
        for n in range(1, max_turns + 1):
            f = e.turn(bot_command(e, smart=smart))
            if f.over:
                break
        out["escaped" if e.state.won else "died" if e.state.character.dead else "timeout"] += 1
        if e.state.character.dead:
            deaths[e.state.deck.index] += 1
        turns.append(n)
    return out["escaped"] / runs, dict(out), dict(sorted(deaths.items())), sum(turns) / runs


if __name__ == "__main__":
    base, step, runs = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    interval = int(sys.argv[4]) if len(sys.argv) > 4 else None
    for smart in (True, False):
        rate, out, deaths, avg = run(base, step, runs, smart, interval)
        print(f"base={base} step={step} int={interval} {'smart' if smart else 'naive'}: {rate:.0%} {out} "
              f"deaths/chapter {deaths} avg turns {avg:.1f}", flush=True)
