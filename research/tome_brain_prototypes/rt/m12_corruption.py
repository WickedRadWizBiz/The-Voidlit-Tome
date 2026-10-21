"""Corruption statistics on the real, unmodified NPC brains: burst rate, strength, and how often the NPC acts
differently from its uncorrupted counterfactual (same brain, same input, no Tome) at three desperation levels."""
import copy, random, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, "/home/user/The-Voidlit-Tome/brain")
from voidlit.tome import load_scenario
from voidlit.limbic.npc_brain import NPCBrain
from voidlit.senses.state import Intent, SensoryState
import corruption as C
import os
C.PERC_GAIN = float(os.environ.get('PG', '1.0'))
SEL = float(os.environ.get('SEL', '0.6'))
DESPS = [float(x) for x in os.environ.get('DESPS', '0.1,0.5,0.9').split(',')]

sc = load_scenario()
TRANSIENT = "--persistent" not in sys.argv
def jitter(s, r):
    e = lambda v: min(1.0, max(0.0, v + r.uniform(-0.02, 0.02)))
    return SensoryState(intent=s.intent, approach=s.approach, target=s.target, aggression=e(s.aggression),
                        deception=e(s.deception), leverage=e(s.leverage), exposure=e(s.exposure),
                        rapport=max(-1, min(1, s.rapport + r.uniform(-0.02, 0.02))), boldness=s.boldness)
TURNS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 60
RESOLVE = {"elara": 2, "marcus": 1}

def player_inputs(seed, n):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        hostile = r.random() < 0.2
        out.append(SensoryState(intent=Intent.TALK, aggression=r.uniform(0.4, 0.9) if hostile else r.uniform(0, 0.2),
                                deception=r.uniform(0, 0.3), leverage=r.uniform(0, 0.5), exposure=r.uniform(0, 0.4),
                                rapport=r.uniform(-0.6, 0.0) if hostile else r.uniform(0.0, 0.8), boldness=r.uniform(0, 1)))
    return out

for nid in ("elara", "marcus"):
    spec = sc.npcs[nid]
    target = spec.role_index("betray")
    for desp in DESPS:
        brain = NPCBrain(spec, seed=1000 + spec.numeric_id)
        twin = NPCBrain(spec, seed=1000 + spec.numeric_id)       # never corrupted, same inputs
        cum_plot = cum_tac = 0
        jit = NPCBrain(spec, seed=1000 + spec.numeric_id)        # twin with tiny random (non-Tome) input jitter
        jr = random.Random(5)
        in_t = in_tw = div_base = 0
        st = C.Corruption()
        rng = random.Random(7)
        changed_plot = changed_tactic = 0
        cs = []
        sel_rng = random.Random(11)
        dominion_mod = int(round(-1 + 6 * desp))          # the Tome's Dominion modifier from its network state
        for k, s in enumerate(player_inputs(3, TURNS)):
            selected = sel_rng.random() < 0.10 + SEL * desp     # dominion-loop selection rate (salience ~ desperation)
            landed = C.attempt(st, desp, dominion_mod, RESOLVE[nid], rng, selected)
            ghost = copy.deepcopy(brain)                         # counterfactual: no Tome this turn
            d0 = ghost.perceive(s)
            s2, undo = C.apply(brain, st, s, target, graft=landed and st.c > 0.5 and desp > 0.6, transient=TRANSIENT,
                                burst_now=landed or ('--always' in sys.argv))
            d1 = brain.perceive(s2)
            undo()
            in_t += int(d1.dominant_plot == target)
            dt_ = twin.perceive(s)
            in_tw += int(dt_.dominant_plot == target)
            dj = jit.perceive(jitter(s, jr))
            div_base += int(dj.dominant_plot != dt_.dominant_plot)
            cum_plot += int(d1.dominant_plot != dt_.dominant_plot); cum_tac += int(d1.tactic != dt_.tactic)
            changed_plot += int(d1.dominant_plot != d0.dominant_plot)
            changed_tactic += int(d1.tactic != d0.tactic)
            cs.append(st.c)
            C.decay(st)
        print(f"{nid:6s} desperation {desp:.1f}: bursts {st.bursts}/{TURNS} turns (attempts {st.attempts}), mean strength "
              f"{np.mean(cs):.2f}, max {max(cs):.2f} (cap {C.cap(desp):.2f}) | acts differently from its own counterfactual: "
              f"plot {changed_plot/TURNS:.0%}, tactic {changed_tactic/TURNS:.0%} | in Tome-serving plot: {in_t/TURNS:.0%} vs twin {in_tw/TURNS:.0%} | twin divergence: Tome {cum_plot/TURNS:.0%} vs 2% input jitter {div_base/TURNS:.0%}")
