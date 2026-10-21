import warnings; warnings.filterwarnings("ignore")
import corruption as K
from voidlit.tome import load_scenario
spec = load_scenario().npcs["elara"]
off_i, mis = spec.plot_index("offering"), [t.id for t in spec.tactics].index("misdirect")
K.W_PLOT, K.W_ANTI, K.W_TACTIC, K.S_MAX = 0.07, -0.04, 0.06, 0.35
row = []
for s in (0.0, 0.05, 0.1, 0.2, 0.35, 1.0):
    b, ds = K.run(lambda k, s=s: s)
    row.append(f"s={s:.2f}: dominant {sum(d.dominant_plot == off_i for d in ds)/15:.2f}, misdirect {sum(d.tactic == mis for d in ds)/15:.2f}")
print("gain 2, cap 0.35 ->", " | ".join(row))
b, ds = K.run(lambda k: 0.35 if k < 6 else 0.0, turns=20)
print("recovery (cap for turns 0-5, then silent): tactic", "".join("M" if d.tactic == mis else "." for d in ds),
      "| offering momentum", [round(float(d.momentum[off_i]), 2) for d in ds][::2])
