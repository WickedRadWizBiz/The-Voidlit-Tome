import warnings; warnings.filterwarnings("ignore")
import corruption as K
from voidlit.tome import load_scenario
spec = load_scenario().npcs["elara"]
off_i, mis = spec.plot_index("offering"), [t.id for t in spec.tactics].index("misdirect")
for gain in (3.0, 5.0, 8.0):
    K.W_PLOT, K.W_ANTI, K.W_TACTIC = 0.035 * gain, -0.02 * gain, 0.03 * gain
    row = []
    for s in (0.0, 0.1, 0.25, 0.5, 1.0):
        b, ds = K.run(lambda k, s=s: s)
        row.append(f"s={s:.2f}: dom {sum(d.dominant_plot == off_i for d in ds)/15:.2f} tac {sum(d.tactic == mis for d in ds)/15:.2f}")
    print(f"gain {gain}: " + " | ".join(row), flush=True)
