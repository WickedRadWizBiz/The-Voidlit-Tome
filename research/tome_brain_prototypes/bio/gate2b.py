import warnings; warnings.filterwarnings("ignore")
import numpy as np
import gates2 as G
for w0 in (0.2, 4.0):
    r, wm = G.vogels_cell(w0)
    print(f"2b Vogels iSTDP (eta={G.TJ.VOGELS['eta']}, 100 PV inputs), w0={w0} nS: rate per 10 s {[round(x,1) for x in r]}"
          f" -> last 30 s {np.mean(r[-3:]):.2f} Hz, mean w {wm:.2f} nS  PASS={abs(np.mean(r[-3:]) - 3.0) < 1.0}", flush=True)
r, _ = G.vogels_cell(0.2, eta=0.0, seconds=30.0)
print(f"   mutant eta=0: {[round(x,1) for x in r]} Hz (stays far from 3 Hz -> the test depends on the rule)")
