from calib_l23 import *  # reruns the sweeps; fine
print("fine sweep:")
for gd in (0.06, 0.07, 0.08, 0.09):
    out = {k: np.round(summary(*sim(1.1 * sum(k), 0.008, gd, steps)), 1).tolist() for k in ((1, 0), (1, 1))}
    mono = np.round(summary(*sim(2.2, 0.008, gd, steps, monotone=True)), 1).tolist()
    print(f"  g_dCaAP={gd}: {out} mono(1,1)={mono}")
print("dt robustness of the XOR at g_dCaAP=0.08:")
for dt in (0.0125, 0.025, 0.05, 0.1):
    st = int(60 / dt)
    out = {k: np.round(summary(*sim(1.1 * sum(k), 0.008, 0.08, st, dt=dt)), 1).tolist() for k in ((0,0),(1, 0), (1, 1))}
    print(f"  dt={dt}: {out}")
print("graded input sweep (single pathway drive / threshold) at g=0.08 -> tuft peak, soma spikes")
for amp in (0.9, 1.0, 1.05, 1.2, 1.4, 1.7, 2.0, 2.5):
    print("  ", amp, np.round(summary(*sim(amp, 0.008, 0.08, steps)), 1).tolist())
