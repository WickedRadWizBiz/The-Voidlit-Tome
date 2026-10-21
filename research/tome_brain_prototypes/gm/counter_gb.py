"""Permanent counter-adaptation as Graupner-Brunel bistable synapses (theta_d = 1.0, theta_p = 1.3).

One 'counter' synapse per player tactic. Pre = the tactic was used; post = it hurt the Tome (a Tome loss). Each paired
event adds a calcium transient; calcium decays with tau_ca (in turns).  Spaced abuse keeps peak Ca between theta_d and
theta_p (LTD: the synapse stays DOWN); clustered abuse sums above theta_p (LTP: it crosses rho* into the UP well), and
the bistable drift keeps it UP with no further drive: a permanent counter.  Exponential-Euler substeps (10 per turn).
"""
import numpy as np

THETA_D, THETA_P, RHO_STAR = 1.0, 1.3, 0.5
GAMMA_P, GAMMA_D, TAU_RHO = 40.0, 2.0, 2.0      # per turn
TAU_CA, CA_JUMP, SUB = 1.5, 0.82, 10


def run(events, n_turns, rho0=0.15):
    rho, ca, trace = rho0, 0.0, []
    dt = 1.0 / SUB
    for t in range(n_turns):
        if t in events:
            ca += CA_JUMP
        for _ in range(SUB):
            drift = -rho * (1 - rho) * (RHO_STAR - rho)
            ltp = GAMMA_P * (1 - rho) * (ca >= THETA_P)
            ltd = -GAMMA_D * rho * (ca >= THETA_D)
            rho = float(np.clip(rho + dt * (drift + ltp + ltd) / TAU_RHO, 0, 1))
            ca *= np.exp(-dt / TAU_CA)
        trace.append(rho)
    return np.array(trace)


cases = {
    "never used": set(),
    "used once": {5},
    "spaced: every 4th turn x6 (t=5..25)": set(range(5, 29, 4)),
    "spaced: every 3rd turn x6": set(range(5, 23, 3)),
    "alternate turns x6": set(range(5, 17, 2)),
    "clustered: 2 in a row": {5, 6},
    "clustered: 3 in a row (abuse)": {5, 6, 7},
    "clustered: 4 of 5 turns": {5, 6, 8, 9},
}
for name, ev in cases.items():
    tr = run(ev, 120)
    print(f"{name:38s} rho@turn10={tr[10]:.2f} rho@30={tr[30]:.2f} rho@120 (no further use)={tr[-1]:.2f} "
          f"-> {'UP (permanent counter)' if tr[-1] > RHO_STAR else 'DOWN'}")
# mutation: theta_p = 1.6 (the spec's value) - clustered abuse can no longer reach LTP
THETA_P = 1.6
for ev, nm in (({5, 6, 7}, "3 in a row"), (set(range(5, 11)), "6 in a row")):
    print(f"MUTANT theta_p=1.6 (spec), {nm}:", "UP" if run(ev, 120)[-1] > RHO_STAR else "DOWN")
