"""Graupner-Brunel (2012) synapse as the 'permanent counter-adaptation' memory.
Calcium model: c jumps by C_pre at pre spike (after delay D) and C_post at post spike, decays with tau_Ca.
rho: tau drho/dt = -rho(1-rho)(rho*-rho) + gp(1-rho)H(c-tp) - gd rho H(c-td)  (noise omitted: deterministic game)
A turn = 200 ms of network time; between turns, a fixed 20 s gap of drift-only consolidation (no calcium).
Parameter set: GB 2012 cortical fit as recalled (verify against the paper's table before shipping)."""
import numpy as np
tau_ca, C_pre, C_post, D = 22.7, 0.56, 1.24, 4.6
td, tp, gd, gp, tau, rho_star = 1.0, 1.3, 332.0, 725.0, 346_000.0, 0.5   # tau in ms

def turn(rho, abuse, f=20.0, lag=10.0, dt=0.1, T=200.0):
    """abuse: exploit cell (pre) and counter MSN (post) fire paired at f Hz, post 'lag' ms after pre."""
    c, t = 0.0, 0.0
    period = 1000.0 / f
    pre_t = np.arange(0, T, period) + D if abuse else np.array([])
    post_t = np.arange(0, T, period) + lag if abuse else np.array([])
    for k in range(int(T / dt)):
        t = k * dt
        c *= np.exp(-dt / tau_ca)
        if np.any(np.abs(pre_t - t) < dt / 2): c += C_pre
        if np.any(np.abs(post_t - t) < dt / 2): c += C_post
        drift = -rho * (1 - rho) * (rho_star - rho)
        rho += dt * (drift + gp * (1 - rho) * (c >= tp) - gd * rho * (c >= td)) / tau
    return float(np.clip(rho, 0, 1))

def gap(rho, seconds=20.0, dt=50.0):
    for _ in range(int(seconds * 1000 / dt)):
        rho += dt * (-rho * (1 - rho) * (rho_star - rho)) / tau
    return rho

for k in (1, 2, 3, 4, 6, 10):
    rho = 0.05
    for _ in range(k):
        rho = gap(turn(rho, True))
    after_abuse = rho
    for _ in range(300):
        rho = gap(turn(rho, False))
    print(f"{k:2d} abusive turns: rho after abuse {after_abuse:.3f} -> after 300 quiet turns {rho:.3f} ({'UP' if rho > 0.5 else 'DOWN'})")
# low-frequency (1 Hz-like sparse) pairing inside a turn: does a single coincidence potentiate?
print("single pairing in a turn:", round(gap(turn(0.05, True, f=5.0)), 4))
