# Graupner & Brunel 2012 (PNAS) calcium-based bistable synapse, "DP" parameter set, driven by spike pairs.
# c(t) = sum C_pre e^{-(t - t_pre - D)/tau_Ca} + sum C_post e^{-(t - t_post)/tau_Ca}
# tau drho/dt = -rho(1-rho)(rho*-rho) + gp(1-rho)H(c-thp) - gd rho H(c-thd)   (+ noise, off here)
import numpy as np
P = dict(tauCa=20.0, Cpre=1.0, Cpost=2.0, thd=1.0, thp=1.3, gd=200.0, gp=321.808, tau=150e3, rs=0.5, D=13.7)
def protocol(dt_pair, f_hz, n_pairs=60, rho0=0.5, dt=0.1, thp=None):
    thp = P['thp'] if thp is None else thp
    per = 1000.0/f_hz; T = n_pairs*per + 200.0
    n = int(T/dt); c = 0.0; rho = rho0
    t_post = np.arange(n_pairs)*per + 50.0
    t_pre = t_post - dt_pair                       # dt_pair = t_post - t_pre
    ev_pre = np.zeros(n); ev_post = np.zeros(n)
    np.add.at(ev_pre, ((t_pre + P['D'])/dt).astype(int), 1); np.add.at(ev_post, (t_post/dt).astype(int), 1)
    dec = np.exp(-dt/P['tauCa'])
    for i in range(n):
        c = c*dec + P['Cpre']*ev_pre[i] + P['Cpost']*ev_post[i]
        drift = -rho*(1-rho)*(P['rs']-rho) + P['gp']*(1-rho)*(c >= thp) - P['gd']*rho*(c >= thd_(c))
        rho = min(1.0, max(0.0, rho + dt*drift/P['tau']))
    return rho
def thd_(c): return P['thd']
print("STDP window, 60 pairs @1 Hz (rho from 0.5; >0.5 = LTP):")
print("  ", [(d, round(protocol(d, 1.0), 3)) for d in (-80, -40, -20, -10, 0, 10, 20, 40, 80)])
print("Frequency dependence, post-before-pre dt=-10 ms, 60 pairs:")
print("  ", [(f, round(protocol(-10, f), 3)) for f in (1, 5, 10, 20, 40)])
# Bistability: no calcium -> rho relaxes to the nearer well and stays (memory), the permanence mechanism
def relax(r, T_s=600.0, dt_ms=1000.0):
    for _ in range(int(T_s*1000/dt_ms)): r = r + dt_ms*(-r*(1-r)*(P['rs']-r))/P['tau']
    return r
print("Bistable relaxation over 10 min without Ca:", [(r0, round(relax(r0),3)) for r0 in (0.45, 0.55)])
