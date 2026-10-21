# De Pitta et al. 2009 G-ChI astrocyte (parameters as tabulated in Stimberg et al. 2019, Brian 2 neuron-glia models).
# Integrated with exponential (Rush-Larsen-style) updates at a coarse step; checks oscillation (FM encoding) and
# agreement with a fine-step reference.
import numpy as np
p = dict(C0=2.0, c1=0.185, rC=6.0, rL=0.11, vER=0.9, KER=0.1, d1=0.13, d2=1.049, d3=0.9434, d5=0.08234, a2=0.2,
         vb=0.5, KR=1.3, Kp=10.0, Kpi=0.6, vd=0.02, kd=1.5, KPLCd=0.1, v3K=2.0, KD=0.7, K3=1.0, r5P=0.04)   # uM, s
def run(glu, dt, T=120.0):
    C, h, I = 0.05, 0.9, 0.05
    out = []
    for k in range(int(T/dt)):
        minf = I/(I + p['d1']); ninf = C/(C + p['d5'])
        Jchan = p['rC']*minf**3*ninf**3*h**3*(p['C0'] - (1 + p['c1'])*C)
        Jleak = p['rL']*(p['C0'] - (1 + p['c1'])*C)
        Jpump = p['vER']*C**2/(C**2 + p['KER']**2)
        Q2 = p['d2']*(I + p['d1'])/(I + p['d3']); hinf = Q2/(Q2 + C); tauh = 1.0/(p['a2']*(Q2 + C))
        Jb = p['vb']*glu**0.7/(glu**0.7 + (p['KR'] + p['Kp']*C/(C + p['Kpi']))**0.7)
        Jd = p['vd']/(1 + I/p['kd'])*C**2/(C**2 + p['KPLCd']**2)
        J3K = p['v3K']*C**4/(C**4 + p['KD']**4)*I/(I + p['K3']); J5P = p['r5P']*I
        # Ca: semi-implicit in its own decay (exp-Euler on the linear-in-C part of leak/channel)
        lin = (p['rC']*minf**3*ninf**3*h**3 + p['rL'])*(1 + p['c1'])
        src = (p['rC']*minf**3*ninf**3*h**3 + p['rL'])*p['C0'] - Jpump
        C = max(1e-6, src/lin + (C - src/lin)*np.exp(-lin*dt))
        h = hinf + (h - hinf)*np.exp(-dt/tauh)
        I = max(1e-6, I + dt*(Jb + Jd - J3K - J5P))
        out.append(C)
    return np.array(out)
def peaks(x, dt, thr=0.4):
    up = np.nonzero((x[1:] >= thr) & (x[:-1] < thr))[0]
    return len(up)/(len(x)*dt)*60.0   # oscillations per minute
for glu in (0.0, 0.5, 2.0, 10.0, 50.0):
    ref = run(glu, 0.001); co = run(glu, 0.05)
    print(f"glutamate {glu:5.1f} uM: Ca peaks/min ref(dt=1ms)={peaks(ref, 0.001):5.1f}  coarse(dt=50ms)={peaks(co, 0.05):5.1f}  "
          f"max Ca ref={ref.max():.2f} coarse={co.max():.2f}  mean Ca ref={ref[20000:].mean():.3f} coarse={co[400:].mean():.3f}")
