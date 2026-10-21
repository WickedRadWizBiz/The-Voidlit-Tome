# Human L2/3 PC: soma (AdEx) + basal + apical tuft, conductance synapses (AMPA, NMDA with Mg block),
# phenomenological dCaAP after Gidon et al. 2020: threshold-triggered Ca conductance whose amplitude
# decays with the suprathreshold synaptic drive at onset. XOR must EMERGE from synaptic volleys X, Y.
import jaxenv, jax, jax.numpy as jnp, numpy as np
dt = 0.25
P = dict(Cs=150., gLs=7.5, Cb=100., gLb=5., Ca=80., gLa=4., gsb=20., gsa=2.0, EL=-70., VT=-50., DT=2.,
         Vr=-58., Vpk=-30., a=2., b=60., tw=150., Vth_a=-42., dca_ref=60., gdca=14., Edca=60.,
         tr=1.5, td=8., tauA=0.3)
def B(V): return 1./(1. + jnp.exp(-0.062*V)/3.57)          # Jahr & Stevens 1990, [Mg2+] = 1 mM
def simulate(gX, gY, key, Ith, T=80., nsyn=20, jitter=2.0, nmda=0.5):
    n = int(T/dt)
    k1, k2 = jax.random.split(key)
    tX = 20. + jitter*jax.random.normal(k1, (nsyn,)); tY = 20. + jitter*jax.random.normal(k2, (nsyn,))
    ts = jnp.arange(n)*dt
    # per-step summed conductance increments from the two pathways (tuft synapses)
    inc = (gX*((tX[None,:] >= ts[:,None]) & (tX[None,:] < ts[:,None]+dt)).sum(1)
           + gY*((tY[None,:] >= ts[:,None]) & (tY[None,:] < ts[:,None]+dt)).sum(1))
    def step(c, x):
        Vs, Vb, Va, w, ref, gA, gNr, gN, s1, s2, aref, A, Ion, spk_n = c
        gA = gA*jnp.exp(-dt/2.) + x
        gNr = gNr*jnp.exp(-dt/2.) + nmda*x                 # NMDA rise stage (tau_r = 2 ms)
        gN = gNr + (gN - gNr)*jnp.exp(-dt/60.)             # NMDA decay stage (tau_d = 60 ms)
        Isyn = gA*(0.-Va) + gN*B(Va)*(0.-Va)
        # dCaAP trigger on upward crossing of Vth_a
        onset = (Va >= P['Vth_a']) & (aref <= 0.)
        A = jnp.where(onset, jnp.exp(-jnp.maximum(0., Isyn - Ith)/(P['tauA']*Ith)), A)
        s1 = jnp.where(onset, s1 + 1., s1); s2 = jnp.where(onset, s2 + 1., s2)
        s1 = s1*jnp.exp(-dt/P['td']); s2 = s2*jnp.exp(-dt/P['tr'])
        aref = jnp.where(onset, P['dca_ref'], jnp.maximum(aref - dt, 0.))
        gd = P['gdca']*A*(s1 - s2)/0.6   # normalised double exponential
        # tuft (exp-Euler, conductances frozen over the step)
        ga = P['gLa'] + P['gsa'] + gA + gN*B(Va) + gd
        Vai = (P['gLa']*P['EL'] + P['gsa']*Vs + gd*P['Edca'])/ga
        Va_n = Vai + (Va - Vai)*jnp.exp(-dt*ga/P['Ca'])
        gb = P['gLb'] + P['gsb']; Vbi = (P['gLb']*P['EL'] + P['gsb']*Vs)/gb
        Vb_n = Vbi + (Vb - Vbi)*jnp.exp(-dt*gb/P['Cb'])
        gs = P['gLs'] + P['gsb'] + P['gsa']
        Vsi = (P['gLs']*P['EL'] + P['gsb']*Vb + P['gsa']*Va + P['gLs']*P['DT']*jnp.exp(jnp.minimum((Vs-P['VT'])/P['DT'], 20.)) - w)/gs
        Vs_n = Vsi + (Vs - Vsi)*jnp.exp(-dt*gs/P['Cs'])
        Vs_n = jnp.where(ref > 0, P['Vr'], Vs_n)
        w_n = P['a']*(Vs-P['EL']) + (w - P['a']*(Vs-P['EL']))*jnp.exp(-dt/P['tw'])
        spk = Vs_n >= P['Vpk']
        Vs_n = jnp.where(spk, P['Vr'], Vs_n); w_n = jnp.where(spk, w_n + P['b'], w_n)
        ref = jnp.where(spk, 2., jnp.maximum(ref - dt, 0.))
        return (Vs_n, Vb_n, Va_n, w_n, ref, gA, gNr, gN, s1, s2, aref, A, Ion, spk_n + spk), (Va_n, Isyn, onset)
    c0 = (P['EL'], P['EL'], P['EL'], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0)
    c0 = tuple(jnp.asarray(v, jnp.float32) if not isinstance(v, int) else jnp.asarray(v) for v in c0)
    cf, (Va, Isyn, on) = jax.lax.scan(step, c0, inc)
    return cf[-1], Va, Isyn, on
_core = jax.jit(simulate, static_argnames=("T","nsyn"))
def sim(gX, gY, key, Ith):
    n, Va, Isyn, on = _core(jnp.float32(gX), jnp.float32(gY), key, jnp.float32(Ith))
    on = np.asarray(on); Isyn = np.asarray(Isyn)
    return int(n), float(np.asarray(Va).max()), (float(Isyn[on][0]) if on.any() else None)
key = jax.random.PRNGKey(0)
# 1) find the single-pathway dCaAP threshold (weight per synapse, nS) and the onset current there
gs_ = np.round(np.arange(0.5, 3.01, 0.02), 3)
first = None
for g in gs_:
    _, vmax, ion = sim(g, 0., key, 1e9)
    if ion is not None: first = (g, ion); break
gth, Ith = first
print(f"single-pathway dCaAP threshold g={gth} nS/synapse, onset current {Ith:.0f} pA")
# 2) XOR truth table across strengths 1.0-1.6x threshold and 30 jitter seeds
rows = []
for m in (1.0, 1.15, 1.3, 1.45, 1.6):
    g = gth*m; res = {"00":0, "10":0, "01":0, "11":0}
    for s in range(30):
        k = jax.random.PRNGKey(100+s)
        for lab, (x, y) in {"00":(0,0), "10":(g,0), "01":(0,g), "11":(g,g)}.items():
            n, vmax, _ = sim(x, y, k, Ith)
            res[lab] += int(n > 0)
    rows.append((m, res))
    print(f"strength {m:.2f}x: P(soma spikes) 00={res['00']/30:.2f} 10={res['10']/30:.2f} 01={res['01']/30:.2f} 11={res['11']/30:.2f}")
# 3) mutation: monotonic dCaAP (amplitude does not decay with drive) -> XOR must fail
P['tauA'] = 1e9
_core = jax.jit(simulate, static_argnames=("T","nsyn"))
n11 = sum(int(sim(gth*1.3, gth*1.3, jax.random.PRNGKey(100+s), Ith)[0] > 0) for s in range(30))
print("mutant (monotonic dCaAP), 11 at 1.3x: P(spike) =", n11/30)
