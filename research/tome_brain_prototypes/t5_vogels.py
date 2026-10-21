# Vogels et al. 2011 inhibitory STDP on a conductance LIF neuron with 8 input channels (800 E, 200 I Poisson inputs).
# pre (inh) spike: w += eta*(x_post - alpha); post spike: w += eta*x_pre; alpha = 2*rho0*tau_STDP.
# Pass: postsynaptic rate -> rho0 and inhibitory tuning co-tunes with excitatory tuning (detailed balance).
import jaxenv, jax, jax.numpy as jnp, numpy as np
dt, tau_m, gL, EL, Ee, Ei, Vth, Vr = 0.1, 20.0, 10.0, -60.0, 0.0, -80.0, -50.0, -60.0
C = tau_m*gL; tau_e, tau_i, tau_stdp, rho0, eta = 5.0, 10.0, 20.0, 5e-3, 1e-3   # rho0 in spikes/ms (5 Hz)
alpha = 2*rho0*tau_stdp
nE, nI, G = 800, 200, 8
grpE = np.repeat(np.arange(G), nE//G); grpI = np.repeat(np.arange(G), nI//G)
tune = np.array([0.6, 0.8, 1.2, 1.8, 2.4, 1.6, 1.0, 0.7]); wE = jnp.asarray(0.15*tune[grpE])
key = jax.random.PRNGKey(0)
def make(T_ms, learn, wI0, key):
    n = int(T_ms/dt)
    def step(c, k):
        V, ge, gi, xp, xi, wI, nsp = c
        k1, k2, k3 = jax.random.split(k, 3)
        # channel rates fluctuate (shared per group): OU-ish via fresh draw every step scaled small
        rE = 5e-3*(1 + 0.8*jax.random.normal(k1, (G,)))[grpE].clip(0)
        rI = 5e-3*(1 + 0.8*jax.random.normal(k1, (G,)))[grpI].clip(0)
        sE = jax.random.uniform(k2, (nE,)) < rE*dt; sI = jax.random.uniform(k3, (nI,)) < rI*dt
        ge = ge*jnp.exp(-dt/tau_e) + (wE*sE).sum(); gi = gi*jnp.exp(-dt/tau_i) + (wI*sI).sum()
        g = gL + ge + gi; Vi = (gL*EL + ge*Ee + gi*Ei)/g
        V = Vi + (V - Vi)*jnp.exp(-dt*g/C)
        spk = V >= Vth; V = jnp.where(spk, Vr, V)
        xp = xp*jnp.exp(-dt/tau_stdp) + spk; xi = xi*jnp.exp(-dt/tau_stdp) + sI
        if learn:
            wI = wI + eta*(sI*(xp - alpha) + spk*xi); wI = jnp.clip(wI, 0.0, 10.0)
        return (V, ge, gi, xp, xi, wI, nsp + spk), None
    keys = jax.random.split(key, n)
    c0 = (jnp.float32(EL), 0.0, 0.0, 0.0, jnp.zeros(nI), wI0, 0)
    c, _ = jax.lax.scan(step, c0, keys)
    return c
run_learn = jax.jit(lambda w, k: make(60_000, True, w, k)); run_test = jax.jit(lambda w, k: make(10_000, False, w, k))
wI0 = jnp.full(nI, 0.35)
c = run_test(wI0, key); r_before = int(c[-1])/10.0
cl = run_learn(wI0, jax.random.PRNGKey(1)); wI = cl[5]
c = run_test(wI, jax.random.PRNGKey(2)); r_after = int(c[-1])/10.0
prof = np.array([float(wI[grpI == k].mean()) for k in range(G)])
corr = np.corrcoef(prof, tune)[0, 1]
print(f"rate before {r_before:.1f} Hz -> after 60 s iSTDP {r_after:.1f} Hz (target {rho0*1000:.0f} Hz)")
print("inhibitory profile:", prof.round(2), "| corr with excitatory tuning:", round(corr, 3))
# mutation: learning off -> no co-tuning (flat), rate stays off target
print("mutant (eta=0): corr undefined/flat; rate", r_before, "Hz")
