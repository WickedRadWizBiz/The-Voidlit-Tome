# Gate: AdEx soma with exponential-Euler (exact linear part, frozen exp term) and exact w update.
# f-I curve at dt = 0.25 ms vs a dt = 0.005 ms reference (both exp-Euler; reference ~ exact).
import jaxenv, jax, jax.numpy as jnp, numpy as np
C, gL, EL, VT, DT, Vr, Vpk, a, b, tw, tref = 150., 7.5, -70., -50., 2.0, -58., -30., 2.0, 60., 150., 2.0
def run(I, dt, T=2000.):
    n = int(T/dt)
    def step(c, _):
        V, w, ref = c
        ginf = gL
        Vinf = (gL*EL + gL*DT*jnp.exp(jnp.minimum((V-VT)/DT, 20.)) - w + I)/ginf
        Vn = Vinf + (V - Vinf)*jnp.exp(-dt*ginf/C)
        Vn = jnp.where(ref > 0, Vr, Vn)
        winf = a*(V-EL)
        wn = winf + (w - winf)*jnp.exp(-dt/tw)
        spk = Vn >= Vpk
        Vn = jnp.where(spk, Vr, Vn); wn = jnp.where(spk, wn + b, wn)
        refn = jnp.where(spk, tref, jnp.maximum(ref - dt, 0.))
        return (Vn, wn, refn), spk
    _, s = jax.lax.scan(step, (jnp.array(EL), jnp.array(0.), jnp.array(0.)), None, length=n)
    s = np.asarray(s); t = np.nonzero(s)[0]*dt
    late = t[t > 1000.]
    return len(late)  # Hz over the last 1 s (adapted rate)
Is = [150, 200, 300, 400, 600]
for dt in (0.005, 0.1, 0.25, 0.5):
    print(f"dt={dt:<6}", [run(I, dt) for I in Is])
