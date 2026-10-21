import jaxpath  # noqa
import jax, jax.numpy as jnp, numpy as np
import l23gm as L
from test_l23 import run_step_current, xor
for i in (100, 150, 200, 300, 400):
    t_sp, vmax, nan = run_step_current(0.05, i_amp=i)
    isi = np.diff(t_sp)
    print(f"I={i:4d} pA: {len(t_sp)} spikes/200 ms ({len(t_sp)*5} Hz) first {t_sp[:1]} ISIs {np.round(isi[:5],1)} peak {vmax:.1f}")
# measure apical drive at dCaAP onset for single-pathway conductances
def onset_drive(g, dt=0.05):
    s = L.init(1); k_on=int(10/dt)
    for k in range(int(40/dt)):
        gg = g if k==k_on else 0.0
        wa = jnp.zeros((1,4)).at[:,2].set(gg); wn = 0.5*wa
        prev = s["ca_ref"][0]
        ga = s["ga"]*jnp.exp(-dt/L.TAU_AMPA)+wa
        s,_ = L.step(s, wa, wn, jnp.zeros((1,4)), jnp.zeros(1), dt)
        if s["ca_ref"][0] > prev:
            gnr=s["gnr"];gnd=s["gnd"]; mg=1/(1+jnp.exp(-0.062*s["v"])/3.57)
            return float(s["ga"][0,2]+s["ga"][0,3]), float(s["v"][0,2])
    return None
for g in (4,5,6,7,8,9,10,12):
    print("g", g, "onset (ampa drive, Vtrunk):", onset_drive(g))
