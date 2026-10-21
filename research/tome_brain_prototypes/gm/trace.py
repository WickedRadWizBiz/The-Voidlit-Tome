import jaxpath  # noqa
import jax, jax.numpy as jnp, numpy as np
import l23gm as L
dt=0.05; s=L.init(1); z=jnp.zeros((1,4)); vs=[]; vb=[]; vt=[]; h=[]; nn=[]
step=jax.jit(L.step, static_argnums=())
for k in range(int(80/dt)):
    t=k*dt; i=jnp.array([200.0 if t>=20 else 0.0])
    s,_=step(s,z,z,z,i,dt); vs.append(float(s["v"][0,0])); vb.append(float(s["v"][0,1])); vt.append(float(s["v"][0,2])); h.append(float(s["h"][0])); nn.append(float(s["n"][0]))
vs=np.array(vs)
for t in np.arange(28,40,0.4):
    k=int(t/dt); print(f"t={t:5.1f} Vs={vs[k]:6.1f} Vb={vb[k]:6.1f} Vtr={vt[k]:6.1f} h={h[k]:.2f} n={nn[k]:.2f}")
