import jaxpath  # noqa
import jax, jax.numpy as jnp, numpy as np, importlib
import l23gm as L
from test_l23 import run_step_current, xor
res=[]
for gbar in (2.0, 3.0, 4.0, 6.0):
    for G0 in (5.0, 6.0, 7.0):
        L.DCA_GBAR, L.DCA_G0, L.DCA_SIG, L.DCA_DECAY = gbar, G0, 1.5, 10.0
        rows=[]
        for g in (7, 8, 9, 10):
            a=xor(g,0,0.05); b=xor(g,g,0.05); c=xor(g,0,0.1); d=xor(g,g,0.1)
            rows.append((g,a[0],b[0],c[0],d[0],round(a[3],2),round(b[3],2)))
        fi=[len(run_step_current(0.05,i_amp=i)[0])*5 for i in (150,250,400)]
        print(f"gbar={gbar} G0={G0} fI(150,250,400pA)Hz={fi} XOR rows (g, X spikes, X+Y spikes, @dt0.1 X, X+Y, ampX, ampXY): {rows}", flush=True)
