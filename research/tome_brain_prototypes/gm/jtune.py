import warnings; warnings.filterwarnings("ignore")
import jaxpath  # noqa
import jax, jax.numpy as jnp, numpy as np, torch
import hybrid as Hy
for we, wi in ((0.30, -1.0), (0.15, -3.0), (0.10, -3.0), (0.06, -2.0)):
    src = open("hybrid.py").read()
    G = Hy.j_build(jax.random.PRNGKey(1))
    inh = G["inhib"]
    W = (G["W"] != 0) * jnp.abs(G["W"])
    G["W"] = W * jnp.where(inh[:, None] > 0, wi, we)
    ep = Hy.make_j_epoch(G); st = Hy.j_init(G); key = jax.random.PRNGKey(0)
    a, b = G["off"]["core"]
    tot = 0; tot23 = 0
    rng = np.random.default_rng(0)
    for e in range(63):
        ext = np.zeros((4, G["n"]), np.float32)
        ext[:, a:b] = 1.1 * (rng.random((4, b - a)) < 0.012)
        st, key, s, s23 = ep(st, key, jnp.asarray(ext))
        tot += float(s.sum()); tot23 += float(s23.sum())
    print(f"E={we} I={wi}: point mean {tot/G['n']/0.252:.1f} Hz, L2/3 mean {tot23/192/0.252:.1f} Hz")
