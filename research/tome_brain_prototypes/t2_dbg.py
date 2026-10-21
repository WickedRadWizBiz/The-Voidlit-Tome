exec(open("t2_dcaap.py").read().split("# 1) find")[0])
for g in (0.1, 0.3, 0.6, 1.0, 2.0):
    print(g, sim(g, 0., jax.random.PRNGKey(0), 1e9))
