src = open("t2_var.py").read().split("# 1) find")[0]
src = src.replace("return (Vs_n, Vb_n, Va_n, w_n, ref, gA, gNr, gN, s1, s2, aref, A, Ion, spk_n + spk), (Va_n, Isyn, onset)",
                  "return (Vs_n, Vb_n, Va_n, w_n, ref, gA, gNr, gN, s1, s2, aref, A, Ion, spk_n + spk), (Va_n, Isyn, onset, Vs_n, gd)")
src = src.replace("cf, (Va, Isyn, on) = jax.lax.scan(step, c0, inc)\n    return cf[-1], Va, Isyn, on",
                  "cf, (Va, Isyn, on, Vs, gd) = jax.lax.scan(step, c0, inc)\n    return cf[-1], Va, Isyn, on, Vs, gd")
exec(src)
for g in (1.7, 2.2, 3.4):
    n, Va, Isyn, on, Vs, gd = _core(jnp.float32(g), jnp.float32(0.), jax.random.PRNGKey(1), jnp.float32(218.))
    Va, Vs, gd = map(np.asarray, (Va, Vs, gd))
    print(g, "spikes", int(n), "maxVa", Va.max().round(1), "maxVs", Vs.max().round(1), "max gd", gd.max().round(1), "onset idx", np.nonzero(np.asarray(on))[0][:3])
