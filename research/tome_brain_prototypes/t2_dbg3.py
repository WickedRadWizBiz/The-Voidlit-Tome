src = open("t2_dcaap.py").read().split("# 1) find")[0]
src = src.replace("gsa=2.0,", "gsa=8.0,").replace("gdca=14.,", "gdca=45.,").replace("tauA=0.3)", "tauA=1e9)").replace("tr=1.5, td=8.,", "tr=1.5, td=15.,")
src = src.replace("return (Vs_n, Vb_n, Va_n, w_n, ref, gA, gNr, gN, s1, s2, aref, A, Ion, spk_n + spk), (Va_n, Isyn, onset)",
                  "return (Vs_n, Vb_n, Va_n, w_n, ref, gA, gNr, gN, s1, s2, aref, A, Ion, spk_n + spk), (Va_n, Isyn, onset, Vs_n, gd)")
src = src.replace("cf, (Va, Isyn, on) = jax.lax.scan(step, c0, inc)\n    return cf[-1], Va, Isyn, on",
                  "cf, (Va, Isyn, on, Vs, gd) = jax.lax.scan(step, c0, inc)\n    return cf[-1], Va, Isyn, on, Vs, gd")
exec(src)
g = 1.92*1.25
for lab,(x,y) in {"10":(g,0),"11":(g,g)}.items():
    n, Va, Isyn, on, Vs, gd = _core(jnp.float32(x), jnp.float32(y), jax.random.PRNGKey(101), jnp.float32(355.))
    Va, Vs, gd, on = map(np.asarray, (Va, Vs, gd, on))
    i = np.nonzero(on)[0]
    print(lab, "spk", int(n), "onsets", i*0.25, "maxVa", Va.max().round(1), "maxVs", Vs.max().round(1), "gd max", gd.max().round(1), "Va@+5ms", Va[i[0]+20].round(1) if len(i) else None)
