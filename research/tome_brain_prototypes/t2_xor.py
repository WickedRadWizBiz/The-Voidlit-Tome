import sys
gsa, gdca, tauA, td = map(float, sys.argv[1:5])
src = open("t2_dcaap.py").read().split("# 1) find")[0]
src = src.replace("gsa=2.0,", f"gsa={gsa},").replace("gdca=14.,", f"gdca={gdca},").replace("tauA=0.3)", f"tauA={tauA})").replace("tr=1.5, td=8.,", f"tr=1.5, td={td},")
exec(src)
seeds = [jax.random.PRNGKey(100+s) for s in range(20)]
# calibrate: weakest per-synapse weight at which >=50% of seeds produce a dCaAP; Ith = median onset current there
for g in np.arange(0.8, 4.0, 0.04):
    ons = [sim(g, 0., k, 1e9)[2] for k in seeds]
    hits = [o for o in ons if o is not None]
    if len(hits) >= 10: gth, Ith = float(g), float(np.median(hits)); break
print(f"threshold g={gth:.2f} nS/syn  Ith={Ith:.0f} pA")
for m in (1.1, 1.25, 1.4, 1.6):
    g = gth*m; r = {}
    for lab, (x, y) in {"00":(0,0), "10":(g,0), "01":(0,g), "11":(g,g)}.items():
        r[lab] = sum(int(sim(x, y, k, Ith)[0] > 0) for k in seeds)/20
    print(f"  {m:.2f}x  00={r['00']:.2f} 10={r['10']:.2f} 01={r['01']:.2f} 11={r['11']:.2f}")
P['tauA'] = 1e9; _core = jax.jit(simulate, static_argnames=("T","nsyn"))
r11 = sum(int(sim(gth*1.25, gth*1.25, k, Ith)[0] > 0) for k in seeds)/20
r10 = sum(int(sim(gth*1.25, 0, k, Ith)[0] > 0) for k in seeds)/20
print(f"  mutant monotonic dCaAP at 1.25x: 10={r10:.2f} 11={r11:.2f}")
