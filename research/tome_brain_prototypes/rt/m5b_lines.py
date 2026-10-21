import time, torch, re, inspect, textwrap
torch.set_num_threads(1)
import cosim as CS
sj = CS.SJPart(0)
src = textwrap.dedent(inspect.getsource(CS.SJPart.step_window))
lines = src.split("\n")
# instrument: time cumulative per top-level statement (rough)
body = []
for i, ln in enumerate(lines):
    body.append(ln)
    if ln.startswith("        ") and not ln.startswith("         ") and not ln.strip().startswith(("for", "#", '"""', "return", "def")) and ln.strip() and not ln.rstrip().endswith((":", ",", "(")):
        body.append(f"        _T.append(({i}, time.perf_counter()))")
code = "\n".join(body).replace('        D = ctx.shape[0]', '        _T.append((0, time.perf_counter()))\n        D = ctx.shape[0]', 1)
ns = {"torch": torch, "time": time, "N_CH": CS.N_CH, "np": CS.np}
exec(code, ns)
f = ns["step_window"]
D = 4
ctx = (torch.rand(D, 256) < 0.01).float(); mf = (torch.rand(D, 128) < 0.03).float()
acc = {}
with torch.no_grad():
    for k in range(300):
        ns["_T"] = []
        ns["_T"].clear()
        import builtins
        _T = []
        ns["_T"] = _T
        f(sj, ctx, torch.zeros(D, 64), mf, torch.zeros(D))
        for (a, ta), (b, tb) in zip(_T, _T[1:]):
            acc[b] = acc.get(b, 0) + (tb - ta)
for b, v in sorted(acc.items(), key=lambda x: -x[1])[:10]:
    print(f"{v/300*1e6:7.1f} us  line {b}: {lines[b].strip()[:90]}")
