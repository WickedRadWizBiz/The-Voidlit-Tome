"""One clock, one spike bus, three frameworks. Per 5-ms window k:
   1. dispatch JAX window k (async)          inputs: bus spikes from BindsNET/SJ windows <= k-1
   2. run BindsNET window k (main thread)    inputs: JAX cortical spikes from windows <= k-1
   3. run SpikingJelly window k              inputs: JAX spikes from windows <= k-1
   4. barrier: block on JAX k; publish all window-k spikes to the bus (DLPack, zero-copy)
Exact because every cross-partition pathway has delay >= 5 ms (conservative min-delay synchronisation)."""
import sys, time, statistics, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
import jaxenv, jax, jax.numpy as jnp
from core_C import build, make_window, init_state, WIN_MS, DMAX
exec(open("time_bindsnet2.py").read().split("for threads in")[0])
torch.set_num_threads(1)
# --- JAX core
netj = build(0); window, N = make_window(netj)
s = init_state(N); ring = jnp.zeros((DMAX, N)); stp = (jnp.full(N, 0.2), jnp.ones(N))
delays = jnp.asarray(np.clip(np.round(netj["L"] / 15.0), 3, 12).astype(np.int32)); key = jax.random.PRNGKey(0)
ext = jnp.zeros((WIN_MS, 5, N))
# --- BindsNET lean partition
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, IzhikevichNodes
from bindsnet.network.topology import Connection
from bindsnet.learning import MSTDPET
bn = Network(dt=1.0, batch_size=1, learning=True)
ctx = Input(n=160, traces=True); sens = Input(n=48, traces=True)
STR = IzhikevichNodes(n=315, excitatory=1.0, traces=True); BLA = LIFNodes(n=50, thresh=-52.0, rest=-65.0, reset=-65.0, tc_decay=20.0, traces=True)
for nm, l in (("ctx", ctx), ("sens", sens), ("STR", STR), ("BLA", BLA)): bn.add_layer(l, name=nm)
kw = dict(update_rule=MSTDPET, nu=1.0, tc_plus=20.0, tc_minus=20.0, tc_e_trace=250.0, wmin=0.0, wmax=1.0)
bn.add_connection(Connection(ctx, STR, w=0.3*(torch.rand(160, 315) < 0.2).float(), **kw), "ctx", "STR")
bn.add_connection(Connection(STR, STR, w=-0.3*(torch.rand(315, 315) < 0.1).float()), "STR", "STR")
bn.add_connection(Connection(sens, BLA, w=0.4*(torch.rand(48, 50) < 0.3).float(), **kw), "sens", "BLA")
bn.add_connection(Connection(ctx, BLA, w=0.3*(torch.rand(160, 50) < 0.1).float()), "ctx", "BLA")
bn.add_connection(Connection(BLA, STR, w=0.2*(torch.rand(50, 315) < 0.1).float()), "BLA", "STR")
# --- SJ cerebellum
src = open("time_sj2.py").read().split("mf = (torch.rand")[0].replace("def window(mf, actual)", "def window_sj(mf, actual)")
exec(src)
# projection index sets on the bus (global ids of JAX neurons feeding the other partitions)
g = netj["groups"]
ctx_ids = np.concatenate([np.arange(*g[f"c{c}.L5_IT"]) for c in (2, 3, 4)] + [np.arange(*g[f"c{c}.L5_ET"]) for c in (2, 3, 4)] + [np.arange(*g[f"c{c}.L23_IT"])[:20] for c in (2, 4)])
mf_ids = np.concatenate([np.arange(*g[f"c{c}.L5_ET"]) for c in (2, 3, 4)] + [np.arange(*g["c0.L5_IT"])])[:64]
ctx_ids_t = torch.as_tensor(ctx_ids); mf_ids_t = torch.as_tensor(mf_ids)
def epoch(state, m0):
    s, ring, stp = state
    last_cnt = torch.zeros(WIN_MS, N)
    global tot_spk
    tot_spk = []
    t_wait = 0.0
    for k in range(50):
        out = window(s, ring, stp, m0 + k*WIN_MS, delays, key, ext)          # async dispatch
        # BindsNET window k: cortical spikes of window k-1 (delay >= 5 ms)
        ctx_in = last_cnt[:, ctx_ids_t].clamp(max=1.0).unsqueeze(1)
        bn.run(inputs={"ctx": ctx_in, "sens": torch.zeros(WIN_MS, 1, 48)}, time=WIN_MS, reward=0.0)
        mf_in = last_cnt[:, mf_ids_t].clamp(max=1.0).repeat_interleave(2, dim=0).unsqueeze(1)  # 1-ms bins -> 0.5-ms steps
        window_sj_out = globals()["window_cb"](mf_in, torch.zeros(10, 1, 12))
        t1 = time.perf_counter()
        s, ring, stp, cnts = out
        cnts.block_until_ready()
        t_wait += time.perf_counter() - t1
        last_cnt = torch.from_dlpack(cnts)
        tot_spk.append((float(cnts.sum()), float(bn.layers['STR'].s.sum()), float(bn.layers['BLA'].s.sum()), float(window_sj_out.sum())))
    return (s, ring, stp), t_wait
globals()["window_cb"] = window_sj

import copy, io
state = (s, ring, stp)
state, _ = epoch(state, 0)
def snapshot():
    buf = io.BytesIO()
    torch.save({"bn": bn.state_dict(), "bn_layers": {k: {n: getattr(l, n).clone() for n in ("v", "s", "x", "u") if hasattr(l, n) and isinstance(getattr(l, n), torch.Tensor)} for k, l in bn.layers.items()},
                "bn_rules": [(c.update_rule.eligibility.clone(), c.update_rule.eligibility_trace.clone(), c.update_rule.p_plus.clone(), c.update_rule.p_minus.clone()) if hasattr(c.update_rule, "eligibility_trace") else None for c in bn.connections.values()],
                "sj": {k: (p.v.clone() if isinstance(p.v, torch.Tensor) else p.v) for k, p in pops.items()}, "W_pf_pc": W_pf_pc.clone(), "elig": elig.clone(),
                "prev": {k: v.clone() for k, v in prev.items()}, "jax": [np.asarray(x) for x in jax.tree_util.tree_leaves(state)]}, buf)
    return buf.getvalue()
def restore(blob):
    global state
    d = torch.load(io.BytesIO(blob), weights_only=False)
    bn.load_state_dict(d["bn"])
    for k, l in bn.layers.items():
        for n, t in d["bn_layers"][k].items(): getattr(l, n).copy_(t)
    for c, r in zip(bn.connections.values(), d["bn_rules"]):
        if r is not None:
            for name, t in zip(("eligibility", "eligibility_trace", "p_plus", "p_minus"), r): getattr(c.update_rule, name).copy_(t)
    for k, p in pops.items():
        if isinstance(p.v, torch.Tensor): p.v = d["sj"][k].clone()
        else: p.v = d["sj"][k]
    W_pf_pc.copy_(d["W_pf_pc"]); elig.copy_(d["elig"])
    for k, v in d["prev"].items(): prev[k] = v
    leaves = [jnp.asarray(x) for x in d["jax"]]
    state = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(state), leaves)
blob = snapshot()
st = state
a1, _ = epoch(st, 250); r1 = list(tot_spk); a2, _ = epoch(a1, 500); r2 = list(tot_spk)
restore(blob)
b1, _ = epoch(state, 250); q1 = list(tot_spk); b2, _ = epoch(b1, 500); q2 = list(tot_spk)
print("snapshot size KB:", len(blob)//1024)
print("restore+replay identical per-window JAX spike totals:", r1 == q1 and r2 == q2, "| JAX spikes", sum(x[0] for x in r1+r2))
print("windows:", len(r1)+len(r2), "| BindsNET STR spikes", sum(x[1] for x in r1+r2), "| SJ DCN spikes", sum(x[3] for x in r1+r2))
