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
    t_wait = 0.0
    for k in range(50):
        out = window(s, ring, stp, m0 + k*WIN_MS, delays, key, ext)          # async dispatch
        # BindsNET window k: cortical spikes of window k-1 (delay >= 5 ms)
        ctx_in = last_cnt[:, ctx_ids_t].clamp(max=1.0).unsqueeze(1)
        bn.run(inputs={"ctx": ctx_in, "sens": torch.zeros(WIN_MS, 1, 48)}, time=WIN_MS, reward=0.0)
        mf_in = last_cnt[:, mf_ids_t].clamp(max=1.0).repeat_interleave(2, dim=0).unsqueeze(1)  # 1-ms bins -> 0.5-ms steps
        window_sj = globals()["window_cb"](mf_in, torch.zeros(10, 1, 12))
        t1 = time.perf_counter()
        s, ring, stp, cnts = out
        cnts.block_until_ready()
        t_wait += time.perf_counter() - t1
        last_cnt = torch.from_dlpack(cnts)                                    # zero-copy JAX -> torch
    return (s, ring, stp), t_wait
globals()["window_cb"] = window_sj
state = (s, ring, stp)
state, _ = epoch(state, 0)
times = []
for rep in range(5):
    t0 = time.perf_counter(); state, tw = epoch(state, (rep+1)*250); times.append(time.perf_counter() - t0)
    print(f"integrated epoch {rep}: {1e3*times[-1]:.0f} ms wall (waiting on JAX at barriers: {1e3*tw:.0f} ms)")
print("median", round(1e3*statistics.median(times)), "ms")
