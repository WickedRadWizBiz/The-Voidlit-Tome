import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/The-Voidlit-Tome/brain")
import torch
torch.set_num_threads(1)
import jaxpath  # noqa
import jax, jax.numpy as jnp, numpy as np
import column as COL
import bg as BG
from spikingjelly.activation_based import functional
from voidlit.cortex.director import load_director

TURN_MS, TICK, EPOCH_MS = 200.0, 0.5, 2.0
TICKS, EPOCHS, TPE = int(TURN_MS / TICK), int(TURN_MS / EPOCH_MS), int(EPOCH_MS / TICK)

# --- JAX column partition
st = COL.init(jax.random.PRNGKey(0))
ext = jnp.zeros((TPE, COL.COLS, COL.N))
st2, sp = COL.epoch(st, ext); jax.block_until_ready(sp)  # compile
t0 = time.perf_counter()
for _ in range(EPOCHS):
    st, sp = COL.epoch(st, ext)
jax.block_until_ready(sp)
t_col = time.perf_counter() - t0
print(f"JAX columns (3 x {COL.N} cells, {COL.N_PC} 9-comp PCs each, dt=0.05 ms): {t_col*1e3:.1f} ms / 200 ms turn "
      f"({EPOCHS} epoch calls)")

# --- DLPack interop
a = jnp.ones((TPE, COL.COLS * COL.N))
t0 = time.perf_counter()
for _ in range(1000):
    t = torch.from_dlpack(a)
    b = jnp.from_dlpack(t)
dl = (time.perf_counter() - t0) / 1000
same = np.shares_memory(np.asarray(a), t.numpy())
print(f"DLPack JAX->torch->JAX round trip: {dl*1e6:.1f} us (zero-copy JAX->torch: {same})")

# --- BindsNET BG partition
net = BG.build()
inp = torch.zeros(int(EPOCH_MS), 1, BG.CTX)
net.run(inputs={"ctx": inp}, time=int(EPOCH_MS), reward=0.0)
t0 = time.perf_counter()
for _ in range(EPOCHS):
    net.run(inputs={"ctx": (torch.rand(int(EPOCH_MS), 1, BG.CTX) < 0.02).float()}, time=int(EPOCH_MS), reward=0.0)
t_bg = time.perf_counter() - t0
print(f"BindsNET BG ({BG.A} channels, {BG.A*BG.PER*2} MSNs, dt=1 ms, MSTDPET): {t_bg*1e3:.1f} ms / turn")

# --- SpikingJelly Director pathway in single-step mode at 1 ms
m = load_director()
functional.set_step_mode(m.net, "s")
functional.reset_net(m.net)
x = torch.rand(1, 18)
t0 = time.perf_counter()
with torch.no_grad():
    acc = 0
    for k in range(int(TURN_MS)):
        acc = acc + m.net((torch.rand(1, 18) < x).float())
t_sj = time.perf_counter() - t0
print(f"SpikingJelly Director pathway, single-step, 200 steps: {t_sj*1e3:.1f} ms / turn")
print(f"TOTAL (sequential, single thread): {(t_col + t_bg + t_sj)*1e3:.1f} ms")
