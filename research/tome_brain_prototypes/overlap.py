import time, threading, warnings
warnings.filterwarnings("ignore")
import torch
torch.set_num_threads(1)
import jaxpath  # noqa
import jax, jax.numpy as jnp
import column2 as COL
from time_limbic2 import build
run = COL.make(0.1, 10)
st, W, idx = COL.init(jax.random.PRNGKey(0), 0.1)
ext = jnp.zeros((2, COL.N)) + 0.0005
for _ in range(3):
    st, sp = run(st, W, idx, ext); sp.block_until_ready()
net = build(640, n_ctx=96)
ctx = torch.zeros(2, 1, 96); it = (torch.rand(2, 1, 64) < 0.2).float()
def jax_loop():
    global st
    for _ in range(100):
        st, sp = run(st, W, idx, ext); sp.block_until_ready()
def bn_loop():
    for _ in range(100):
        net.run(inputs={"ctx": ctx, "intero": it}, time=2, reward=0.0)
for name, fn in (("jax", jax_loop), ("bn", bn_loop)):
    t0 = time.perf_counter(); fn(); print(name, "alone ms", round((time.perf_counter() - t0) * 1e3, 1))
t0 = time.perf_counter()
th = threading.Thread(target=jax_loop); th.start(); bn_loop(); th.join()
print("both in parallel threads ms", round((time.perf_counter() - t0) * 1e3, 1))
