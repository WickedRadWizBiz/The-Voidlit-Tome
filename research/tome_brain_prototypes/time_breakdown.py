import time, warnings
warnings.filterwarnings("ignore")
import torch
import jaxpath  # noqa
import jax
import integrated as I
feats = torch.rand(18)
tome = I.Tome(); tome.turn(feats, seed=0)
def timed(fn, n=4):
    out = []
    for s in range(n):
        t0 = time.perf_counter(); fn(s); out.append(time.perf_counter() - t0)
    return round(sorted(out)[n // 2] * 1e3, 1)
print("full turn ms:", timed(lambda s: tome.turn(feats, seed=s + 1)))
ext = jax.numpy.zeros((2, I.COL.N))
def jax_only(s):
    st = tome.jst
    for _ in range(100):
        st, sp = tome.jrun(st, tome.W, tome.idx, ext)
    jax.block_until_ready(sp)
print("JAX partition alone (100 x 2 ms epochs):", timed(jax_only))
ctx = torch.zeros(2, 1, 96); it = torch.zeros(2, 1, 64)
def bn_only(s):
    for _ in range(100):
        tome.limbic.run(inputs={"ctx": ctx, "intero": it}, time=2, reward=0.0)
print("BindsNET limbic alone:", timed(bn_only))
mf = torch.zeros(10, 1, 96)
def cb_only(s):
    with torch.no_grad():
        for _ in range(20): tome.cb(mf)
print("SJ cerebellum alone (20 x 10 ms):", timed(cb_only))
