"""One network, one clock: the JAX, BindsNET and SpikingJelly partitions advance together in W-ms windows over a
shared spike bus (conservative synchronisation, as in NEST/MUSIC: every cross-partition projection has an axonal
delay >= W, so a window only reads spikes written in earlier windows). The partitions can therefore run in any
order, or concurrently in threads, and the result is bit-identical.

    python one_clock.py            timing (sequential vs threaded), determinism, save/load continuation
"""
from __future__ import annotations

import hashlib
import io
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import jaxpath  # noqa: F401
import jax
import jax.numpy as jnp

import tome_jax as TJ
import tome_bn2 as TB
import tome_sj as TS

W = 4
TURN_MS = 248          # 62 windows
D_BUS = 64             # bus history (ms)

# ---------------------------------------------------------------------------------------------- bus wiring
JAX_TO_BN = [f"{c}.L5_IT" for c in TJ.COLS] + ["P.L5_ET", "P.TC_mtx", "CA1"]      # 96+24+14+64 = 198
JAX_TO_SJ = ["P.L5_ET", "I.L5_ET", "P.L5_IT", "I.L5_IT"]                        # 96 mossy fibres
BUS_DELAY = {"jax->bn": 6, "sj->bn": 5, "bn->jax": 5, "jax->sj": 5, "sj->jax": 6, "bn->sj": 8}
assert min(BUS_DELAY.values()) >= W, "conservative sync needs every cross-partition delay >= W"


def idx(names):
    return np.concatenate([TJ.sl(n) for n in names])


J2B, J2S = idx(JAX_TO_BN), idx(JAX_TO_SJ)
TC_MTX_P, TC_MTX_I = TJ.sl("P.TC_mtx"), TJ.sl("I.TC_mtx")
LC, PVN, VTA_G = TJ.sl("LC"), TJ.sl("PVN"), TJ.sl("VTA_GABA")


class Bus:
    """Ring buffers of spikes per source partition, indexed by absolute ms."""

    def __init__(self):
        self.buf = {"jax": np.zeros((D_BUS, TJ.N), np.float32), "bn_pal": np.zeros((D_BUS, TB.N_PAL), np.float32),
                    "bn_amy": np.zeros((D_BUS, TB.N_AMY), np.float32), "sj_dcn": np.zeros((D_BUS, TS.N_DCN), np.float32),
                    "sj_habit": np.zeros((D_BUS, 32), np.float32), "jax_mod": np.zeros((D_BUS, 4), np.float32)}

    def write(self, src, t0, x):
        for i in range(x.shape[0]):
            self.buf[src][(t0 + i) % D_BUS] = x[i]

    def read(self, src, t0, delay, cols=None):
        rows = [(t0 + i - delay) % D_BUS for i in range(W)]
        out = self.buf[src][rows]
        return out if cols is None else out[:, cols]


class TomeBrain:
    def __init__(self, seed=0):
        torch.set_num_threads(1)
        self.t = 0
        self.jst, self.jstatic = TJ.build(seed)
        self.jwin = make_jax_window()
        self.bn, self.bn_mons, self.bn_tonic = TB.build(seed)
        self.cb, self.habit = TS.Cerebellum(seed), TS.Habit()
        self.gen = torch.Generator().manual_seed(seed + 17)
        self.bus = Bus()
        g = np.random.default_rng(seed + 5)
        # fixed bus->JAX current maps (pA per spike), BN/SJ -> target cells
        self.w_gpi = np.zeros((TB.N_PAL, TJ.N), np.float32)
        gpi0 = TB.PAL["GPi"][0]
        for c in range(TB.CH):
            self.w_gpi[gpi0 + 2 * c:gpi0 + 2 * c + 2, TC_MTX_P[c]] = -60.0      # GPi inhibits its VA channel
        self.w_amy = np.zeros((TB.N_AMY, TJ.N), np.float32)
        cea = slice(*TB.AMY["CeA"])
        self.w_amy[cea][:, np.concatenate([LC, PVN, VTA_G])] = 30.0 * (g.random((16, 36)) < 0.3)
        self.w_dcn = np.zeros((TS.N_DCN, TJ.N), np.float32)
        self.w_dcn[:, np.concatenate([TC_MTX_P, TC_MTX_I])] = 25.0 * (g.random((TS.N_DCN, 28)) < 0.3)
        self.features = torch.full((1, 18), 0.3)
        self.drive = jnp.zeros(TJ.N, jnp.float32)
        self.pool = ThreadPoolExecutor(max_workers=3)

    # ---------------------------------------------------------------------------------- partition steps
    def _jax_launch(self, t0):
        b = self.bus
        i_bus = (b.read("bn_pal", t0, BUS_DELAY["bn->jax"]) @ self.w_gpi
                 + b.read("bn_amy", t0, BUS_DELAY["bn->jax"]) @ self.w_amy
                 + b.read("sj_dcn", t0, BUS_DELAY["sj->jax"]) @ self.w_dcn)
        self.jst, spk = self.jwin(self.jstatic, self.jst, self.drive, jnp.asarray(i_bus))   # async dispatch
        return spk

    def _jax(self, t0):
        spk = self._jax_launch(t0)
        return np.asarray(spk), np.asarray(self.jst["mod"])

    def _bn(self, t0):
        b = self.bus
        ctx = np.concatenate([b.read("jax", t0, BUS_DELAY["jax->bn"], J2B),
                              b.read("sj_habit", t0, BUS_DELAY["sj->bn"])], axis=1)
        mod = b.read("jax_mod", t0, BUS_DELAY["jax->bn"]).mean(0)
        TB.window(self.bn, self.bn_tonic, torch.from_numpy(ctx), da_phasic=float(mod[0] - 0.5),
                  da_tonic=float(mod[0]))
        out = {k: m.get("s").float()[:, 0].numpy().copy() for k, m in self.bn_mons.items()}
        for m in self.bn_mons.values():
            m.reset_state_variables()
        return out

    def _sj(self, t0):
        b = self.bus
        mf = np.concatenate([b.read("jax", t0, BUS_DELAY["jax->sj"], J2S),
                             1.0 - np.minimum(1.0, b.read("bn_pal", t0, BUS_DELAY["bn->sj"],
                                                          slice(*TB.PAL["GPi"]))[:, :24])], axis=1)
        pc, dcn = self.cb(torch.from_numpy(mf).unsqueeze(1), torch.zeros(W, 1, TS.N_PC))
        h = self.habit(self.features, self.gen)
        return dcn[:, 0].numpy().copy(), h[:, 0].numpy().copy()

    def window(self, threaded):
        t0 = self.t
        if threaded == "async":           # XLA computes the JAX window while Python steps BindsNET and SJ
            spk_f = self._jax_launch(t0)
            bn = self._bn(t0)
            dcn, h = self._sj(t0)
            spk, mod = np.asarray(spk_f), np.asarray(self.jst["mod"])
        elif threaded:
            fj, fb, fs = (self.pool.submit(self._jax, t0), self.pool.submit(self._bn, t0),
                          self.pool.submit(self._sj, t0))
            (spk, mod), bn, (dcn, h) = fj.result(), fb.result(), fs.result()
        else:
            spk, mod = self._jax(t0)
            bn = self._bn(t0)
            dcn, h = self._sj(t0)
        b = self.bus                                         # barrier passed: publish this window's spikes
        b.write("jax", t0, spk)
        b.write("jax_mod", t0, np.repeat(mod[None], W, 0))
        b.write("bn_pal", t0, bn["pal"])
        b.write("bn_amy", t0, bn["amy"])
        b.write("sj_dcn", t0, dcn)
        b.write("sj_habit", t0, h)
        self.t += W
        return bn

    def turn(self, threaded=True):
        gpi = np.zeros(TB.N_PAL)
        for _ in range(TURN_MS // W):
            bn = self.window(threaded)
            gpi += bn["pal"].sum(0)
        return gpi

    # ---------------------------------------------------------------------------------- persistence
    def state_bytes(self):
        jst = jax.device_get(self.jst)
        blob = {"t": self.t, "jax": jst, "bus": self.bus.buf, "bn": self.bn, "cb": self.cb.state_dict(),
                "cb_lif": [(m.v.clone() if isinstance(m.v, torch.Tensor) else m.v) for m in (self.cb.gc, self.cb.go, self.cb.pc, self.cb.dcn)],
                "habit_lif": [m.v.clone() if isinstance(m.v, torch.Tensor) else m.v
                              for m in self.habit.modules() if hasattr(m, "v") and m is not self.habit],
                "gen": self.gen.get_state(), "features": self.features}
        f = io.BytesIO()
        torch.save(blob, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        return f.getvalue()

    def load_bytes(self, data):
        blob = torch.load(io.BytesIO(data), weights_only=False)
        self.t = blob["t"]
        self.jst = jax.tree_util.tree_map(jnp.asarray, blob["jax"])
        self.bus.buf = blob["bus"]
        self.bn = blob["bn"]
        self.bn_mons = {k: self.bn.monitors[k] for k in ("str", "pal", "amy")}
        self.cb.load_state_dict(blob["cb"])
        for m, v in zip((self.cb.gc, self.cb.go, self.cb.pc, self.cb.dcn), blob["cb_lif"]):
            m.v = v
        mods = [m for m in self.habit.modules() if hasattr(m, "v") and m is not self.habit]
        for m, v in zip(mods, blob["habit_lif"]):
            m.v = v
        self.gen.set_state(blob["gen"])
        self.features = blob["features"]

    def digest(self):
        h = hashlib.sha256()
        for leaf in jax.tree_util.tree_leaves(jax.device_get(self.jst)):
            h.update(np.asarray(leaf).tobytes())
        for k in sorted(self.bus.buf):
            h.update(self.bus.buf[k].tobytes())
        for p in self.bn.state_dict().values():
            h.update(p.detach().cpu().numpy().tobytes())
        for p in self.cb.state_dict().values():
            h.update(p.cpu().numpy().tobytes())
        return h.hexdigest()[:16]


def make_jax_window():
    @jax.jit
    def window(static, st, drive, i_bus):
        def body(st, i_ms):
            st, spk = TJ.ms_step(static, st, drive + i_ms)
            return st, spk
        return jax.lax.scan(body, st, i_bus)
    return window


if __name__ == "__main__":
    brain = TomeBrain(0)
    brain.turn(threaded=False)               # compile + warm up
    for mode in (False, "async", False, "async", "async"):
        t0 = time.perf_counter()
        brain.turn(threaded=mode)
        print(f"one turn = {TURN_MS} ms of brain time, {str(mode):10s}: "
              f"{1e3*(time.perf_counter()-t0):6.1f} ms wall", flush=True)
    # determinism: two fresh brains, same seed, threaded vs sequential, 3 turns each
    a, b = TomeBrain(1), TomeBrain(1)
    for _ in range(3):
        a.turn(threaded="async"); b.turn(threaded=False)
    print("deterministic (async-overlapped == sequential, same seed):", a.digest() == b.digest(), a.digest(), b.digest())
    # save/load continuation
    saved = a.state_bytes()
    a.turn(threaded=True); a.turn(threaded=True)
    c = TomeBrain(1)
    c.load_bytes(saved)
    c.turn(threaded=True); c.turn(threaded=True)
    print(f"save/load continuation identical: {a.digest() == c.digest()}  (save blob {len(saved)/1e6:.1f} MB)")
