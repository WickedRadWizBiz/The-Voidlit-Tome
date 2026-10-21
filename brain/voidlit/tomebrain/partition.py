"""Partitions of the one network, and the projections that join them through the spike bus.

A partition advances W ms of the master clock at a time. It reads only bus rows written in earlier windows and
publishes its own rows when the window ends. The JAX partition runs asynchronously: it is dispatched first, the torch
partitions run on the Python thread while XLA computes, and the join is the only synchronisation point.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .bus import SpikeBus
from .jaxpart import network as NW
from .jaxpart import params as PR

ANATOMY_SEED = 0   # the Tome's wiring is the same in every game; the run seed only keys its noise


@dataclass(frozen=True)
class BusEdge:
    """A projection from a bus source into JAX cells: fixed in-degree K, conductance synapses."""

    src: str
    src_width: int
    dst: str                 # JAX population or assembly name
    K: int
    recs: tuple[int, ...]
    comp: int
    w: float
    delay: int


class Partition:
    """Interface: ``outputs`` maps each bus source this partition writes to its width.

    ``state_dict`` must hold everything that affects the future, including the seeds of its noise streams, so a
    brain loaded into an instance built with another run seed continues bit-exactly (gate G7)."""

    name = "partition"
    outputs: dict[str, int] = {}

    def step(self, bus: SpikeBus, t0: int, W: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, d: dict) -> None:
        raise NotImplementedError

    def digest_arrays(self) -> list[np.ndarray]:
        return []


class JaxPartition(Partition):
    """Cortex, thalamus, hippocampus, nuclei, glia and energy (``jaxpart.network``)."""

    name = "jax"

    def __init__(self, run_seed: int, edges: list[BusEdge] = (), outputs: dict[str, str | list[str]] | None = None,
                 flags: dict | None = None, net: NW.Net | None = None) -> None:
        import jax

        self.net = net or NW.build(seed=ANATOMY_SEED)
        self.window = NW.make_window(self.net, flags=flags)
        self.state = NW.init_state(self.net, jax.random.PRNGKey(run_seed))
        self.out_idx = {name: self._cells(spec) for name, spec in (outputs or {}).items()}
        self.outputs = {name: len(idx) for name, idx in self.out_idx.items()}
        self.edges = [self._compile(e) for e in edges]
        self.ext_fn = None          # optional (t0, W) -> ext currents [W, N]
        self.glucose = 1.0

    def _cells(self, spec) -> np.ndarray:
        names = [spec] if isinstance(spec, str) else list(spec)
        return np.concatenate([np.arange(self.net.idx(n).start, self.net.idx(n).stop) for n in names])

    def _compile(self, e: BusEdge):
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{e.src}>{e.dst}".encode()).digest()[:4], "little"))
        rows, slots, ws = [], [], []
        for post in self._cells(e.dst):
            for pre in rng.choice(e.src_width, size=min(e.K, e.src_width), replace=False):
                jitter = rng.uniform(0.7, 1.3)
                for r in e.recs:
                    rows.append(pre)
                    slots.append(self.net.slot(int(post), e.comp, r))
                    ws.append(e.w * jitter * (0.5 if r == PR.NMDA else 1.0))
        return e, np.array(rows), np.array(slots), np.array(ws, np.float32)

    def gather(self, bus: SpikeBus, t0: int, W: int) -> np.ndarray:
        g = np.zeros((W, self.net.n_slot + 1), np.float32)
        for e, rows, slots, ws in self.edges:
            arriving = bus.read(e.src, t0, W, e.delay)
            contrib = arriving[:, rows] * ws
            for t in range(W):
                g[t] += np.bincount(slots, weights=contrib[t], minlength=self.net.n_slot + 1).astype(np.float32)
        return g

    def dispatch(self, bus: SpikeBus, t0: int, W: int):
        import jax.numpy as jnp

        ext = self.ext_fn(t0, W) if self.ext_fn else np.zeros((W, self.net.n), np.float32)
        gbus = self.gather(bus, t0, W)
        self.state, (s, _) = self.window(self.state, jnp.asarray(ext), self.glucose, None, jnp.asarray(gbus))
        return s                      # still computing; np.asarray() joins

    def finish(self, pending) -> dict[str, np.ndarray]:
        s = np.asarray(pending)
        self.last = s
        return {name: s[:, idx] for name, idx in self.out_idx.items()}

    def step(self, bus: SpikeBus, t0: int, W: int) -> dict[str, np.ndarray]:
        return self.finish(self.dispatch(bus, t0, W))

    def state_dict(self) -> dict:
        return {k: np.asarray(v) for k, v in self.state.items()}

    def load_state_dict(self, d: dict) -> None:
        import jax.numpy as jnp

        missing = set(self.state) ^ set(d)
        if missing:
            raise ValueError(f"JAX state keys differ: {sorted(missing)}")
        for k, v in d.items():
            if np.shape(v) != np.shape(self.state[k]):
                raise ValueError(f"JAX state {k}: shape {np.shape(v)} != {np.shape(self.state[k])}")
        self.state = {k: jnp.asarray(v, dtype=self.state[k].dtype) for k, v in d.items()}

    def digest_arrays(self) -> list[np.ndarray]:
        return [np.asarray(self.state[k]) for k in sorted(self.state)]
