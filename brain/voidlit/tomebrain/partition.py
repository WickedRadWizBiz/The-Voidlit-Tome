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
MAX_W = 4          # the JAX loop is compiled once for windows of up to MAX_W ms


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
    src_range: tuple[int, int] | None = None     # only these rows of the source (topographic projections)


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

    GRADED = {"J.nm": 4}      # graded bus rows: the four neuromodulator concentrations

    def __init__(self, run_seed: int, edges: list[BusEdge] = (), outputs: dict[str, str | list[str]] | None = None,
                 flags: dict | None = None, net: NW.Net | None = None, graded: tuple[str, ...] = ()) -> None:
        import jax

        self.net = net or NW.build(seed=ANATOMY_SEED)
        self.window = NW.make_window(self.net, flags=flags, max_w=MAX_W)
        self.state = NW.init_state(self.net, jax.random.PRNGKey(run_seed))
        self.out_idx = {name: self._cells(spec) for name, spec in (outputs or {}).items()}
        self.outputs = {name: len(idx) for name, idx in self.out_idx.items()}
        self.graded = tuple(graded)
        self.outputs.update({g: self.GRADED[g] for g in self.graded})
        groups: dict[tuple[str, int], object] = {}
        for e in edges:                      # one sparse matrix per (source, delay): all its projections summed
            e, mat = self._compile(e)
            key = (e.src, e.delay)
            groups[key] = mat if key not in groups else groups[key] + mat
        self.edges = [(src, delay, mat.T.tocsr()) for (src, delay), mat in groups.items()]
        self.ext_fn = None          # optional (t0, W) -> ext currents [W, N]
        self.glucose = 1.0

    def _cells(self, spec) -> np.ndarray:
        names = [spec] if isinstance(spec, str) else list(spec)
        return np.concatenate([np.arange(self.net.idx(n).start, self.net.idx(n).stop) for n in names])

    def _compile(self, e: BusEdge):
        tag = f"{e.src}{e.src_range}>{e.dst}".encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(tag).digest()[:4], "little"))
        rows, slots, ws = [], [], []
        lo, hi = e.src_range or (0, e.src_width)
        for post in self._cells(e.dst):
            for pre in lo + rng.choice(hi - lo, size=min(e.K, hi - lo), replace=False):
                jitter = rng.uniform(0.7, 1.3)
                for r in e.recs:
                    rows.append(pre)
                    slots.append(self.net.slot(int(post), e.comp, r))
                    ws.append(e.w * jitter * (0.5 if r == PR.NMDA else 1.0))
        from scipy import sparse
        # one sparse matrix per projection: source line -> receptor slot (duplicate entries sum)
        mat = sparse.csr_matrix((np.array(ws, np.float32), (np.array(rows), np.array(slots))),
                                shape=(e.src_width, self.net.n_slot + 1))
        return e, mat

    def gather(self, bus: SpikeBus, t0: int, W: int) -> np.ndarray:
        """Bus conductance per receptor slot for each tick of the window (each tick's column is summed in the same
        order whatever W is, so windowed and lockstep runs agree bitwise)."""
        g = np.zeros((W, self.net.n_slot + 1), np.float32)
        for src, delay, matT in self.edges:
            arriving = bus.read(src, t0, W, delay)
            for t in np.flatnonzero(arriving.any(1)):
                g[t] += matT @ arriving[t]        # always the single-vector product: scipy's multi-vector kernel
        return g                                  # sums in another order, which would break W=4 == W=1

    def dispatch(self, bus: SpikeBus, t0: int, W: int):
        import jax.numpy as jnp

        if W > MAX_W:
            raise ValueError(f"window {W} ms exceeds the compiled maximum of {MAX_W}")
        ext = np.zeros((MAX_W, self.net.n), np.float32)
        if self.ext_fn:
            ext[:W] = self.ext_fn(t0, W)
        gbus = np.zeros((MAX_W, self.net.n_slot + 1), np.float32)
        gbus[:W] = self.gather(bus, t0, W)
        self.state, s, probe = self.window(self.state, jnp.asarray(ext), self.glucose, jnp.asarray(gbus), W)
        return s, probe, W            # still computing; np.asarray() joins

    def finish(self, pending) -> dict[str, np.ndarray]:
        s, probe = np.asarray(pending[0])[:pending[2]], np.asarray(pending[1])[:pending[2]]
        self.last = s
        out = {name: s[:, idx] for name, idx in self.out_idx.items()}
        if "J.nm" in self.graded:
            out["J.nm"] = probe[:, 2:6]
        return out

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
