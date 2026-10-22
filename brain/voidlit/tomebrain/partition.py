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
from .rng import Streams

ANATOMY_SEED = 0   # the Tome's wiring is the same in every game; the run seed only keys its noise
MAX_W = 4          # the JAX loop is compiled once for windows of up to MAX_W ms


def shapes_of(d: dict) -> dict:
    """The shape of every number or array of numbers in a nested state dict (None for anything else): the template
    ``check_shapes`` holds a save against."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = shapes_of(v)
        else:
            a = np.asarray(v)
            out[k] = a.shape if a.dtype.kind in "biuf" else None
    return out


def check_shapes(what: str, d, want: dict) -> None:
    """Raise ValueError unless ``d`` has exactly the keys of ``want``, each a number or an array of numbers of the
    shape there (a nested dict for a nested one; None takes anything, for the caller to check)."""
    if not isinstance(d, dict):
        raise ValueError(f"{what}: not a dict")
    if set(d) != set(want):
        raise ValueError(f"{what}: keys differ: {sorted(map(str, set(d) ^ set(want)))}")
    for k, shape in want.items():
        if isinstance(shape, dict):
            check_shapes(f"{what}.{k}", d[k], shape)
        elif shape is not None:
            a = np.asarray(d[k])
            if a.dtype.kind not in "biuf" or a.shape != tuple(shape):
                raise ValueError(f"{what}.{k}: {a.dtype} of shape {a.shape}, not numbers of shape {tuple(shape)}")


def check_streams(what: str, d, partition: str) -> None:
    """Raise ValueError unless ``d`` is the state of the ``partition`` partition's noise streams (an integer seed:
    ``Streams`` would round any other number into another seed's noise)."""
    try:
        ok = Streams.from_state(d).partition == partition and np.asarray(d["seed"]).dtype.kind in "iu"
    except (KeyError, TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError(f"{what}: not the {partition!r} partition's noise streams")


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

    def check_state_dict(self, d: dict) -> None:
        """Raise ValueError unless ``load_state_dict`` can take ``d`` whole: every key it reads and nothing else, each
        at the shape this instance holds. It changes nothing, so a save can be refused before any partition loads.
        By default: the keys and shapes of this instance's own state."""
        check_shapes(self.name, d, shapes_of(self.state_dict()))

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
        self.reset_counts()

    def reset_counts(self) -> None:
        """Start a new turn's tally: spikes per cell, ms simulated, and probe sums (for the readouts)."""
        self.counts = np.zeros(self.net.n, np.float64)
        self.count_ms = 0
        self.probe_sum = np.zeros(NW.PROBE_N, np.float64)

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
        self.counts += s.sum(0)
        self.count_ms += s.shape[0]
        for row in probe:                 # one ms at a time in float64: summing the window's float32 rows first
            self.probe_sum += row         # would round differently for each W and break W=4 == W=1
        out = {name: s[:, idx] for name, idx in self.out_idx.items()}
        if "J.nm" in self.graded:
            out["J.nm"] = probe[:, 2:6]
        return out

    def step(self, bus: SpikeBus, t0: int, W: int) -> dict[str, np.ndarray]:
        return self.finish(self.dispatch(bus, t0, W))

    def state_dict(self) -> dict:
        return {k: np.asarray(v) for k, v in self.state.items()}

    def check_state_dict(self, d: dict) -> None:
        check_shapes("JAX state", d, {k: np.shape(v) for k, v in self.state.items()})

    def load_state_dict(self, d: dict) -> None:
        import jax.numpy as jnp

        self.check_state_dict(d)
        self.state = {k: jnp.asarray(v, dtype=self.state[k].dtype) for k, v in d.items()}

    def digest_arrays(self) -> list[np.ndarray]:
        return [np.asarray(self.state[k]) for k in sorted(self.state)]
