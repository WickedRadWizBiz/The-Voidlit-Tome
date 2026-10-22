"""One clock: the master clock and the window scheduler that co-simulates every partition on it.

Window k covers ticks [kW, kW + W). Each partition reads only bus rows from earlier windows, so the order in which
partitions run inside a window, whether JAX runs asynchronously, and the window size itself never change the result
(gates G6b-d). The JAX partition is dispatched first and joined last; the torch partitions run in between.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np

from .bus import SpikeBus
from .partition import JaxPartition, Partition

W_DEFAULT = 4


@dataclass
class MasterClock:
    t_ms: int = 0
    turn: int = 0
    phase: str = "idle"           # idle | A | B | quiet

    def state_dict(self) -> dict:
        return {"t_ms": self.t_ms, "turn": self.turn, "phase": self.phase}

    def load_state_dict(self, d: dict) -> None:
        self.t_ms, self.turn, self.phase = int(d["t_ms"]), int(d["turn"]), str(d["phase"])


class Cosim:
    def __init__(self, jax_part: JaxPartition, others: list[Partition], W: int = W_DEFAULT) -> None:
        self.jax = jax_part
        self.others = list(others)
        self.W = W
        sizes = dict(jax_part.outputs)
        for p in self.others:
            sizes.update(p.outputs)
        self.bus = SpikeBus(sizes, W)
        self.clock = MasterClock()
        self.order = [p.name for p in self.others]
        # wall seconds spent in each part since the caller last cleared it: developer telemetry, never saved or digested
        self.perf: dict[str, float] = {}

    def _lap(self, part: str, since: float) -> float:
        now = time.perf_counter()
        self.perf[part] = self.perf.get(part, 0.0) + now - since
        return now

    @property
    def partitions(self) -> list[Partition]:
        return [self.jax, *self.others]

    def run(self, ms: int, record: tuple[str, ...] = (), async_jax: bool = True) -> dict[str, np.ndarray]:
        """Advance ``ms`` milliseconds (a multiple of W). Returns the recorded bus sources as [ms, n] rasters.
        ``async_jax=False`` joins JAX before the torch partitions run (same result; gate G6c)."""
        if ms % self.W:
            raise ValueError(f"run length {ms} is not a whole number of {self.W}-ms windows")
        rec: dict[str, list[np.ndarray]] = {k: [] for k in record}
        by_name = {p.name: p for p in self.others}
        for _ in range(ms // self.W):
            t0 = self.clock.t_ms
            lap = time.perf_counter()
            pending = self.jax.dispatch(self.bus, t0, self.W)          # XLA computes while torch runs
            lap = self._lap("jax_dispatch", lap)
            out: dict[str, np.ndarray] = {}
            if not async_jax:
                out = self.jax.finish(pending)
                lap = self._lap("jax_join", lap)
            for name in self.order:
                out.update(by_name[name].step(self.bus, t0, self.W))
                lap = self._lap(name, lap)
            if async_jax:
                out.update(self.jax.finish(pending))
                lap = self._lap("jax_join", lap)
            for src, rows in out.items():
                self.bus.write(src, t0, np.asarray(rows, np.float32))
                if src in rec:
                    rec[src].append(np.asarray(rows))
            self._lap("bus", lap)
            self.clock.t_ms += self.W
        return {k: np.concatenate(v) if v else np.zeros((0, 0)) for k, v in rec.items()}

    def digest(self) -> str:
        """SHA-256 over every partition's state, the bus and the clock."""
        h = hashlib.sha256()
        for p in self.partitions:
            h.update(p.name.encode())
            for a in p.digest_arrays():
                h.update(np.ascontiguousarray(a).tobytes())
        for k in sorted(self.bus.buf):
            h.update(self.bus.buf[k].tobytes())
        h.update(str(sorted(self.bus.written.items())).encode())
        h.update(str(self.clock.state_dict()).encode())
        return h.hexdigest()
