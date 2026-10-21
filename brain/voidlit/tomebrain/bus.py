"""The spike bus: every cross-partition projection goes through here, one 1-ms row per tick.

Conservative synchronisation (as in NEST/MUSIC): every cross-partition delay is at least W ticks, so a partition
integrating the window [t0, t0 + W) only ever needs rows written in earlier windows. ``read`` checks this on every
call and raises if a row has not been written yet, so a causality bug can never pass silently.
"""

from __future__ import annotations

import numpy as np

HORIZON = 128   # ms of history kept per source; must exceed the longest bus delay plus one window


class CausalityError(RuntimeError):
    pass


class SpikeBus:
    def __init__(self, sizes: dict[str, int], W: int, horizon: int = HORIZON) -> None:
        if W < 1:
            raise ValueError("window must be at least 1 ms")
        self.W, self.h = W, horizon
        self.buf = {k: np.zeros((horizon, n), np.float32) for k, n in sizes.items()}
        self.written = {k: -1 for k in sizes}       # last tick written, per source

    def write(self, src: str, t0: int, rows: np.ndarray) -> None:
        """Publish rows for ticks [t0, t0 + len(rows)). Rows are spikes (0/1) or graded values."""
        if t0 != self.written[src] + 1:
            raise CausalityError(f"{src}: wrote tick {t0} after {self.written[src]}; ticks must be contiguous")
        idx = np.arange(t0, t0 + rows.shape[0]) % self.h
        self.buf[src][idx] = rows
        self.written[src] = t0 + rows.shape[0] - 1

    def read(self, src: str, t0: int, n: int, delay: int) -> np.ndarray:
        """Rows that arrive at ticks [t0, t0 + n) through an axon of ``delay`` ms (written at tick - delay)."""
        if delay < self.W:
            raise CausalityError(f"{src}: delay {delay} ms is shorter than the window ({self.W} ms)")
        if delay + n > self.h:
            raise CausalityError(f"{src}: delay {delay} ms exceeds the bus horizon")
        last_needed = t0 + n - 1 - delay
        if last_needed > self.written[src]:
            raise CausalityError(f"{src}: tick {last_needed} is not written yet (written up to {self.written[src]})")
        ticks = np.arange(t0, t0 + n) - delay
        out = self.buf[src][ticks % self.h].copy()
        out[ticks < 0] = 0.0          # before the brain was born: silence
        return out

    def state_dict(self) -> dict:
        return {"buf": {k: v.copy() for k, v in self.buf.items()}, "written": dict(self.written), "W": self.W}

    def load_state_dict(self, d: dict) -> None:
        self.buf = {k: np.asarray(v, np.float32).copy() for k, v in d["buf"].items()}
        self.written = {k: int(v) for k, v in d["written"].items()}
