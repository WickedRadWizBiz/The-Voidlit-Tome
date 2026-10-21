"""Counter-based noise for the torch partitions, keyed by (seed, partition, stream, absolute tick).

Every draw is a pure function of its key, so results do not depend on window size, call batching, the order in
which partitions run, or where a save happened. That is what makes windowed simulation bitwise equal to lockstep
(gate G6b); stateful torch generators would break it. The JAX partition does the same with
``jax.random.fold_in(key, tick)``.
"""

from __future__ import annotations

import hashlib

import numpy as np

PARTITIONS = {"jax": 0, "sj": 1, "bn": 2, "rules": 3}


def _key(seed: int, partition: str, stream: str) -> int:
    h = hashlib.sha256(f"{seed}:{partition}:{stream}".encode()).digest()
    return int.from_bytes(h[:16], "little")


class Streams:
    """Named Philox streams for one partition. ``gen(stream, tick)`` is a fresh generator for that tick."""

    def __init__(self, seed: int, partition: str) -> None:
        if partition not in PARTITIONS:
            raise ValueError(f"unknown partition {partition!r}")
        self.seed, self.partition = seed, partition
        self._keys: dict[str, int] = {}

    def gen(self, stream: str, tick: int) -> np.random.Generator:
        key = self._keys.get(stream)
        if key is None:
            key = self._keys[stream] = _key(self.seed, self.partition, stream)
        return np.random.Generator(np.random.Philox(key=key, counter=[int(tick), 0, 0, 0]))

    def normal(self, stream: str, tick: int, shape) -> np.ndarray:
        return self.gen(stream, tick).standard_normal(shape, dtype=np.float32)

    def uniform(self, stream: str, tick: int, shape) -> np.ndarray:
        return self.gen(stream, tick).random(shape, dtype=np.float32)

    def gumbel(self, stream: str, tick: int, shape) -> np.ndarray:
        u = self.gen(stream, tick).random(shape, dtype=np.float64)
        return (-np.log(-np.log(np.clip(u, 1e-12, 1 - 1e-12)))).astype(np.float32)

    def state_dict(self) -> dict:
        """Everything needed to reproduce the streams (they are pure functions of seed, partition and tick)."""
        return {"seed": self.seed, "partition": self.partition}

    @classmethod
    def from_state(cls, d: dict) -> "Streams":
        return cls(int(d["seed"]), str(d["partition"]))
