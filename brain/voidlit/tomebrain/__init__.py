"""The Tome's mind: one hybrid spiking network on one clock (docs/TOME_BRAIN_DESIGN.md).

Three partitions share a 1-ms master clock and a spike bus: JAX (cortex, thalamus, hippocampus, neuromodulatory and
hypothalamic nuclei, glia, energy), SpikingJelly (habit cortex, basal ganglia, cerebellum) and BindsNET (amygdala,
the corruption afferents). The NPC brains in ``voidlit.limbic`` stay separate minds.

Determinism: XLA runs single-threaded on CPU in float32, and every random draw is keyed by absolute tick, so the
same seed gives the same brain however the simulation is windowed. This must be set before JAX is imported.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
_flags = os.environ.get("XLA_FLAGS", "")
if "xla_cpu_multi_thread_eigen" not in _flags:
    os.environ["XLA_FLAGS"] = (_flags + " --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1").strip()
