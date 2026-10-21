"""Save and load the whole brain (design §13): every partition, the bus, the clock and the noise keys.

The file is a ``torch.save`` of tensors and plain containers only, read back with ``weights_only=True``. Loading
refuses a different format, version or connectome; a different software build only warns, because the state is
still valid but bit-exact continuation is guaranteed only on the same build. ``read`` makes every check without
changing the brain and ``apply`` then loads it, so a game can refuse a save before it changes anything else.
"""

from __future__ import annotations

import platform
import warnings
from pathlib import Path

import numpy as np

FORMAT, VERSION = "voidlit.tomebrain", 1


def build_info() -> dict:
    import jax
    import torch

    info = {"python": platform.python_version(), "numpy": np.__version__, "jax": jax.__version__,
            "torch": torch.__version__, "cpu": platform.machine()}
    try:
        import bindsnet
        import spikingjelly  # noqa: F401
        info["bindsnet"] = getattr(bindsnet, "__version__", "?")
    except ImportError:
        pass
    return {k: str(v) for k, v in info.items()}     # plain str: torch.__version__ is a TorchVersion object


def _to_torch(obj):
    import torch

    if isinstance(obj, np.ndarray):
        return torch.from_numpy(np.array(obj, order="C", copy=True))   # (ascontiguousarray turns 0-d into 1-d)
    if isinstance(obj, np.generic):        # numpy scalars are not weights_only-safe: keep them as 0-d (same dtype)
        return torch.from_numpy(np.array(obj)) if obj.dtype.kind in "biuf" else obj.item()
    if isinstance(obj, dict):
        return {k: _to_torch(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_torch(v) for v in obj]
    return obj


def _to_numpy(obj):
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.numpy()
    if isinstance(obj, dict):
        return {k: _to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_numpy(v) for v in obj]
    return obj


def save(cosim, path: str | Path, extra: dict | None = None) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "format": FORMAT, "version": VERSION, "build": build_info(),
        "connectome_sha256": cosim.jax.net.digest, "W": cosim.W,
        "clock": cosim.clock.state_dict(), "bus": cosim.bus.state_dict(),
        "partitions": {p.name: p.state_dict() for p in cosim.partitions},
        "extra": extra or {},
    }
    torch.save(_to_torch(blob), path)
    return path


def read(cosim, path: str | Path) -> dict:
    """Read a brain save and check that it fits this brain (format, version, connectome, window, partitions),
    changing nothing. Raises ValueError if it does not fit."""
    import torch

    blob = _to_numpy(torch.load(Path(path), weights_only=True))
    if not isinstance(blob, dict) or blob.get("format") != FORMAT or blob.get("version") != VERSION:
        raise ValueError("not a Tome brain save of this version")
    if blob["connectome_sha256"] != cosim.jax.net.digest:
        raise ValueError("this save was made with a different connectome; refusing to load it")
    if int(blob["W"]) != cosim.W:
        raise ValueError(f"this save was made with a {blob['W']}-ms window, not {cosim.W}")
    if blob["build"] != build_info():
        warnings.warn("loading a brain saved by a different software build; it may not continue bit-exactly")
    names = {p.name for p in cosim.partitions}
    if set(blob["partitions"]) != names:
        raise ValueError(f"partitions differ: saved {sorted(blob['partitions'])}, have {sorted(names)}")
    for p in cosim.partitions:
        p.check_state_dict(blob["partitions"][p.name])
    return blob


def apply(cosim, blob: dict) -> dict:
    """Load a save that ``read`` accepted. Returns its ``extra``."""
    for p in cosim.partitions:
        p.load_state_dict(blob["partitions"][p.name])
    cosim.bus.load_state_dict(blob["bus"])
    cosim.clock.load_state_dict(blob["clock"])
    return blob["extra"]


def load(cosim, path: str | Path) -> dict:
    return apply(cosim, read(cosim, path))
