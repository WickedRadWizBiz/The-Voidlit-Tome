"""Save and load the whole brain (design §13): every partition, the bus, the clock and the noise keys.

The file is a ``torch.save`` of tensors and plain containers only, read back with ``weights_only=True``. Loading
refuses a different format, version or connectome; a different software build only warns, because the state is
still valid but bit-exact continuation is guaranteed only on the same build. ``read`` makes every check without
changing the brain and ``apply`` then loads it, all or nothing, so a game can refuse a save before it changes anything
else. ``snapshot`` is a save held in memory: applying it takes the brain back to that moment (a failed load's undo).
"""

from __future__ import annotations

import platform
import warnings
from pathlib import Path

import numpy as np

from .partition import check_shapes, shapes_of

# 2: the basal ganglia's and the amygdala's scalars by name, every loop's fallbacks (M6)
FORMAT, VERSION = "voidlit.tomebrain", 2


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


def snapshot(cosim, extra: dict | None = None) -> dict:
    """The whole brain as a save holds it (arrays, not tensors), in memory."""
    return {
        "format": FORMAT, "version": VERSION, "build": build_info(),
        "connectome_sha256": cosim.jax.net.digest, "W": cosim.W,
        "clock": cosim.clock.state_dict(), "bus": cosim.bus.state_dict(),
        "partitions": {p.name: p.state_dict() for p in cosim.partitions},
        "extra": extra or {},
    }


def save(cosim, path: str | Path, extra: dict | None = None) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_to_torch(snapshot(cosim, extra)), path)
    return path


def check(cosim, blob: dict) -> None:
    """Raise ValueError unless ``blob`` fits this brain whole (format, version, connectome, window, every partition,
    the bus and the clock), changing nothing."""
    if not isinstance(blob, dict) or blob.get("format") != FORMAT:
        raise ValueError("not a Tome brain save")
    version = blob.get("version")
    if isinstance(version, int) and 0 < version < VERSION:
        raise ValueError(f"this brain was saved by an earlier version of the Tome (brain save version {version}; this "
                         f"build reads version {VERSION} and does not convert older ones)")
    if version != VERSION:
        raise ValueError("not a Tome brain save of this version")
    try:
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
        clock, bus = blob["clock"], blob["bus"]
        check_shapes("clock", clock, {"t_ms": (), "turn": (), "phase": None})
        check_shapes("bus", bus, shapes_of(cosim.bus.state_dict()))
        if int(bus["W"]) != cosim.W or clock["phase"] not in ("idle", "A", "B", "quiet"):
            raise ValueError("the bus or the clock does not fit this brain")
        # every window writes every source, so the bus holds each one up to the tick before the clock's
        if any(int(t) != int(clock["t_ms"]) - 1 for t in bus["written"].values()):
            raise ValueError("the bus and the clock disagree")
        if not isinstance(blob["extra"], dict):
            raise ValueError("the save's extra state is not a dict")
    except (KeyError, TypeError, IndexError) as e:
        raise ValueError(f"damaged brain save: {e!r}") from None


def read(cosim, path: str | Path) -> dict:
    """Read a brain save and ``check`` it, changing nothing. Raises ValueError if it does not fit."""
    import torch

    blob = _to_numpy(torch.load(Path(path), weights_only=True))
    check(cosim, blob)
    return blob


def _apply(cosim, blob: dict) -> None:
    for p in cosim.partitions:
        p.load_state_dict(blob["partitions"][p.name])
    cosim.bus.load_state_dict(blob["bus"])
    cosim.clock.load_state_dict(blob["clock"])


def apply(cosim, blob: dict) -> dict:
    """Load a save that ``read`` accepted, or a ``snapshot``, all or nothing: should a part still fail to take it, the
    brain takes back what it held and the error is raised. Returns its ``extra``."""
    held = snapshot(cosim)
    try:
        _apply(cosim, blob)
    except Exception:
        _apply(cosim, held)         # (should even this fail, its error is raised: the brain is then broken)
        raise
    return blob["extra"]


def load(cosim, path: str | Path) -> dict:
    """Read, check and load a brain save, all or nothing. Returns its ``extra``."""
    return apply(cosim, read(cosim, path))
