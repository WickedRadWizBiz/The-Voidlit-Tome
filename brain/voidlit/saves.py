"""What every save shares: it is written all or nothing, and what it holds is checked against the game's own types.

``write_slot`` writes a save directory beside the old one and puts it in its place only once it is whole, so a save
that fails leaves the old slot as it was. ``check_types`` holds the plain values a save file gives back (JSON, or
``torch.load`` with ``weights_only``) to the types the game declares for them, so a damaged value is refused at load
instead of crashing a later turn.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import os
import shutil
import types
import typing
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np

SLOT_ENTRIES = ("campaign.json", "brains")      # and brain files (*.pt): what a save directory holds


def write_slot(path: str | Path, write: Callable[[Path, str], None]) -> None:
    """Write the save directory ``path`` all or nothing. ``write(directory, save_id)`` fills a new directory beside it,
    stamping every file with the one ``save_id``; that directory takes the place of the old slot only once it is whole
    and on the disk. Raises ValueError, rather than replace it, if ``path`` holds anything a slot does not."""
    path = Path(path).resolve()
    if path.exists() and not (path.is_dir() and all(e in SLOT_ENTRIES or e.endswith(".pt")
                                                    for e in os.listdir(path))):
        raise ValueError(f"{path} is not a save slot; not replacing it")
    path.parent.mkdir(parents=True, exist_ok=True)
    new = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.new")
    new.mkdir()                                     # (not mkdtemp, which would make the slot private)
    try:
        write(new, uuid.uuid4().hex)
        for p in [*new.rglob("*"), new]:            # on the disk before it is put in place
            _fsync(p)
        _swap(new, path)
    finally:
        shutil.rmtree(new, ignore_errors=True)      # (gone already once it is the slot)
    with contextlib.suppress(OSError):              # the swap itself; the slot is in place either way
        _fsync(path.parent)


def check_stamp(what: str, stamp, save_id: str) -> None:
    """Raise ValueError unless the file ``what`` was stamped with ``save_id``: written by the same save as the rest of
    its slot (a slot torn by a save that failed before saves were atomic, or put together by hand, is not one)."""
    if stamp != save_id:
        raise ValueError(f"{what} belongs to another save than campaign.json")


def _fsync(p: Path) -> None:
    """Flush a file, or a directory's entries, to the disk (only POSIX opens a directory, so elsewhere only files)."""
    if p.is_dir() and os.name != "posix":
        return
    fd = os.open(p, os.O_RDONLY if p.is_dir() else os.O_RDWR)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _swap(new: Path, path: Path) -> None:
    """Rename ``new`` to ``path``, setting the old slot aside first (a directory cannot be renamed over a full one)
    and putting it back should the rename fail."""
    if not path.exists():
        os.replace(new, path)
        return
    old = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.old")
    os.replace(path, old)
    try:
        os.replace(new, path)
    except BaseException:
        os.replace(old, path)
        raise
    shutil.rmtree(old, ignore_errors=True)


@functools.cache
def _fields(cls: type) -> dict:
    hints = typing.get_type_hints(cls)
    return {f.name: hints[f.name] for f in dataclasses.fields(cls)}


def _number(value, kinds: str) -> bool:
    """A number of one of numpy's dtype ``kinds``: a Python one (a bool is not a number), or a numpy scalar or 0-d
    array (what a brain file gives back)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return isinstance(value, int) or "f" in kinds
    return isinstance(value, (np.ndarray, np.generic)) and value.shape == () and value.dtype.kind in kinds


def check_types(what: str, value, hint) -> None:
    """Raise ValueError unless ``value`` is of the type ``hint``: ``bool``; ``int``; ``float`` (an int is one too, a
    bool neither); ``str``; ``None``; a union; ``list[X]``, ``set[X]``, ``tuple[X, ...]`` or ``tuple[A, B]`` (a list
    in a save); ``dict[K, V]``; a dataclass (a dict of exactly its fields, each of its field's type); a dict of
    hints (exactly those keys, each of its type); or any other class (an instance of it). The message names the place
    and the types, never a value."""
    if isinstance(hint, dict) or dataclasses.is_dataclass(hint):
        want = hint if isinstance(hint, dict) else _fields(hint)
        if not isinstance(value, dict):
            raise ValueError(f"{what}: {type(value).__name__}, not a record")
        if set(value) != set(want):
            raise ValueError(f"{what}: keys differ: {sorted(map(str, set(value) ^ set(want)))}")
        for k, v in value.items():
            check_types(f"{what}.{k}", v, want[k])
        return
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin in (types.UnionType, typing.Union):
        if value is None and type(None) in args:
            return
        alts = [a for a in args if a is not type(None)]
        if len(alts) == 1:                          # X | None: X's own message says where it does not fit
            return check_types(what, value, alts[0])
        for alt in alts:
            try:
                return check_types(what, value, alt)
            except ValueError:
                pass
        raise ValueError(f"{what}: {type(value).__name__}, not {hint}")
    if origin in (list, set, tuple):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{what}: {type(value).__name__}, not a list")
        if origin is tuple and args[-1] is not Ellipsis:        # tuple[A, B]: one type per place
            if len(value) != len(args):
                raise ValueError(f"{what}: {len(value)} items, not {len(args)}")
            for i, (v, a) in enumerate(zip(value, args)):
                check_types(f"{what}[{i}]", v, a)
            return
        for i, v in enumerate(value):
            check_types(f"{what}[{i}]", v, args[0])
        return
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{what}: {type(value).__name__}, not a dict")
        for k, v in value.items():
            check_types(f"{what} key", k, args[0])
            check_types(f"{what}.{k}", v, args[1])
        return
    if hint is None or hint is type(None):
        fits = value is None
    elif hint is bool:
        fits = isinstance(value, bool)
    elif hint in (int, float):
        fits = _number(value, "iu" if hint is int else "iuf")
    else:
        fits = isinstance(value, hint)
    if not fits:
        raise ValueError(f"{what}: {type(value).__name__}, not {getattr(hint, '__name__', hint)}")
