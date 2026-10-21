"""Developer mode. Debug telemetry (the brains' internals, every check) exists only when ``VOIDLIT_DEV=1``.

A release build never sends it: the player must not be able to see a die, a check or the Tome's mind.
"""

from __future__ import annotations

import os

DEV_ENV = "VOIDLIT_DEV"


def dev_mode() -> bool:
    return os.environ.get(DEV_ENV) == "1"


def require_dev(feature: str) -> None:
    if not dev_mode():
        raise PermissionError(f"{feature} is a developer feature; set {DEV_ENV}=1 to enable it")
