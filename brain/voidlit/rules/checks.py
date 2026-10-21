"""Invisible checks. Every action is decided by a die, and the player never sees one.

* A **direct action** by any character rolls **d20 + modifier** against a DC. The margin sets the degree:

  ======== ====================
  critical margin >= 10
  success  margin >= 0
  mixed    margin >= -5
  fail     otherwise
  ======== ====================

  A natural 20 moves the result up one step and a natural 1 moves it down one.
* A **significant upper hand** while directly opposing the Tome adds **+1d6** to the total. A defender with the
  upper hand adds the d6 to the DC instead.
* A **passive event** rolls **d6** (+ a score) against a target.

Rolls come from the engine's seeded ``random.Random``. Each one is appended to the engine's ``CheckLog``, which
never leaves the process: not in frames, not in the protocol, not in narration.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from .dice import Outcome, PoolRoll
from .relevance import D6, D20, NO_CHECK

CRITICAL_MARGIN = 10
MIXED_MARGIN = -5
_STEPS = (Outcome.FAIL, Outcome.MIXED, Outcome.SUCCESS, Outcome.CRITICAL)


@dataclass(frozen=True)
class CheckRecord:
    actor: str        # "player", an NPC id, or "tome"
    action: str       # a key of relevance.D20 / D6 / NO_CHECK
    die: str          # "d20", "d6" or "none"
    attribute: str
    mod: int          # d20: the modifier; d6: the score added
    dc: int           # the target (for d20, including any defender d6)
    natural: int
    upper_hand_d6: int = 0
    defence_d6: int = 0
    total: int = 0
    degree: Outcome = Outcome.SUCCESS

    @property
    def margin(self) -> int:
        return self.total - self.dc

    def to_dict(self) -> dict:
        d = asdict(self)
        d["degree"] = self.degree.value
        return d


class CheckLog:
    """Every check this run, in order. Internal only (G21 reads it; dev builds may save it)."""

    def __init__(self, limit: int = 4000) -> None:
        self.records: list[CheckRecord] = []
        self.limit = limit

    def add(self, rec: CheckRecord) -> CheckRecord:
        self.records.append(rec)
        if len(self.records) > self.limit:
            del self.records[: len(self.records) - self.limit]
        return rec

    def note(self, actor: str, action: str) -> None:
        """Record an action that takes no check (it must be on the no-check list)."""
        if action not in NO_CHECK:
            raise KeyError(f"{action!r} is not on the no-check list")
        self.add(CheckRecord(actor, action, "none", "none", 0, 0, 0))

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)


def degree_for(natural: int, total: int, dc: int) -> Outcome:
    m = total - dc
    step = 3 if m >= CRITICAL_MARGIN else 2 if m >= 0 else 1 if m >= MIXED_MARGIN else 0
    if natural == 20:
        step = min(3, step + 1)
    elif natural == 1:
        step = max(0, step - 1)
    return _STEPS[step]


def d20(rng: random.Random, log: CheckLog | None, actor: str, action: str, attribute: str, modifier: int,
        dc: int, *, upper_hand: bool = False, defender_upper_hand: bool = False) -> CheckRecord:
    if action not in D20:
        raise KeyError(f"{action!r} is not a d20 action")
    if upper_hand and not D20[action].upper_hand:
        raise ValueError(f"{action!r} does not oppose the Tome; no upper-hand die")
    natural = rng.randint(1, 20)
    uh = rng.randint(1, 6) if upper_hand else 0
    dd = rng.randint(1, 6) if defender_upper_hand else 0
    total = natural + modifier + uh
    rec = CheckRecord(actor, action, "d20", str(attribute), modifier, dc + dd, natural, uh, dd, total,
                      degree_for(natural, total, dc + dd))
    return log.add(rec) if log is not None else rec


def d6(rng: random.Random, log: CheckLog | None, actor: str, action: str, attribute: str, score: int,
       target: int, *, strict: bool = False) -> CheckRecord:
    """Passive d6 + score against a target (>= target, or > target when ``strict``)."""
    if action not in D6:
        raise KeyError(f"{action!r} is not a passive d6 event")
    natural = rng.randint(1, 6)
    total = natural + score
    ok = total > target if strict else total >= target
    rec = CheckRecord(actor, action, "d6", str(attribute), score, target, natural, total=total,
                      degree=Outcome.SUCCESS if ok else Outcome.FAIL)
    return log.add(rec) if log is not None else rec


def d6_table(rng: random.Random, log: CheckLog | None, action: str, n: int) -> int:
    """Pick one of ``n`` table rows with a d6 (rows repeat when n < 6). Returns a 0-based index."""
    if action not in D6:
        raise KeyError(f"{action!r} is not a passive d6 event")
    natural = rng.randint(1, 6)
    if log is not None:
        log.add(CheckRecord("world", action, "d6", "none", 0, 0, natural, total=natural))
    return (natural - 1) * n // 6


def as_pool(rec: CheckRecord) -> PoolRoll:
    """Adapter onto the obstacle rules: fail 0 hits, mixed 1, success 2, critical 2 with two sixes."""
    hits = {Outcome.FAIL: 0, Outcome.MIXED: 1, Outcome.SUCCESS: 2, Outcome.CRITICAL: 2}[rec.degree]
    return PoolRoll(dice=(rec.natural,), pool=rec.mod, hits=hits, sixes=2 if rec.degree is Outcome.CRITICAL else 0)
