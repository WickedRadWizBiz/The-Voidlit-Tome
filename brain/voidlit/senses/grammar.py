"""GBNF grammar for llama.cpp's sensory pass, plus a strict parser for its output.

The grammar is rebuilt every turn so the ``target`` field can only name
entities that are actually in the scene. Every key and all punctuation are
fixed literals, so the model samples only the enum choices and about a dozen
numeric tokens.

llama.cpp's GBNF parser ends a rule at a newline unless it is inside
parentheses, so the multi-line ``root`` body is wrapped in ``( ... )``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

from ..rules.bearer import Stat
from .state import Intent, SensoryState

_FLOAT_FIELDS = ("aggression", "deception", "leverage", "exposure")
_ID_RE = re.compile(r"[^a-z0-9_]")


def sanitize_id(s: str) -> str:
    return _ID_RE.sub("_", s.strip().lower())[:32] or "none"


def _alts(values: Iterable[str]) -> str:
    return " | ".join(f'"{v}"' for v in values)


def build_sensory_gbnf(targets: Iterable[str] = ()) -> str:
    ids = ["none"] + sorted({sanitize_id(t) for t in targets if sanitize_id(t) != "none"})
    fields = [
        r'"{\"intent\": \"" intent "\", "',
        r'"\"stat\": \"" stat "\", "',
        r'"\"target\": \"" target "\", "',
    ]
    fields += [rf'"\"{f}\": " unit ", "' for f in _FLOAT_FIELDS]
    fields += [r'"\"rapport\": " signed ", "', r'"\"boldness\": " unit "}"']
    return "\n".join([
        "root ::= (\n  " + "\n  ".join(fields) + "\n)",
        f"intent ::= {_alts(i.value for i in Intent)}",
        f"stat ::= {_alts(a.value for a in Stat)}",
        f"target ::= {_alts(ids)}",
        'unit ::= ("0." [0-9] [0-9]?) | "1.0"',
        'signed ::= "-"? unit',
        "",
    ])


class SensoryParseError(ValueError):
    pass


def parse_sensory_json(text: str, targets: Iterable[str] = ()) -> SensoryState:
    """Parse the grammar-constrained JSON. Raises SensoryParseError on anything off-schema."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise SensoryParseError(f"no JSON object in: {text[:80]!r}")
    try:
        d = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise SensoryParseError(str(e)) from e
    try:
        intent = Intent(d["intent"])
        stat = Stat(d["stat"])
        floats = {k: float(d[k]) for k in (*_FLOAT_FIELDS, "rapport", "boldness")}
    except (KeyError, ValueError, TypeError) as e:
        raise SensoryParseError(f"bad sensory field: {e}") from e
    known = {sanitize_id(t) for t in targets}
    target = sanitize_id(str(d.get("target", "none")))
    if target not in known:
        target = "none"
    return SensoryState(intent=intent, stat=stat, target=target, **floats).clamped()
