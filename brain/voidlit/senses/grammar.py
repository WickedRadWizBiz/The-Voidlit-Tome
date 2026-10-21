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

from ..rules.character import Approach
from .state import Intent, SensoryState

_FLOAT_FIELDS = ("aggression", "deception", "leverage", "exposure")
_ID_RE = re.compile(r"[^a-z0-9_]")


def sanitize_id(s: str) -> str:
    return _ID_RE.sub("_", s.strip().lower())[:32] or "none"


def _alts(values: Iterable[str]) -> str:
    return " | ".join(f'"{v}"' for v in values)


ASKS_FIXED = ("none", "vague", "unclear")


def _ids(values: Iterable[str], fixed: tuple[str, ...] = ("none",)) -> list[str]:
    return list(fixed) + sorted({sanitize_id(v) for v in values} - set(fixed))


def build_sensory_gbnf(targets: Iterable[str] = (), topics: Iterable[str] = (), claims: Iterable[str] = ()) -> str:
    ids = _ids(targets)
    fields = [
        r'"{\"intent\": \"" intent "\", "',
        r'"\"approach\": \"" approach "\", "',
        r'"\"target\": \"" target "\", "',
    ]
    fields += [rf'"\"{f}\": " unit ", "' for f in _FLOAT_FIELDS]
    fields += [r'"\"rapport\": " signed ", "', r'"\"boldness\": " unit ", "',
               r'"\"asks\": \"" asks "\", "', r'"\"claims\": \"" claims "\", "',
               r'"\"addressed_tome\": " bool "}"']
    return "\n".join([
        "root ::= (\n  " + "\n  ".join(fields) + "\n)",
        f"intent ::= {_alts(i.value for i in Intent)}",
        f"approach ::= {_alts(a.value for a in Approach)}",
        f"target ::= {_alts(ids)}",
        f"asks ::= {_alts(_ids(topics, ASKS_FIXED))}",
        f"claims ::= {_alts(_ids(claims))}",
        'bool ::= "true" | "false"',
        'unit ::= ("0." [0-9] [0-9]?) | "1.0"',
        'signed ::= "-"? unit',
        "",
    ])


class SensoryParseError(ValueError):
    pass


def parse_sensory_json(text: str, targets: Iterable[str] = (), topics: Iterable[str] = (),
                       claims: Iterable[str] = ()) -> SensoryState:
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
        approach = Approach(d["approach"])
        floats = {k: float(d[k]) for k in (*_FLOAT_FIELDS, "rapport", "boldness")}
    except (KeyError, ValueError, TypeError) as e:
        raise SensoryParseError(f"bad sensory field: {e}") from e
    known = {sanitize_id(t) for t in targets}
    target = sanitize_id(str(d.get("target", "none")))
    if target not in known:
        target = "none"
    asks = sanitize_id(str(d.get("asks", "none")))
    claim = sanitize_id(str(d.get("claims", "none")))
    return SensoryState(intent=intent, approach=approach, target=target, **floats,
                        asks=asks if asks in _ids(topics, ASKS_FIXED) else "none",
                        claims=claim if claim in _ids(claims) else "none",
                        addressed_tome=d.get("addressed_tome") is True or target == "tome").clamped()
