"""Scripted players for automated playtests and balance reports."""

from __future__ import annotations

from .rules.character import Approach
from .tome.engine import TomeEngine

PIP_COMMAND = {"grit": "I force my way through it", "guile": "I carefully sneak past it",
               "insight": "I analyze it for a weakness"}


def bot_command(engine: TomeEngine, smart: bool = True) -> str:
    """A simple, legal player. The smart one tackles obstacles with its best matching approach and uses
    its kit; the naive one always answers the first pip."""
    st = engine.state
    ch = st.character
    if smart:
        if ch.hull <= 2 and any("heal" in i.tags and i.uses for i in ch.inventory):
            return "I use my " + next(i.name for i in ch.inventory if "heal" in i.tags and i.uses)
        if st.lumen.dice <= 2 and any("light" in i.tags and i.uses for i in ch.inventory):
            return "I light the " + next(i.name for i in ch.inventory if "light" in i.tags and i.uses)
    active = engine._active()
    if active:
        pips = active[0].pips
        if smart:
            def value(a):  # matching hits strip a pip each; off-approach hits need two per pip
                return ch.pool_for(Approach(a))[0] * (1.0 if a in pips else 0.5)
            return PIP_COMMAND[max(sorted(PIP_COMMAND), key=value)]
        return PIP_COMMAND[pips[0]]
    if (ch.hull <= ch.hull_max // 2 or ch.stress >= 4) and not st.rest_used:
        return "I rest for a moment"
    return "I press on"
