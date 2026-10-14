"""Scripted players for automated playtests and balance reports.

The bot answers in plain text, the way a player would, so a playtest exercises the senses, the turn loop and
the voice as well as the rules.
"""

from __future__ import annotations

from .rules.bearer import STATS, Stat
from .tome.engine import TomeEngine

KIND_STATS = {"hostile": ("might", "grace", "nerve"), "hazard": ("grace", "might"), "relic": ("wits", "might"),
              "witness": ("nerve", "wits")}
STAT_COMMAND = {"might": "I force it with everything I have", "grace": "I slip past it, quick and careful",
                "wits": "I study it and work out how it fits together", "nerve": "I hold steady and face it"}


def bot_command(engine: TomeEngine, smart: bool = True) -> str:
    """A simple, legal player. The smart one answers with its best accepted stat, pushes when a Hit is close
    and patches Wounds; the naive one always uses the first accepted stat and never pushes."""
    st, b, tr = engine.state, engine.bearer, engine.trial
    if st.stage == "haven":
        return "I go on to the next threshold"
    if st.stage == "fallen":
        return "Someone new takes up the TOME"
    if tr is None:
        return "I open the TOME"
    p = tr.pending
    if p:
        kind = p["type"]
        if kind == "draw":
            return "I touch the first one"
        if kind == "push":
            return "again" if smart else "no"
        if kind == "bargain":
            weakest = min(STATS, key=lambda s: b.base(s))
            return f"yes, take my {weakest.value}"
        if kind == "breather":
            return "I breathe and calm down" if b.stress >= 4 else "I patch myself up"
        if kind == "warning":
            return "keep the order"
        if kind == "temptation":
            return "no"
        if kind == "judgment":
            return "I accept"
    ch = tr.chapter
    if ch is None:
        return "I go on"
    if smart and b.bleeding and b.class_id != "vessel" and b.find_item("medkit") and (b.find_item("medkit").uses or 0) > 0:
        return "I use the medkit on my wound"
    accepted = KIND_STATS[ch.active_kind.value]
    stat = max(accepted, key=lambda s: b.stat(Stat(s))) if smart else accepted[0]
    return STAT_COMMAND[stat]


def play(engine: TomeEngine, max_turns: int = 400, smart: bool = True) -> dict:
    """Play until the game is over or the TOME has run one trial and reached the Haven."""
    engine.start()
    turns = 0
    while turns < max_turns:
        frame = engine.turn(bot_command(engine, smart))
        turns += 1
        if frame.over or engine.state.stage == "haven":
            break
    return {"turns": turns, "survived": not engine.bearer.dead and engine.state.stage == "haven",
            "dead": engine.bearer.dead, "stage": engine.state.stage}


def play_campaign(engine: TomeEngine, trials: int = 12, max_turns: int = 4000, smart: bool = True) -> dict:
    """Play trial after trial, through Havens and Legacies, until the campaign ends or ``trials`` have run."""
    engine.start()
    turns = 0
    while turns < max_turns and engine.state.stage != "over" and engine.state.campaign.trials < trials:
        engine.turn(bot_command(engine, smart))
        turns += 1
    c = engine.state.campaign
    return {"turns": turns, "trials": c.trials, "fallen": len(c.fallen), "fragments": c.fragments,
            "hunger": c.hunger, "stage": engine.state.stage}
