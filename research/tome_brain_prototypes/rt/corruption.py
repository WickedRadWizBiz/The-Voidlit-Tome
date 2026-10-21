"""NPC corruption projection: the Tome -> an NPC's own BindsNET brain, through the brain's public surface only.

The NPC brain code is untouched. Corruption acts where gossip already acts (plot momentum, grafted branches) plus
on what the NPC perceives (a SensoryState blend), at turn boundaries (coupling delay = one turn >> D_MIN).

State per NPC: strength c in [0, c_cap(desp)], resistance r (refractory), cooldown counter.
* A burst can only be attempted when the dominion loop selects that NPC (salience ~ desperation x opportunity) and
  the NPC is not refractory. The attempt is the Tome's invisible d20 check: d20 + Dominion vs 10 + Resolve + r.
* Success: c <- min(c_cap, c + A (1 - c)); grafts one Tome-sourced branch (source_npc = TOME_ID) on a strong burst.
* Every turn: c decays (tau 2 turns); r decays (tau 4 turns); after a burst r += 3 and cooldown = 2 turns.
* c_cap = 0.30 + 0.45 * desperation  (<= 0.75: the NPC's own brain always keeps >= 25% of the say).
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field

import torch

from voidlit.senses.state import SensoryState

TOME_ID = 9
PERC_GAIN = 1.0   # whisper strength relative to c (calibrated below)


@dataclass
class Corruption:
    c: float = 0.0
    r: float = 0.0
    cooldown: int = 0
    bursts: int = 0
    attempts: int = 0

    def to_dict(self):
        return dict(self.__dict__)


def cap(desp: float) -> float:
    return 0.30 + 0.45 * max(0.0, min(1.0, desp))


def tome_view(s: SensoryState) -> SensoryState:
    """How the Tome wants the NPC to perceive the player: hiding something, contemptuous, not to be trusted."""
    return SensoryState(intent=s.intent, approach=s.approach, target=s.target, aggression=min(1, s.aggression + 0.3),
                        deception=0.8, leverage=s.leverage, exposure=0.7, rapport=-0.6, boldness=s.boldness)


def blend(s: SensoryState, t: SensoryState, c: float) -> SensoryState:
    mix = lambda a, b: (1 - c) * a + c * b
    return SensoryState(intent=s.intent, approach=s.approach, target=s.target, aggression=mix(s.aggression, t.aggression),
                        deception=mix(s.deception, t.deception), leverage=mix(s.leverage, t.leverage),
                        exposure=mix(s.exposure, t.exposure), rapport=mix(s.rapport, t.rapport),
                        boldness=s.boldness)


def attempt(state: Corruption, desp: float, dominion_mod: int, resolve: int, rng: random.Random,
            selected: bool) -> bool:
    """The Tome's corruption check against an NPC (invisible). Returns True on a landed burst."""
    if not selected or state.cooldown > 0:
        return False
    state.attempts += 1
    dc = 10 + resolve + int(round(state.r))
    roll = rng.randint(1, 20) + dominion_mod
    if roll < dc:
        state.r += 1.0                     # a failed push still alerts the NPC's own mind
        return False
    A = 0.25 + 0.35 * desp + 0.05 * (roll - dc)
    state.c = min(cap(desp), state.c + A * (1 - state.c))
    state.r += 3.0
    state.cooldown = 2
    state.bursts += 1
    return True


def decay(state: Corruption) -> None:
    state.c *= math.exp(-1 / 2.0)
    state.r *= math.exp(-1 / 4.0)
    state.cooldown = max(0, state.cooldown - 1)


def apply(brain, state: Corruption, s: SensoryState, target_plot: int, graft: bool, transient: bool = True,
          burst_now: bool = True):
    """Bias the NPC's perception and plot momentum by the current corruption strength; optionally graft.
    Returns (perceived state, undo). With transient=True the momentum push is removed after perceive (undo()),
    so the Tome's hand leaves no permanent edit; only what the NPC's own plasticity made of the turn persists."""
    if state.c <= 1e-3:
        return s, (lambda: None)
    delta = 0.30 * state.c
    brain.momentum[target_plot] = float(min(1.0, float(brain.momentum[target_plot]) + delta))
    def undo():
        if transient:   # perceive() decays momentum by 0.9 before adding spikes: remove the decayed push
            brain.momentum[target_plot] = float(max(0.0, float(brain.momentum[target_plot]) - 0.9 * delta))
    if graft:
        cols = [0, 3, 5]  # aggression, exposure, rapport- channels (suspicion of the player)
        brain.spawn_branch(target_plot, cols, 0.35 + 0.4 * state.c, origin_role="betray", source_npc=TOME_ID,
                           evict=True)
    # the whisper (perception bias) only on the burst turn itself; between bursts only the fading push remains
    return (blend(s, tome_view(s), PERC_GAIN * state.c) if burst_now else s), undo
