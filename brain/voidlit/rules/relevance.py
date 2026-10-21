"""Which check every action gets: the die, the attribute, and what it is opposed by.

Every action any character takes (the player, an NPC, the Tome) lands in exactly one place:

* ``D20``: a direct action by a character, d20 + attribute modifier against a DC;
* ``D6``: a passive event that no character directly does;
* ``NO_CHECK``: unopposed acts with no failure state (speech nobody resists, feeding the light).

The engine refuses to roll for an action that is not in the matching table, and the coverage test (G21) checks
that every action the engine records is listed here. ``dc`` is the DC rule as documentation; the engine computes
it. ``upper_hand`` marks actions that directly oppose the Tome, where a significant upper hand adds +1d6.
"""

from __future__ import annotations

from dataclasses import dataclass

from .attributes import Attribute

APPROACH = "approach"  # the attribute follows the approach the player chose


@dataclass(frozen=True)
class Rule:
    attribute: Attribute | str
    dc: str
    upper_hand: bool = False


G, U, I, R = Attribute.GRIT, Attribute.GUILE, Attribute.INSIGHT, Attribute.RESOLVE

D20: dict[str, Rule] = {
    # the player
    "player.engage": Rule(APPROACH, "obstacle", upper_hand=True),  # +d6 only against a Tome-twisted obstacle
    "player.search": Rule(I, "obstacle"),
    "player.advance_hazard": Rule(U, "obstacle"),
    "player.flee": Rule(U, "obstacle"),
    "player.look": Rule(I, "10 + act"),
    "player.rest": Rule(R, "8 + act"),
    "player.heal": Rule(I, "8"),
    "player.lie_npc": Rule(U, "10 + npc insight"),
    "player.threaten_npc": Rule(G, "10 + npc resolve"),
    "player.appeal_npc": Rule(I, "10 + npc insight"),
    "player.probe_npc": Rule(I, "10 + npc guile"),
    "player.bargain_npc": Rule(U, "10 + npc insight"),
    "player.attack_npc": Rule(G, "10 + npc guile"),
    "player.strike_tome": Rule(G, "10 + tome grit", upper_hand=True),
    "player.smother_tome": Rule(U, "10 + tome insight", upper_hand=True),
    "player.lie_tome": Rule(U, "10 + tome insight", upper_hand=True),
    "player.threaten_tome": Rule(G, "10 + tome resolve", upper_hand=True),
    # NPCs (their tactics, as the player hears them)
    "npc.help": Rule(APPROACH, "10"),
    "npc.misdirect": Rule(U, "10 + player insight"),
    "npc.probe": Rule(I, "10 + player guile"),
    "npc.interrogate": Rule(G, "10 + player resolve"),
    "npc.menace": Rule(G, "10 + player resolve"),
    "npc.bargain": Rule(U, "10 + player insight"),
    # the Tome
    "tome.escalate": Rule(G, "10 + player guile"),
    "tome.hazard": Rule(G, "10 + player insight"),
    "tome.twist": Rule(I, "10 + player guile"),
}

D6: dict[str, Rule] = {
    "passive.lumen": Rule("pool", "each 1 is lost"),
    "passive.panic": Rule(R, "d6 + resolve + 3 > stress (+1 phobia)"),
    "passive.last_light": Rule(G, "d6 + grit >= 5 + overflow"),
    "passive.hazard_pick": Rule("none", "table"),
    "passive.twist_pip": Rule("none", "1-2 grit, 3-4 guile, 5-6 insight"),
    "passive.drop_item": Rule("none", "table"),
    "passive.gossip": Rule("none", ">= 4"),
}

NO_CHECK: frozenset[str] = frozenset({
    "player.feed_light",       # no failure state
    "player.ask_tome",         # an unopposed question; a truthful answer is a rule, not a roll
    "player.talk_alone",       # nobody to answer
    "player.chat_npc",         # plain words nobody resists
    "player.advance_open",     # the way is already open
    "player.advance_blocked",  # refused before anything is attempted
    "player.rest_refused",     # a threat is present, or rest is spent
    "player.flee_nothing",     # nothing pursues
    "player.idle",             # nothing in the scene resists or changes
    "player.lost_turn",        # panic took the moment
    "npc.reassure", "npc.orders", "npc.confide",
    "tome.calm", "tome.clue", "tome.respite", "tome.advance", "tome.npc",
})


def table_for(action: str) -> str:
    """Which table an action belongs to: 'd20', 'd6' or 'none'. Raises if it is in none (or several)."""
    homes = [name for name, table in (("d20", D20), ("d6", D6), ("none", NO_CHECK)) if action in table]
    if len(homes) != 1:
        raise KeyError(f"action {action!r} must be in exactly one check table, found {homes}")
    return homes[0]
