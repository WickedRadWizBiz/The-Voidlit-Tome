"""SensoryState: what the Tome perceives in one line of player input."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from ..rules.bearer import Stat


class Intent(str, Enum):
    LOOK = "look"
    TALK = "talk"
    FIGHT = "fight"
    FORCE = "force"
    SNEAK = "sneak"
    SEARCH = "search"
    USE = "use"
    REST = "rest"
    ADVANCE = "advance"
    FLEE = "flee"
    OTHER = "other"


# The stat each kind of action rolls when the player does not name one (TOME: Core Rules 0.2, "Stats").
DEFAULT_STAT: dict[Intent, Stat] = {
    Intent.LOOK: Stat.NONE,
    Intent.TALK: Stat.NERVE,
    Intent.FIGHT: Stat.MIGHT,
    Intent.FORCE: Stat.MIGHT,
    Intent.SNEAK: Stat.GRACE,
    Intent.SEARCH: Stat.WITS,
    Intent.USE: Stat.WITS,
    Intent.REST: Stat.NONE,
    Intent.ADVANCE: Stat.NONE,
    Intent.FLEE: Stat.GRACE,
    Intent.OTHER: Stat.NONE,
}
# Intents that engage a chapter (cast the pool); the rest are looked at, said or waited out.
ENGAGING = {Intent.FIGHT, Intent.FORCE, Intent.SNEAK, Intent.SEARCH, Intent.USE, Intent.FLEE, Intent.TALK}

# Channels the limbic (NPC) brains receive, in order.
NPC_CHANNELS = ("aggression", "deception", "leverage", "exposure", "rapport_pos", "rapport_neg")


def _unit(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


@dataclass
class SensoryState:
    intent: Intent = Intent.OTHER
    stat: Stat = Stat.NONE
    target: str = "none"
    aggression: float = 0.0   # hostility, threat, violence            [0, 1]
    deception: float = 0.0    # lying, bluffing, misdirection           [0, 1]
    leverage: float = 0.0     # bargaining power, offers, threats-if    [0, 1]
    exposure: float = 0.0     # probing the listener's secrets/plot     [0, 1]
    rapport: float = 0.0      # warmth (+) or contempt (-)              [-1, 1]
    boldness: float = 0.0     # appetite for risk                       [0, 1]

    def clamped(self) -> "SensoryState":
        return SensoryState(
            intent=self.intent, stat=self.stat, target=self.target or "none",
            aggression=_unit(self.aggression), deception=_unit(self.deception),
            leverage=_unit(self.leverage), exposure=_unit(self.exposure),
            rapport=float(min(1.0, max(-1.0, self.rapport))), boldness=_unit(self.boldness),
        )

    @property
    def effective_stat(self) -> Stat:
        if self.stat is not Stat.NONE:
            return self.stat
        return DEFAULT_STAT[self.intent]

    @property
    def mode(self) -> str:
        """How a Hostile chapter is met: melee, ranged, hide or talk."""
        stat = self.effective_stat
        if self.intent in (Intent.SNEAK, Intent.FLEE):
            return "hide"
        if self.intent is Intent.TALK or stat is Stat.NERVE:
            return "talk"
        return "ranged" if stat is Stat.GRACE else "melee"

    @property
    def shock(self) -> float:
        """How jarring this input is to whoever hears it (drives trauma sprouting)."""
        return max(self.aggression, self.exposure, abs(self.rapport))

    def npc_channels(self) -> list[float]:
        return [self.aggression, self.deception, self.leverage, self.exposure,
                max(0.0, self.rapport), max(0.0, -self.rapport)]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["intent"] = self.intent.value
        d["stat"] = self.stat.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SensoryState":
        return cls(
            intent=Intent(d.get("intent", "other")), stat=Stat(d.get("stat", "none")),
            target=d.get("target", "none"), aggression=d.get("aggression", 0.0),
            deception=d.get("deception", 0.0), leverage=d.get("leverage", 0.0),
            exposure=d.get("exposure", 0.0), rapport=d.get("rapport", 0.0),
            boldness=d.get("boldness", 0.0),
        ).clamped()
