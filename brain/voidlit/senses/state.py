"""SensoryState: what the Tome perceives in one line of player input."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from ..rules.character import Approach


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


DEFAULT_APPROACH: dict[Intent, Approach] = {
    Intent.LOOK: Approach.NONE,
    Intent.TALK: Approach.NONE,
    Intent.FIGHT: Approach.GRIT,
    Intent.FORCE: Approach.GRIT,
    Intent.SNEAK: Approach.GUILE,
    Intent.SEARCH: Approach.INSIGHT,
    Intent.USE: Approach.INSIGHT,
    Intent.REST: Approach.NONE,
    Intent.ADVANCE: Approach.NONE,
    Intent.FLEE: Approach.GUILE,
    Intent.OTHER: Approach.NONE,
}

# Channels the limbic (NPC) brains receive, in order.
NPC_CHANNELS = ("aggression", "deception", "leverage", "exposure", "rapport_pos", "rapport_neg")


def _unit(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


@dataclass
class SensoryState:
    intent: Intent = Intent.OTHER
    approach: Approach = Approach.NONE
    target: str = "none"
    aggression: float = 0.0   # hostility, threat, violence            [0, 1]
    deception: float = 0.0    # lying, bluffing, misdirection           [0, 1]
    leverage: float = 0.0     # bargaining power, offers, threats-if    [0, 1]
    exposure: float = 0.0     # probing the listener's secrets/plot     [0, 1]
    rapport: float = 0.0      # warmth (+) or contempt (-)              [-1, 1]
    boldness: float = 0.0     # appetite for risk                       [0, 1]

    def clamped(self) -> "SensoryState":
        return SensoryState(
            intent=self.intent, approach=self.approach, target=self.target or "none",
            aggression=_unit(self.aggression), deception=_unit(self.deception),
            leverage=_unit(self.leverage), exposure=_unit(self.exposure),
            rapport=float(min(1.0, max(-1.0, self.rapport))), boldness=_unit(self.boldness),
        )

    @property
    def effective_approach(self) -> Approach:
        if self.approach is not Approach.NONE:
            return self.approach
        return DEFAULT_APPROACH[self.intent]

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
        d["approach"] = self.approach.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SensoryState":
        return cls(
            intent=Intent(d.get("intent", "other")), approach=Approach(d.get("approach", "none")),
            target=d.get("target", "none"), aggression=d.get("aggression", 0.0),
            deception=d.get("deception", 0.0), leverage=d.get("leverage", 0.0),
            exposure=d.get("exposure", 0.0), rapport=d.get("rapport", 0.0),
            boldness=d.get("boldness", 0.0),
        ).clamped()
