"""Off-screen gossip: NPC brains infect each other between scenes.

When a speaker A talks to a listener B (directional trust ``T`` = B's trust in
A, in [-1, 1]):

* **Plot resonance:** B's matching plot momentum shifts by ``0.25 * C_dom(A) * T``.
  A hated rival (T < 0) pushes B toward protecting the player instead.
* **Horizontal grafting:** A's most salient branch (``Ca + 0.5 V > 0.45``) is
  cloned into B when ``|T| >= 0.30``, attenuated by ``|T|``, and tagged with A as
  its source. Grafted from a hated rival, a betrayal branch flips into an
  alliance branch: B warns the player about whatever A is planning.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .npc_brain import NPCBrain

RESONANCE = 0.25
SALIENCE_MIN = 0.45
GRAFT_TRUST_MIN = 0.30


@dataclass
class GossipResult:
    speaker: str
    listener: str
    trust: float
    resonated_plot: int | None
    flux: float
    grafted_branch: int | None
    inverted: bool

    def describe(self) -> str:
        parts = [f"{self.speaker} whispered to {self.listener} (trust {self.trust:+.2f})"]
        if self.grafted_branch is not None:
            parts.append(f"grafted branch {self.grafted_branch}" + (" (inverted)" if self.inverted else ""))
        return "; ".join(parts)


def gossip(speaker: NPCBrain, listener: NPCBrain, trust: float) -> GossipResult:
    trust = float(max(-1.0, min(1.0, trust)))
    dom = int(torch.argmax(speaker.momentum))
    role = speaker.spec.plots[dom].role
    target = listener.spec.role_index(role) if trust >= 0 else listener.spec.role_index("ally")
    flux = float(speaker.momentum[dom]) * trust * RESONANCE
    if target is not None:
        listener.momentum[target] = float(min(1.0, max(0.0, float(listener.momentum[target]) + abs(flux))))

    result = GossipResult(speaker.spec.id, listener.spec.id, trust, target, flux, None, False)
    if abs(trust) < GRAFT_TRUST_MIN:
        return result

    best_slot, best_sal = None, SALIENCE_MIN
    for slot in speaker.active_slots():
        m = speaker.meta[slot]
        sal = float(speaker.branches.calcium[slot]) + 0.5 * m.volume
        if sal > best_sal:
            best_slot, best_sal = slot, sal
    if best_slot is None:
        return result

    src = speaker.meta[best_slot]
    src_role = speaker.spec.plots[src.target_plot].role
    inverted = trust < -GRAFT_TRUST_MIN and src_role == "betray"
    plot = listener.spec.role_index("ally" if inverted else src_role)
    if plot is None:
        return result
    a = abs(trust)
    cols = speaker.spine_mask[:, best_slot]
    channels = cols.nonzero().flatten().tolist()
    weight = float(speaker.c_spines.w[cols, best_slot].mean()) * a
    slot = listener.spawn_branch(
        plot, channels, max(0.10, min(0.90, weight)), origin_role=src.origin_role,
        parent=src.branch_id, source_npc=speaker.spec.numeric_id,
        volume=max(0.20, min(0.85, src.volume * a)),
        calcium=0.5 * float(speaker.branches.calcium[best_slot]), evict=True,
    )
    if slot is not None:
        with torch.no_grad():
            w = (speaker.c_spines.w[:, best_slot] * a).clamp(0.10, 0.90)
            listener.c_spines.w[:, slot] = torch.where(cols, w, torch.zeros_like(w))
        result.grafted_branch = listener.meta[slot].branch_id
        result.inverted = inverted
    return result
