"""The pacing teacher: hand-written Dungeon Master instincts the Director is trained on.

These are the designer's rules of thumb for a horror one-shot. Tension should
climb with the act, and a struggling player gets air. A stalled scene gets a
push, and success invites a twist. The spiking Director distils them into a
smooth, noise-tolerant policy, and playtest logs can fine-tune it later
without touching this file.
"""

from __future__ import annotations

import torch

from .features import BEATS, N_FEATURES, Beat, DirectorFeatures

TEMPERATURE = 0.5


def target_tension(progress: torch.Tensor) -> torch.Tensor:
    return 0.2 + 0.6 * progress


def teacher_logits(x: torch.Tensor) -> torch.Tensor:
    """Batch of features [N, F] -> beat logits [N, len(BEATS)]."""
    f = {name: x[:, i] for i, name in enumerate(DirectorFeatures.__dataclass_fields__)}
    deficit = target_tension(f["progress"]) - f["tension"]
    struggling = 0.5 * (1.0 - f["hull"]) + 0.5 * f["stress"]
    ob = f["obstacle_active"]
    dark = (f["lumen"] <= 0.01).float()
    logits = {
        Beat.CALM: 0.3 - 1.6 * deficit - 0.8 * ob + 0.3 * (1.0 - f["boldness"]) - 0.6 * f["chapter_turns"],
        Beat.ESCALATE: 0.1 + 2.2 * deficit + 0.8 * f["since_threat"] + 0.5 * f["boldness"]
        - 1.0 * ob - 0.9 * struggling + 1.5 * dark + 0.4 * f["clock_max"],
        Beat.CLUE: -0.1 + 0.9 * f["last_success"] + 0.4 * (1.0 - ob) - 0.4 * f["last_fail"],
        Beat.NPC: -0.7 + 1.5 * f["npc_hostility"] + 0.4 * f["npc_present"] + 0.2 * (1.0 - ob),
        Beat.HAZARD: -0.1 + 1.5 * deficit + 0.6 * (1.0 - ob) + 0.5 * f["boldness"] + 0.6 * (1.0 - f["lumen"])
        - 0.5 * struggling,
        Beat.RESPITE: -0.8 + 2.6 * struggling + 0.8 * f["since_respite"] - 1.6 * ob - 1.2 * dark
        - 0.8 * torch.clamp(deficit, min=0.0),
        Beat.TWIST: -0.5 + 1.0 * deficit + 1.0 * f["last_success"] * f["progress"] + 0.6 * f["clock_max"]
        + 0.3 * f["obstacle_left"] * ob,
        Beat.ADVANCE: -0.6 + 2.0 * (1.0 - ob) * f["chapter_turns"] + 0.3 * f["progress"] - 0.5 * deficit,
    }
    return torch.stack([logits[b] for b in BEATS], dim=1)


def teacher_distribution(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(teacher_logits(x) / TEMPERATURE, dim=1)


def sample_states(n: int, generator: torch.Generator) -> torch.Tensor:
    """Plausible random table states for training (consistent one-hots and couplings)."""
    x = torch.rand(n, N_FEATURES, generator=generator)
    idx = {name: i for i, name in enumerate(DirectorFeatures.__dataclass_fields__)}
    ob = (torch.rand(n, generator=generator) < 0.45).float()
    x[:, idx["obstacle_active"]] = ob
    x[:, idx["obstacle_left"]] *= ob
    last = torch.randint(0, 4, (n,), generator=generator)  # 0 none, 1 fail, 2 mixed, 3 success
    for k, name in ((1, "last_fail"), (2, "last_mixed"), (3, "last_success")):
        x[:, idx[name]] = (last == k).float()
    x[:, idx["npc_present"]] = (torch.rand(n, generator=generator) < 0.5).float()
    x[:, idx["npc_hostility"]] *= x[:, idx["npc_present"]]
    dark = torch.rand(n, generator=generator) < 0.1
    x[dark, idx["lumen"]] = 0.0
    return x
