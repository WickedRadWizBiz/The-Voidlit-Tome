"""The pacing teacher: hand-written game-master instincts the Director is trained on.

The TOME's deck already escalates (Depth, Abyssal Leaves, the Lumen, the Crucible), so the teacher only shapes
presentation. Tension should climb through a trial. A struggling Bearer is offered the shapes (a choice), and a
comfortable one is met with the dark. Success invites a revealed truth, and a hostile person speaks up. The
spiking Director distils these into a smooth, noise-tolerant policy; playtest logs can fine-tune it later without
touching this file.
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
    struggling = 0.4 * (1.0 - f["grit"]) + 0.3 * f["stress"] + 0.3 * f["wounds"]
    dark = (f["lumen"] <= 0.01).float()
    logits = {
        Beat.CALM: 0.3 - 1.6 * deficit + 0.3 * (1.0 - f["boldness"]) - 0.6 * f["crucible"] + 0.4 * struggling,
        Beat.FORESHADOW: 0.1 + 2.0 * deficit + 0.5 * f["boldness"] + 0.8 * (1.0 - f["lumen"]) + 0.6 * f["crucible"]
        - 0.8 * struggling + 0.8 * dark,
        Beat.CLUE: -0.1 + 0.9 * f["last_success"] + 0.4 * (1.0 - f["depth_left"]) - 0.4 * f["last_fail"],
        Beat.NPC: -1.2 + 1.5 * f["npc_hostility"] + 1.0 * f["npc_present"],
        Beat.DOT: 0.5 - 0.6 * struggling,
        Beat.SHAPES: -0.4 + 1.8 * struggling + 0.4 * (1.0 - f["progress"]),
        Beat.DARK: -0.7 + 1.6 * deficit + 0.8 * f["hunger"] + 0.4 * f["boldness"] - 0.5 * struggling,
    }
    return torch.stack([logits[b] for b in BEATS], dim=1)


def teacher_distribution(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(teacher_logits(x) / TEMPERATURE, dim=1)


def sample_states(n: int, generator: torch.Generator) -> torch.Tensor:
    """Plausible random table states for training (consistent one-hots and couplings)."""
    x = torch.rand(n, N_FEATURES, generator=generator)
    idx = {name: i for i, name in enumerate(DirectorFeatures.__dataclass_fields__)}
    inside = (torch.rand(n, generator=generator) < 0.6).float()
    x[:, idx["in_chapter"]] = inside
    x[:, idx["depth_left"]] *= inside
    x[:, idx["crucible"]] = inside * (torch.rand(n, generator=generator) < 0.2).float()
    x[:, idx["wounds"]] = torch.randint(0, 4, (n,), generator=generator).float() / 3
    last = torch.randint(0, 4, (n,), generator=generator)  # 0 none, 1 fail, 2 short, 3 success
    for k, name in ((1, "last_fail"), (2, "last_short"), (3, "last_success")):
        x[:, idx[name]] = (last == k).float()
    x[:, idx["npc_present"]] = (torch.rand(n, generator=generator) < 0.3).float()
    x[:, idx["npc_hostility"]] *= x[:, idx["npc_present"]]
    dark = torch.rand(n, generator=generator) < 0.1
    x[dark, idx["lumen"]] = 0.0
    return x
