"""Dendritic branch compartments as a BindsNET node population.

Each node is one dendritic branch. Spine synapses sum into the branch
voltage; when it crosses ``theta`` the branch fires an NMDA-style *plateau*:
it emits spikes for ``plateau_ms`` steps, which is the super-linear output
that drives its plot ensemble. Weak, scattered input decays away and never
reaches the soma. Each plateau onset adds to a slow calcium store that
survives between turns and drives structural plasticity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from bindsnet.network.nodes import Nodes


class DendriticBranchNodes(Nodes):
    def __init__(
        self,
        n: int,
        theta: float = 0.65 / 0.30,
        tc_decay: float = 15.0,
        plateau_ms: int = 3,
        refrac: int = 2,
        calcium_per_plateau: float = 0.06,
        **kwargs,
    ) -> None:
        super().__init__(n=n, **kwargs)
        self.register_buffer("theta", torch.tensor(float(theta)))
        self.register_buffer("tc_decay", torch.tensor(float(tc_decay)))
        self.register_buffer("decay", torch.tensor(0.0))
        self.plateau_ms = int(plateau_ms)
        self.refrac = int(refrac)
        self.calcium_per_plateau = float(calcium_per_plateau)
        self.register_buffer("v", torch.FloatTensor())
        self.register_buffer("plateau_count", torch.FloatTensor())
        self.register_buffer("refrac_count", torch.FloatTensor())
        # Persistent across turns (not cleared by reset_state_variables):
        self.register_buffer("calcium", torch.zeros(n))
        self.register_buffer("active", torch.zeros(n, dtype=torch.bool))
        # Plateau onsets during the current turn (readout):
        self.register_buffer("plateaus", torch.zeros(n))

    def forward(self, x: torch.Tensor) -> None:
        self.v.mul_(self.decay)
        open_ = self.active & (self.plateau_count <= 0) & (self.refrac_count <= 0)
        self.v.add_(x * open_.float())
        onset = open_ & (self.v >= self.theta)
        self.plateau_count.masked_fill_(onset, float(self.plateau_ms))
        self.v.masked_fill_(onset, 0.0)

        self.s = self.plateau_count > 0
        self.plateau_count.sub_(1.0).clamp_(min=0.0)
        ended = self.s & (self.plateau_count <= 0)
        self.refrac_count.sub_(1.0).clamp_(min=0.0)
        self.refrac_count.masked_fill_(ended, float(self.refrac))

        onsets = onset.float().sum(0)
        self.calcium.add_(self.calcium_per_plateau * onsets)
        self.plateaus.add_(onsets)
        super().forward(x)

    def reset_state_variables(self) -> None:
        super().reset_state_variables()
        self.v.zero_()
        self.plateau_count.zero_()
        self.refrac_count.zero_()
        self.plateaus.zero_()

    def compute_decays(self, dt) -> None:
        super().compute_decays(dt=dt)
        self.decay = torch.exp(-self.dt / self.tc_decay)

    def set_batch_size(self, batch_size) -> None:
        super().set_batch_size(batch_size=batch_size)
        self.v = torch.zeros(batch_size, *self.shape, device=self.v.device)
        self.plateau_count = torch.zeros_like(self.v)
        self.refrac_count = torch.zeros_like(self.v)


@dataclass
class BranchMeta:
    """Lineage and provenance of one dendritic branch (the slot's identity)."""

    branch_id: int
    parent_lineage_id: int  # 0 = an original or trauma root
    origin_role: str        # role of the plot this lineage first served: ally | extract | betray | ...
    target_plot: int        # index of the plot ensemble this branch drives now
    source_npc: int = 0     # 0 = first-hand experience; otherwise the numeric id of the gossip source
    volume: float = 0.55    # structural health; pruned below 0.15
    trauma: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BranchMeta":
        return cls(**d)


def poisson_spikes(rates: torch.Tensor, steps: int, f_max_per_ms: float,
                   generator: torch.Generator) -> torch.Tensor:
    """[steps, 1, n] float spike trains; ``rates`` in [0, 1] scale ``f_max_per_ms``."""
    p = (rates.clamp(0.0, 1.0) * f_max_per_ms).expand(steps, 1, -1)
    return (torch.rand(p.shape, generator=generator) < p).float()


def soft_bound(w: torch.Tensor) -> torch.Tensor:
    return 0.2 + w * (1.0 - w)

