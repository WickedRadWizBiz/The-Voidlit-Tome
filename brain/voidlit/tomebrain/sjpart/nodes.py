"""A SpikingJelly LIF node with per-neuron time constants, thresholds and bias (one fused population)."""

from __future__ import annotations

import torch
from spikingjelly.activation_based import neuron


class PerNeuronLIF(neuron.BaseNode):
    """v <- v (1 - 1/tau) + I + bias; spike at v >= threshold, then v <- 0. ``tau`` and ``threshold`` are
    tensors, so every population of the basal ganglia shares one node and one recurrent matrix."""

    def __init__(self, tau: torch.Tensor, threshold: torch.Tensor | float = 1.0, bias: torch.Tensor | float = 0.0):
        super().__init__(v_threshold=1.0, v_reset=0.0, detach_reset=True, step_mode="s")
        self.register_buffer("decay", 1.0 - 1.0 / torch.as_tensor(tau, dtype=torch.float32))
        self.register_buffer("theta", torch.as_tensor(threshold, dtype=torch.float32).expand_as(self.decay).clone())
        self.register_buffer("bias", torch.as_tensor(bias, dtype=torch.float32).expand_as(self.decay).clone())
        self.v = torch.zeros_like(self.decay)

    def neuronal_charge(self, x: torch.Tensor) -> None:
        self.v = self.v * self.decay + x + self.bias

    def neuronal_fire(self) -> torch.Tensor:
        return (self.v >= self.theta).float()

    def neuronal_reset(self, spike: torch.Tensor) -> None:
        self.v = self.v * (1.0 - spike)

    def extra_repr(self) -> str:
        return f"n={self.decay.numel()}"
