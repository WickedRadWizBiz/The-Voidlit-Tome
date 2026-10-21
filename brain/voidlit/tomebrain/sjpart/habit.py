"""The habit cortex: the trained Director runs exactly as trained and becomes the network's pacing prior.

At the start of Phase B the unchanged ``DirectorSNN`` is reset, fed 16 Poisson-encoded steps of the 18 Director
features (drawn from the SpikingJelly Philox stream, so the draw is keyed by tick) and reset again. Its beat
probabilities p set the drive of 32 habit LIF cells (4 per beat): each fires regularly at a rate proportional to
(log p_b - log p_max + 4)+ for the rest of Phase B. They reach the PACE D1 populations through fixed synapses.
"""

from __future__ import annotations

import numpy as np
import torch
from spikingjelly.activation_based import functional

from ...cortex.director import DirectorSNN, load_director
from ...cortex.features import BEATS
from ..rng import Streams

PER_BEAT = 4
TAU = 5.0
RATE_MAX = 0.1          # spikes per ms at the top of the scale (100 Hz)
SCALE = 4.0             # log-probability span mapped onto the rate (Gumbel units)


def rate_for(logp: np.ndarray) -> np.ndarray:
    """Habit firing rate (spikes/ms) per beat: proportional to (log p_b - log p_max + 4)+."""
    return RATE_MAX * np.clip(logp - logp.max() + SCALE, 0.0, None) / SCALE


def current_for(rate: np.ndarray, tau: float = TAU, theta: float = 1.0) -> np.ndarray:
    """Constant input that makes a discrete LIF (v <- v(1 - 1/tau) + I, threshold theta) fire at ``rate``."""
    rate = np.asarray(rate, float)
    out = np.zeros_like(rate)
    on = rate > 1e-4
    period = 1.0 / rate[on]
    a = 1.0 - 1.0 / tau
    # after n steps from 0: v = I (1 - a^n) / (1 - a); fire when it reaches theta at n = period
    out[on] = theta * (1.0 - a) / (1.0 - a ** period) * 1.0001
    return out


class Habit:
    def __init__(self, model: DirectorSNN | None, streams: Streams, temperature: float = 1.0) -> None:
        self.model = model or load_director()
        if self.model is None:
            raise RuntimeError("cortex/director.pt is missing; run `python -m voidlit train-director`")
        self.model.eval()
        self.streams = streams
        self.temperature = temperature
        self.v = torch.zeros(len(BEATS) * PER_BEAT)
        self.drive = torch.zeros(len(BEATS) * PER_BEAT)
        self.p = np.full(len(BEATS), 1.0 / len(BEATS))

    @torch.no_grad()
    def prime(self, features: np.ndarray, tick: int) -> np.ndarray:
        """Run the Director on 16 clock ms of Poisson spikes starting at ``tick``. Returns p over the beats."""
        x = torch.as_tensor(features, dtype=torch.float32).clamp(0, 1)[None, :]
        T = self.model.timesteps
        r = torch.from_numpy(np.stack([self.streams.uniform("habit", tick + i, x.shape[1]) for i in range(T)]))
        spikes = (r[:, None, :] < x[None]).float()
        functional.reset_net(self.model.net)
        logits = self.model.net(spikes).mean(0)[0]
        functional.reset_net(self.model.net)
        p = torch.softmax(logits / self.temperature, 0).numpy().astype(np.float64)
        self.p = p
        rates = rate_for(np.log(p + 1e-9))
        self.drive = torch.from_numpy(np.repeat(current_for(rates), PER_BEAT).astype(np.float32))
        # the 4 cells of a beat start a quarter-cycle apart, so their summed output is smooth, not a sawtooth
        self.v = torch.from_numpy(np.tile(np.arange(PER_BEAT) / PER_BEAT, len(BEATS)).astype(np.float32)) \
            * (self.drive > 0)
        return p

    def quiet(self) -> None:
        self.drive = torch.zeros_like(self.drive)

    def step(self) -> torch.Tensor:
        self.v = self.v * (1.0 - 1.0 / TAU) + self.drive
        s = (self.v >= 1.0).float()
        self.v = self.v * (1.0 - s)
        return s

    def state_dict(self) -> dict:
        return {"v": self.v.numpy().copy(), "drive": self.drive.numpy().copy(), "p": self.p.copy()}

    def load_state_dict(self, d: dict) -> None:
        self.v = torch.from_numpy(np.array(d["v"], np.float32))
        self.drive = torch.from_numpy(np.array(d["drive"], np.float32))
        self.p = np.array(d["p"], np.float64)


def expected_spikes(p: np.ndarray, ms: int) -> np.ndarray:
    """Spikes per beat from its 4 habit cells over ``ms`` (for tests)."""
    r = rate_for(np.log(np.asarray(p) + 1e-9))
    return PER_BEAT * np.floor(ms * r + 1e-9)
