"""The Director: a SpikingJelly network that paces the TOME.

The deck decides what happens (TOME: Core Rules 0.2). The Director only decides how the TOME presents it: which
way the TOME asks to be touched when a Leaf is ready (the dot, the shapes or the dark), and what narrative beat
follows a moment (quiet, a foreshadowing, a revealed truth, or a person acting on their own). It never rolls a
die, changes the deck or touches a number.

Architecture (multi-step mode, T timesteps per decision)::

    Poisson spikes [T, N, F]
      -> Linear(F, H) -> LIF (ATan surrogate)
      -> Linear(H, H) -> LIF (ATan surrogate)
      -> Linear(H, n_beats), averaged over T  (a non-leaky integrator readout)

It is trained offline with surrogate gradients (BPTT) to match the pacing
teacher's beat distribution, then frozen. At the table it reads the current
state, and the engine samples a beat from its output after masking out the
beats the rules forbid right now.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import functional, layer, neuron, surrogate

from .features import BEATS, N_FEATURES, Beat, DirectorFeatures
from .teacher import sample_states, teacher_distribution

DEFAULT_WEIGHTS = Path(__file__).with_name("director.pt")
TIMESTEPS = 16
HIDDEN = 64
WEIGHTS_VERSION = 2


class DirectorSNN(torch.nn.Module):
    def __init__(self, n_in: int = N_FEATURES, hidden: int = HIDDEN, n_out: int = len(BEATS),
                 timesteps: int = TIMESTEPS) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.net = torch.nn.Sequential(
            layer.Linear(n_in, hidden),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.Linear(hidden, hidden),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.Linear(hidden, n_out),
        )
        functional.set_step_mode(self.net, "m")

    def encode(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Poisson-encode [N, F] features in [0, 1] into [T, N, F] spikes."""
        x = x.clamp(0.0, 1.0)
        r = torch.rand((self.timesteps, *x.shape), generator=generator, device=x.device)
        return (r < x.unsqueeze(0)).float()

    def forward(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        functional.reset_net(self.net)
        out = self.net(self.encode(x, generator)).mean(0)
        functional.reset_net(self.net)
        return out


def train_director(steps: int = 800, batch: int = 512, lr: float = 3e-3, seed: int = 7,
                   log_every: int = 0) -> tuple[DirectorSNN, list[float]]:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = DirectorSNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    losses: list[float] = []
    for step in range(steps):
        x = sample_states(batch, gen)
        target = teacher_distribution(x)
        logits = model(x, gen)
        loss = F.kl_div(F.log_softmax(logits, dim=1), target, reduction="batchmean")
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss))
        if log_every and step % log_every == 0:
            print(f"step {step:4d}  KL {float(loss):.4f}")
    model.eval()
    return model, losses


def save_director(model: DirectorSNN, path: Path = DEFAULT_WEIGHTS) -> None:
    torch.save({"version": WEIGHTS_VERSION, "n_in": N_FEATURES, "beats": [b.value for b in BEATS],
                "state": model.state_dict()}, path)


def load_director(path: Path = DEFAULT_WEIGHTS) -> DirectorSNN | None:
    try:
        blob = torch.load(path, weights_only=True)
    except (FileNotFoundError, RuntimeError, OSError):
        return None
    if (blob.get("version") != WEIGHTS_VERSION or blob.get("n_in") != N_FEATURES
            or blob.get("beats") != [b.value for b in BEATS]):
        return None
    model = DirectorSNN()
    model.load_state_dict(blob["state"])
    model.eval()
    return model


class Director:
    """Runtime wrapper: features in, one beat out (seeded and reproducible)."""

    def __init__(self, model: DirectorSNN | None = None, seed: int = 0, temperature: float = 1.0) -> None:
        if model is None:
            model = load_director()
        if model is None:
            model, _ = train_director(steps=300)
        self.model = model
        self.gen = torch.Generator().manual_seed(seed)
        self.temperature = temperature

    @torch.no_grad()
    def probabilities(self, features: DirectorFeatures) -> torch.Tensor:
        logits = self.model(features.tensor().unsqueeze(0), self.gen)[0]
        return torch.softmax(logits / self.temperature, dim=0)

    def choose(self, features: DirectorFeatures, rng: random.Random,
               allowed: set[Beat] | None = None) -> tuple[Beat, dict[str, float]]:
        probs = self.probabilities(features)
        if allowed is not None:
            mask = torch.tensor([b in allowed for b in BEATS], dtype=torch.float)
            if mask.sum() > 0:
                probs = probs * mask
        probs = probs / probs.sum()
        r, acc = rng.random(), 0.0
        choice = BEATS[-1]
        for beat, p in zip(BEATS, probs.tolist()):
            acc += p
            if r <= acc:
                choice = beat
                break
        return choice, {b.value: round(p, 4) for b, p in zip(BEATS, probs.tolist())}
