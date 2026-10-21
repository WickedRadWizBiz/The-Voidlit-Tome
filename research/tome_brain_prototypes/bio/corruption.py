"""The Tome's corruption projection into an NPC's own BindsNET brain, attached from outside (npc_brain.py untouched,
the same way contagion.py already reaches into brains). A 'puppeteer' afferent layer replays the spike train of the
Tome's L5 ET puppeteer assembly for that NPC during the NPC's 100-ms turn, through fixed, capped synapses onto the
NPC's plots and tactics. Silent schedule == bit-identical NPC.

Gate (run this file):
  1. silent projection: decisions, momentum and learned weights identical to an unattached brain over 15 turns
  2. dose-response: Tome-serving plot momentum and tactic share rise with burst strength, and saturate at the cap
  3. recovery: after the bursts stop, the NPC drifts back toward its own policy (its momentum decays 0.9/turn)
"""
from __future__ import annotations

import sys
import warnings

import torch
from bindsnet.network.nodes import Input
from bindsnet.network.topology import Connection

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/The-Voidlit-Tome/brain")
from voidlit.limbic.npc_brain import NPCBrain  # noqa: E402
from voidlit.senses.state import SensoryState  # noqa: E402
from voidlit.tome import load_scenario  # noqa: E402

N_PUPPET = 24          # the Tome's P.L5_ET puppeteer cells for this NPC
S_MAX = 0.5            # hard cap on corruption strength (fraction of the puppeteer assembly allowed to burst)
W_PLOT, W_ANTI, W_TACTIC = 0.035, -0.02, 0.03


class PuppeteerAfferent(Input):
    """Emits a preloaded [T, n] spike schedule, one row per simulated ms; ignores its input."""

    def __init__(self, n: int):
        super().__init__(n=n, traces=False)
        self.schedule = torch.zeros(0, n)
        self.k = 0

    def forward(self, x: torch.Tensor) -> None:
        row = self.schedule[self.k] if self.k < self.schedule.shape[0] else torch.zeros(self.n)
        self.k += 1
        super().forward(row.view(1, -1).bool())

    def reset_state_variables(self) -> None:
        super().reset_state_variables()
        self.k = 0


def attach(brain: NPCBrain, serve_plot: str, anti_plot: str, serve_tactic: str) -> PuppeteerAfferent:
    spec = brain.spec
    pa = PuppeteerAfferent(N_PUPPET)
    w_p = torch.zeros(N_PUPPET, brain.P)
    w_p[:, spec.plot_index(serve_plot)] = W_PLOT
    w_p[:, spec.plot_index(anti_plot)] = W_ANTI
    w_t = torch.zeros(N_PUPPET, brain.T)
    w_t[:, [t.id for t in spec.tactics].index(serve_tactic)] = W_TACTIC
    brain.net.add_layer(pa, name="tome_puppeteer")
    brain.net.add_connection(Connection(pa, brain.plots, w=w_p), "tome_puppeteer", "plots")
    brain.net.add_connection(Connection(pa, brain.tactics, w=w_t), "tome_puppeteer", "tactics")
    return pa


def burst_schedule(strength: float, gen: torch.Generator, ms: int = 100) -> torch.Tensor:
    """The puppeteer assembly's spikes for one NPC turn: a capped fraction of cells bursting at ~100 Hz."""
    s = max(0.0, min(S_MAX, strength))
    active = torch.rand(N_PUPPET, generator=gen) < s
    return ((torch.rand(ms, N_PUPPET, generator=gen) < 0.1) & active).float()


def sensory(k: int, gen: torch.Generator) -> SensoryState:
    r = torch.rand(6, generator=gen)
    return SensoryState(aggression=0.2 * r[0], deception=0.2 * r[1], leverage=0.3 * r[2], exposure=0.2 * r[3],
                        rapport=0.4 * (r[4] - 0.3), boldness=r[5]).clamped()


def run(strength_by_turn, attach_layer=True, turns=15, seed=4):
    spec = load_scenario().npcs["elara"]
    b = NPCBrain(spec, seed=seed)
    pa = attach(b, "offering", "passage", "misdirect") if attach_layer else None
    g, gs = torch.Generator().manual_seed(99), torch.Generator().manual_seed(7)
    out = []
    for k in range(turns):
        if pa is not None:
            pa.schedule = burst_schedule(strength_by_turn(k), gs)
        d = b.perceive(sensory(k, g))
        out.append(d)
    return b, out


if __name__ == "__main__":
    off = lambda k: 0.0
    b0, d0 = run(off, attach_layer=False)
    b1, d1 = run(off, attach_layer=True)
    same = all(x.to_dict() == y.to_dict() for x, y in zip(d0, d1)) and torch.equal(b0.c_spines.w, b1.c_spines.w) \
        and torch.equal(b0.c_tactics.w, b1.c_tactics.w)
    print("1. silent projection is bit-identical to an unattached NPC brain:", same)
    spec = load_scenario().npcs["elara"]
    off_i, mis = spec.plot_index("offering"), [t.id for t in spec.tactics].index("misdirect")
    print("2. dose-response (15 turns of bursts every turn):")
    for s in (0.0, 0.1, 0.25, 0.5, 1.0):
        b, ds = run(lambda k, s=s: s)
        share = sum(d.tactic == mis for d in ds) / len(ds)
        dom = sum(d.dominant_plot == off_i for d in ds) / len(ds)
        print(f"   strength {s:4.2f}: 'offering' momentum {float(b.momentum[off_i]):.3f}  dominant-plot share "
              f"{dom:.2f}  'misdirect' tactic share {share:.2f}")
    b, ds = run(lambda k: 0.5 if k < 6 else 0.0, turns=20)
    print("3. recovery: bursts on turns 0-5 at the cap, then silent:")
    print("   'offering' momentum by turn:", [round(float(d.momentum[off_i]), 3) for d in ds])
    print("   tactic by turn:", "".join("M" if d.tactic == mis else "." for d in ds), "(M = misdirect)")
