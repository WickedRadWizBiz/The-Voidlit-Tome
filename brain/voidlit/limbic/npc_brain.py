"""Per-NPC limbic brain: a BindsNET network that schemes, learns and holds grudges.

Topology (one network per NPC)::

    senses (Input, 6 channels: aggression, deception, leverage, exposure, rapport+, rapport-)
      | spine synapses: MSTDPET (R-STDP), masked so each branch only has its own spines
      v
    branches (DendriticBranchNodes, fixed slot count; slots are grown, split and pruned)
      | fixed routing: each active branch drives exactly one plot
      v
    plots (LIF ensembles, winner-take-all through recurrent lateral inhibition)
      | plot->tactic synapses: MSTDPET (R-STDP)
      v
    tactics (LIF; the most active one is the NPC's move this turn)

Causal turn order (the reward for turn N-1's tactic only arrives on turn N):

1. The transducer extracts a SensoryState from the player's reply.
2. ``consolidate``: dopamine R and its prediction error R - R_bar multiply the
   eligibility traces left over from the previous turn, *before* simulating.
3. ``simulate``: 100 ms of Poisson-driven spiking (BindsNET ``Network.run``)
   with reward 0, so the traces accumulate but no weights move.
4. Slow plot momentum and structural plasticity: prune, split and trauma-sprout.
5. The decision (dominant plot, subtext plot, tactic, winning branch
   provenance) goes to the prompt compiler.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from bindsnet.learning import MSTDPET
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

from ..senses.state import NPC_CHANNELS, SensoryState
from .dendrites import BranchMeta, DendriticBranchNodes, poisson_spikes, soft_bound

TURN_MS = 100
F_MAX_PER_MS = 0.25        # 250 Hz at full intensity
BRANCH_THETA = 1.20        # plateau threshold, in units of summed spine weight (~2-3 coincident spikes)
PLATEAU_MS = 5
ROUTE_GAIN = 0.60          # per plateau step into the plot; a 5 ms plateau is the super-linear burst
INHIBITION = 0.45          # WTA: a plot spike subtracts this from every rival
TACTIC_THETA = 1.20
MOMENTUM_BIAS = 0.05       # per-ms voltage injected into each plot, times its momentum C_k
ETA = 0.30                 # dopamine learning rate
ELIG_GAIN = 15.0           # squashes the raw MSTDPET trace into roughly [-1, 1]
W_MIN, W_MAX = 0.02, 0.98
PRUNE_VOLUME = 0.15
SPLIT_CALCIUM = 0.85
TRAUMA_SHOCK = 0.85
TRAUMA_COOLDOWN = 3

CH = {name: i for i, name in enumerate(NPC_CHANNELS)}


@dataclass
class PlotSpec:
    id: str
    name: str
    role: str = "other"  # ally | extract | betray | other
    momentum: float = 0.2


@dataclass
class TacticSpec:
    id: str
    directive: str
    plot: str = ""        # plot this tactic naturally serves (initial weight bias)
    lines: list[str] = field(default_factory=list)  # authored lines for the offline voice


@dataclass
class BranchSeed:
    plot: str
    channels: list[str]
    weight: float = 0.55


@dataclass
class NPCSpec:
    id: str
    name: str
    numeric_id: int
    persona: str = ""
    color: str = "#9fb4ff"
    aliases: list[str] = field(default_factory=list)
    plots: list[PlotSpec] = field(default_factory=list)
    tactics: list[TacticSpec] = field(default_factory=list)
    branches: list[BranchSeed] = field(default_factory=list)
    reward: dict[str, float] = field(default_factory=lambda: {"rapport": 0.60, "deception": -0.50, "exposure": -0.40})
    trust: dict[str, float] = field(default_factory=dict)

    def plot_index(self, ref: str) -> int:
        """Look a plot up by id first, then by role."""
        for i, p in enumerate(self.plots):
            if p.id == ref:
                return i
        for i, p in enumerate(self.plots):
            if p.role == ref:
                return i
        raise KeyError(f"{self.id}: no plot with id or role {ref!r}")

    def role_index(self, role: str) -> int | None:
        for i, p in enumerate(self.plots):
            if p.role == role:
                return i
        return None


@dataclass
class BrainDecision:
    dominant_plot: int
    secondary_plot: int
    tactic: int
    branch_id: int
    branch_origin: str
    branch_source_npc: int
    branch_parent: int
    plot_spikes: list[int]
    tactic_spikes: list[int]
    momentum: list[float]
    reward: float | None = None
    rpe: float | None = None
    events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class NPCBrain:
    def __init__(self, spec: NPCSpec, seed: int = 0, max_branches: int = 32) -> None:
        if not spec.plots or not spec.tactics:
            raise ValueError(f"{spec.id}: an NPC needs at least one plot and one tactic")
        self.spec = spec
        self.P, self.T, self.C, self.B = len(spec.plots), len(spec.tactics), len(NPC_CHANNELS), max_branches
        self.gen = torch.Generator().manual_seed(seed)

        net = Network(dt=1.0, batch_size=1, learning=True)
        self.senses = Input(n=self.C)
        self.branches = DendriticBranchNodes(n=self.B, theta=BRANCH_THETA, plateau_ms=PLATEAU_MS,
                                             calcium_per_plateau=0.05)
        self.plots = LIFNodes(n=self.P, thresh=1.0, rest=0.0, reset=0.0, refrac=1, tc_decay=15.0, lbound=-1.0)
        self.tactics = LIFNodes(n=self.T, thresh=TACTIC_THETA, rest=0.0, reset=0.0, refrac=2, tc_decay=15.0,
                                lbound=-1.0)
        for name, layer in (("senses", self.senses), ("branches", self.branches),
                            ("plots", self.plots), ("tactics", self.tactics)):
            net.add_layer(layer, name=name)

        rule_kw = dict(update_rule=MSTDPET, nu=1.0, tc_plus=20.0, tc_minus=20.0, tc_e_trace=float(TURN_MS))
        self.c_spines = Connection(self.senses, self.branches, w=torch.zeros(self.C, self.B),
                                   wmin=0.0, wmax=1.0, **rule_kw)
        self.c_route = Connection(self.branches, self.plots, w=torch.zeros(self.B, self.P))
        self.c_inhib = Connection(self.plots, self.plots, w=-INHIBITION * (1.0 - torch.eye(self.P)))
        w_t = torch.full((self.P, self.T), 0.25)
        for t, tac in enumerate(spec.tactics):
            if tac.plot:
                w_t[spec.plot_index(tac.plot), t] = 0.75
        self.c_tactics = Connection(self.plots, self.tactics, w=w_t, wmin=0.0, wmax=1.0, **rule_kw)
        net.add_connection(self.c_spines, "senses", "branches")
        net.add_connection(self.c_route, "branches", "plots")
        net.add_connection(self.c_inhib, "plots", "plots")
        net.add_connection(self.c_tactics, "plots", "tactics")

        self.m_plots = Monitor(self.plots, state_vars=("s",), time=TURN_MS)
        self.m_tactics = Monitor(self.tactics, state_vars=("s",), time=TURN_MS)
        net.add_monitor(self.m_plots, "plots")
        net.add_monitor(self.m_tactics, "tactics")
        self.net = net

        self.spine_mask = torch.zeros(self.C, self.B, dtype=torch.bool)
        self.meta: list[BranchMeta | None] = [None] * self.B
        self.momentum = torch.tensor([p.momentum for p in spec.plots], dtype=torch.float)
        self.reward_baseline = 0.0
        self.next_branch_id = 1
        self.turn = 0
        self.last_trauma_turn = -TRAUMA_COOLDOWN
        self.last_decision: BrainDecision | None = None

        for seed_b in spec.branches:
            plot = spec.plot_index(seed_b.plot)
            self.spawn_branch(plot, [CH[c] for c in seed_b.channels], seed_b.weight,
                              origin_role=spec.plots[plot].role)

    # ------------------------------------------------------------------ slots
    def active_slots(self) -> list[int]:
        return [i for i, m in enumerate(self.meta) if m is not None]

    def _free_slot(self) -> int | None:
        for i, m in enumerate(self.meta):
            if m is None:
                return i
        return None

    def _weakest_slot(self) -> int:
        slots = self.active_slots()
        return min(slots, key=lambda i: self.meta[i].volume)

    def spawn_branch(self, target_plot: int, channels: list[int], weight: float, *, origin_role: str,
                     parent: int = 0, source_npc: int = 0, volume: float = 0.55, calcium: float = 0.0,
                     trauma: bool = False, evict: bool = False) -> int | None:
        slot = self._free_slot()
        if slot is None:
            if not evict:
                return None
            slot = self._weakest_slot()
            self._clear_slot(slot)
        meta = BranchMeta(branch_id=self.next_branch_id, parent_lineage_id=parent, origin_role=origin_role,
                          target_plot=target_plot, source_npc=source_npc, volume=volume, trauma=trauma)
        self.next_branch_id += 1
        self.meta[slot] = meta
        self.spine_mask[:, slot] = False
        self.spine_mask[channels, slot] = True
        with torch.no_grad():
            self.c_spines.w[:, slot] = 0.0
            self.c_spines.w[channels, slot] = float(min(W_MAX, max(W_MIN, weight)))
        self._route(slot, target_plot)
        self.branches.active[slot] = True
        self.branches.calcium[slot] = calcium
        self._zero_eligibility_column(slot)
        return slot

    def _route(self, slot: int, plot: int) -> None:
        with torch.no_grad():
            self.c_route.w[slot, :] = 0.0
            self.c_route.w[slot, plot] = ROUTE_GAIN

    def _clear_slot(self, slot: int) -> None:
        self.meta[slot] = None
        self.spine_mask[:, slot] = False
        with torch.no_grad():
            self.c_spines.w[:, slot] = 0.0
            self.c_route.w[slot, :] = 0.0
        self.branches.active[slot] = False
        self.branches.calcium[slot] = 0.0
        self._zero_eligibility_column(slot)

    def _zero_eligibility_column(self, slot: int) -> None:
        rule = self.c_spines.update_rule
        if hasattr(rule, "eligibility_trace"):
            rule.eligibility_trace[..., slot] = 0.0

    def slot_of(self, branch_id: int) -> int | None:
        for i, m in enumerate(self.meta):
            if m is not None and m.branch_id == branch_id:
                return i
        return None

    # ------------------------------------------------------------- the turn
    def perceive(self, s: SensoryState) -> BrainDecision:
        """Run the full causal turn: consolidate, simulate, restructure, decide."""
        reward = rpe = None
        if self.turn > 0:
            reward, rpe = self.consolidate(s)
        plot_spikes, tactic_spikes, plateaus = self.simulate(s)
        self.momentum = (0.90 * self.momentum + 0.25 * 0.05 * plot_spikes.float()).clamp(0.0, 1.0)
        events = self.restructure(s)
        decision = self._decide(plot_spikes, tactic_spikes, plateaus, events)
        decision.reward, decision.rpe = reward, rpe
        self.turn += 1
        self.last_decision = decision
        return decision

    def reward_of(self, s: SensoryState) -> float:
        r = self.spec.reward
        raw = (r.get("rapport", 0.0) * s.rapport + r.get("deception", 0.0) * s.deception
               + r.get("exposure", 0.0) * s.exposure + r.get("leverage", 0.0) * s.leverage
               + r.get("aggression", 0.0) * s.aggression)
        return float(max(-1.0, min(1.0, raw)))

    def consolidate(self, s: SensoryState) -> tuple[float, float]:
        """Stage 2: retroactive dopamine. Turn N's reaction pays out turn N-1's traces."""
        reward = self.reward_of(s)
        rpe = reward - self.reward_baseline
        self.reward_baseline = 0.85 * self.reward_baseline + 0.15 * reward
        for conn, mask in ((self.c_spines, self.spine_mask), (self.c_tactics, None)):
            rule = conn.update_rule
            if not hasattr(rule, "eligibility_trace"):
                continue
            e = torch.tanh(ELIG_GAIN * rule.eligibility_trace[0])
            with torch.no_grad():
                w = conn.w
                dw = ETA * rpe * e * soft_bound(w)
                if mask is not None:
                    dw = dw * mask.float()
                w += dw
                if mask is not None:
                    w.copy_(torch.where(mask, w.clamp(W_MIN, W_MAX), torch.zeros_like(w)))
                else:
                    w.clamp_(W_MIN, W_MAX)
            rule.eligibility_trace.mul_(0.5)
        return reward, rpe

    def simulate(self, s: SensoryState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stage 3: 100 ms of spiking. Returns plot, tactic and branch-plateau counts."""
        self.net.reset_state_variables()
        for conn in (self.c_spines, self.c_tactics):
            rule = conn.update_rule
            for name in ("p_plus", "p_minus", "eligibility"):
                if hasattr(rule, name):
                    getattr(rule, name).zero_()
        rates = torch.tensor(s.npc_channels(), dtype=torch.float)
        spikes = poisson_spikes(rates, TURN_MS, F_MAX_PER_MS, self.gen)
        self.net.run(
            inputs={"senses": spikes}, time=TURN_MS, one_step=True, reward=0.0,
            masks={("senses", "branches"): ~self.spine_mask},
            injects_v={"plots": MOMENTUM_BIAS * self.momentum},
        )
        plot_spikes = self.m_plots.get("s").float().sum(dim=(0, 1))
        tactic_spikes = self.m_tactics.get("s").float().sum(dim=(0, 1))
        return plot_spikes, tactic_spikes, self.branches.plateaus.clone()

    def restructure(self, s: SensoryState) -> list[str]:
        """Stage 4: grow, split and prune branches from calcium and weight."""
        events: list[str] = []
        calcium = self.branches.calcium
        betray = self.spec.role_index("betray")
        hostile = s.aggression > 0.7 or s.exposure > 0.7
        for slot in self.active_slots():
            meta = self.meta[slot]
            cols = self.spine_mask[:, slot]
            mean_w = float(self.c_spines.w[cols, slot].mean()) if cols.any() else 0.0
            ca = float(calcium[slot])
            meta.volume = min(1.0, max(0.0, 0.92 * meta.volume + 0.20 * ca + 0.08 * mean_w))
            if meta.volume < PRUNE_VOLUME:
                events.append(f"pruned branch {meta.branch_id}")
                self._clear_slot(slot)
                continue
            if ca > SPLIT_CALCIUM:
                target = betray if (hostile and betray is not None) else meta.target_plot
                channels = cols.nonzero().flatten().tolist()
                weight = float(self.c_spines.w[cols, slot].mean())
                child = self.spawn_branch(target, channels, weight, origin_role=meta.origin_role,
                                          parent=meta.branch_id, source_npc=meta.source_npc, volume=0.45)
                if child is not None:
                    with torch.no_grad():
                        self.c_spines.w[:, child] = self.c_spines.w[:, slot]
                    turned = " toward betrayal" if target != meta.target_plot else ""
                    events.append(f"branch {meta.branch_id} split into {self.meta[child].branch_id}{turned}")
                    calcium[slot] *= 0.4
            calcium[slot] *= 0.5
        if (s.shock > TRAUMA_SHOCK and betray is not None
                and self.turn - self.last_trauma_turn >= TRAUMA_COOLDOWN):
            slot = self.spawn_branch(betray, [CH["aggression"], CH["exposure"], CH["rapport_neg"]], 0.70,
                                     origin_role="betray", trauma=True, evict=True)
            if slot is not None:
                self.last_trauma_turn = self.turn
                events.append(f"trauma sprouted branch {self.meta[slot].branch_id}")
        return events

    def _decide(self, plot_spikes: torch.Tensor, tactic_spikes: torch.Tensor, plateaus: torch.Tensor,
                events: list[str]) -> BrainDecision:
        order = torch.argsort(self.momentum, descending=True).tolist()
        dominant = order[0]
        secondary = order[1] if len(order) > 1 else order[0]
        if float(tactic_spikes.max()) > 0:
            tactic = int(torch.argmax(tactic_spikes))
        else:
            tactic = int(torch.argmax(self.c_tactics.w[dominant]))

        best_slot, best_key = None, None
        for slot in self.active_slots():
            m = self.meta[slot]
            key = (m.target_plot == dominant, float(plateaus[slot]), float(self.branches.calcium[slot]), m.volume)
            if best_key is None or key > best_key:
                best_slot, best_key = slot, key
        m = self.meta[best_slot] if best_slot is not None else None
        return BrainDecision(
            dominant_plot=dominant, secondary_plot=secondary, tactic=tactic,
            branch_id=m.branch_id if m else 0, branch_origin=m.origin_role if m else "other",
            branch_source_npc=m.source_npc if m else 0, branch_parent=m.parent_lineage_id if m else 0,
            plot_spikes=[int(x) for x in plot_spikes.tolist()],
            tactic_spikes=[int(x) for x in tactic_spikes.tolist()],
            momentum=[round(float(x), 4) for x in self.momentum.tolist()], events=events,
        )

    # ----------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        def trace(conn):
            rule = conn.update_rule
            return rule.eligibility_trace.clone() if hasattr(rule, "eligibility_trace") else None

        return {
            "version": 1, "npc_id": self.spec.id, "B": self.B, "P": self.P, "T": self.T,
            "w_spines": self.c_spines.w.detach().clone(), "w_tactics": self.c_tactics.w.detach().clone(),
            "w_route": self.c_route.w.detach().clone(), "spine_mask": self.spine_mask.clone(),
            "e_spines": trace(self.c_spines), "e_tactics": trace(self.c_tactics),
            "calcium": self.branches.calcium.clone(), "momentum": self.momentum.clone(),
            "meta": [m.to_dict() if m else None for m in self.meta],
            "next_branch_id": self.next_branch_id, "reward_baseline": self.reward_baseline,
            "turn": self.turn, "last_trauma_turn": self.last_trauma_turn,
            "gen_state": self.gen.get_state(),
        }

    def load_state_dict(self, d: dict) -> None:
        if d.get("version") != 1 or d.get("npc_id") != self.spec.id:
            raise ValueError(f"brain state for {d.get('npc_id')!r} v{d.get('version')} does not fit {self.spec.id!r}")
        if (d["B"], d["P"], d["T"]) != (self.B, self.P, self.T):
            raise ValueError("brain state shape does not match this NPC's spec")
        with torch.no_grad():
            self.c_spines.w.copy_(d["w_spines"])
            self.c_tactics.w.copy_(d["w_tactics"])
            self.c_route.w.copy_(d["w_route"])
        self.spine_mask = d["spine_mask"].clone()
        for conn, key in ((self.c_spines, "e_spines"), (self.c_tactics, "e_tactics")):
            if d[key] is not None:
                conn.update_rule.eligibility_trace = d[key].clone()
        self.branches.calcium.copy_(d["calcium"])
        self.momentum = d["momentum"].clone()
        self.meta = [BranchMeta.from_dict(m) if m else None for m in d["meta"]]
        self.branches.active.copy_(torch.tensor([m is not None for m in self.meta]))
        self.next_branch_id = d["next_branch_id"]
        self.reward_baseline = d["reward_baseline"]
        self.turn = d["turn"]
        self.last_trauma_turn = d["last_trauma_turn"]
        self.gen.set_state(d["gen_state"])

    def save(self, path) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path) -> None:
        self.load_state_dict(torch.load(path, weights_only=False))
