"""The BindsNET partition: the Tome's amygdala, where its opinion of the player lives.

One BindsNET ``Network`` (dt 1 ms):

    ctx  Input 128   cortical L5 IT (S, T, P, I), the T liar readout, CA1  (bus, 6 ms)
    us   Input 24    6 unconditioned stimuli x 4 Poisson cells (from the rules)
    BLA  LIF 48      4 assemblies of 12: bond, threat (from the player), contempt, loss (NPCs moving on the Tome)
    ITC  LIF 8       intercalated cells, gating the central amygdala
    CeA  LIF 16      central amygdala: fear and rage out to LC, PVN and RMTg

ctx -> BLA learns with BindsNET's MSTDPET (Florian 2007). The teaching signal is a vector over BLA cells: each
assembly is taught by its own stimuli, r_k = US_k (1 + 0.5 DA_phasic). **The learned ctx -> BLA weights are the
Tome's opinion of the player**: slow, persistent and saved, and never shown anywhere.
"""

from __future__ import annotations

import numpy as np
import torch
from bindsnet.learning import MSTDPET
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

from ..partition import Partition, check_shapes, check_streams, shapes_of
from ..rng import Streams

US_NAMES = ("harm", "defiance", "gift", "loss", "bearer_near_death", "lie")
ASSEMBLY = ("bond", "threat", "contempt", "loss")
PER_US, PER_ASM = 4, 12
N_CTX, N_US, N_BLA, N_ITC, N_CEA = 128, len(US_NAMES) * PER_US, len(ASSEMBLY) * PER_ASM, 8, 16
# which assembly each stimulus teaches (and drives)
US_TO_BLA = {
    "gift": {"bond": 1.0},
    "harm": {"threat": 1.0},
    "defiance": {"threat": 0.5, "contempt": 0.7},
    "lie": {"contempt": 1.0},
    "loss": {"loss": 1.0},
    "bearer_near_death": {"loss": 0.6, "threat": 0.3},
}
US_RATE = 0.1              # spikes/ms per US cell at full intensity (100 Hz)
NU = 2e-3
TONIC_TAU_MS = 6_000.0
QUANTUM = 4                # ms: the teaching signal is constant within each 4-ms quantum
RULE_STATE = ("p_plus", "p_minus", "eligibility", "eligibility_trace")   # MSTDPET's traces, made at its first step


class BNPartition(Partition):
    name = "bn"

    def __init__(self, run_seed: int, anatomy_seed: int = 0) -> None:
        torch.set_num_threads(1)
        self.streams = Streams(run_seed, "bn")
        g = torch.Generator().manual_seed(2000 + anatomy_seed)
        net = Network(dt=1.0)
        lif = dict(rest=-65.0, reset=-65.0, refrac=2, traces=True)
        net.add_layer(Input(n=N_CTX, traces=True), name="ctx")
        net.add_layer(Input(n=N_US, traces=True), name="us")
        net.add_layer(LIFNodes(n=N_BLA, thresh=-52.0, tc_decay=20.0, **lif), name="bla")
        net.add_layer(LIFNodes(n=N_ITC, thresh=-52.0, tc_decay=10.0, **lif), name="itc")
        net.add_layer(LIFNodes(n=N_CEA, thresh=-52.0, tc_decay=15.0, **lif), name="cea")
        L = net.layers
        net.add_connection(Connection(L["ctx"], L["bla"], w=torch.rand(N_CTX, N_BLA, generator=g) * 0.3,
                                      update_rule=MSTDPET, nu=NU, wmin=0.0, wmax=1.5), "ctx", "bla")
        w_us = torch.zeros(N_US, N_BLA)
        for u, name in enumerate(US_NAMES):
            for asm, k in US_TO_BLA[name].items():
                a = ASSEMBLY.index(asm)
                w_us[u * PER_US:(u + 1) * PER_US, a * PER_ASM:(a + 1) * PER_ASM] = 3.0 * k
        net.add_connection(Connection(L["us"], L["bla"], w=w_us), "us", "bla")
        net.add_connection(Connection(L["bla"], L["cea"], w=(torch.rand(N_BLA, N_CEA, generator=g) < 0.3).float() * 3.0),
                           "bla", "cea")
        net.add_connection(Connection(L["bla"], L["itc"], w=(torch.rand(N_BLA, N_ITC, generator=g) < 0.5).float() * 3.0),
                           "bla", "itc")
        net.add_connection(Connection(L["itc"], L["cea"], w=torch.full((N_ITC, N_CEA), -2.0)), "itc", "cea")
        mutual = torch.zeros(N_BLA, N_BLA)
        bond = slice(0, PER_ASM)
        for rival in ("threat", "contempt"):
            r = ASSEMBLY.index(rival)
            mutual[bond, r * PER_ASM:(r + 1) * PER_ASM] = -1.5
            mutual[r * PER_ASM:(r + 1) * PER_ASM, bond] = -1.5
        net.add_connection(Connection(L["bla"], L["bla"], w=mutual), "bla", "bla")
        self.net = net
        self._mon = {}
        for layer in ("bla", "cea"):
            self._mon[layer] = Monitor(net.layers[layer], state_vars=("s",), time=QUANTUM)
            net.add_monitor(self._mon[layer], name=f"mon_{layer}")
        self._reward = torch.zeros(N_BLA)
        self.us = np.zeros(len(US_NAMES), np.float32)       # stimulus intensities this turn, 0..1
        self.da_tonic, self.nm_ticks = 0.0, 0
        self.teach = torch.zeros(len(ASSEMBLY), len(US_NAMES))
        for u, name in enumerate(US_NAMES):
            for asm, k in US_TO_BLA[name].items():
                self.teach[ASSEMBLY.index(asm), u] = k
        self.outputs = {"B.bla": N_BLA, "B.cea": N_CEA}

    def set_us(self, intensities: dict[str, float]) -> None:
        self.us = np.array([float(np.clip(intensities.get(n, 0.0), 0.0, 1.0)) for n in US_NAMES], np.float32)

    def step(self, bus, t0: int, W: int) -> dict[str, np.ndarray]:
        ctx = torch.from_numpy(bus.read("J.amy", t0, W, 6))
        nm = bus.read("J.nm", t0, W, 6)
        out_bla = np.zeros((W, N_BLA), np.float32)
        out_cea = np.zeros((W, N_CEA), np.float32)
        rates = np.repeat(self.us, PER_US) * US_RATE
        us_k = self.teach @ torch.from_numpy(self.us)                          # per assembly
        us = torch.from_numpy(np.stack([(self.streams.uniform("us", t0 + i, N_US) < rates).astype(np.float32)
                                        for i in range(W)]))
        # The teaching signal changes only on 4-ms quantum boundaries (dopamine sampled at the quantum's first
        # tick), so BindsNET can run each quantum's ms in one call and 1-ms and 4-ms windows learn identically.
        i = 0
        while i < W:
            tick = t0 + i
            if tick % QUANTUM == 0:
                self.nm_ticks += 1
                k = 1.0 / min(self.nm_ticks, TONIC_TAU_MS / QUANTUM)
                self.da_tonic += (float(nm[i, 0]) - self.da_tonic) * k
                da = float(nm[i, 0]) / max(self.da_tonic, 1e-3) - 1.0 if self.nm_ticks > 125 else 0.0
                self._reward = (us_k * (1.0 + 0.5 * max(-0.5, min(1.0, da)))).repeat_interleave(PER_ASM)
            n = min(W - i, QUANTUM - tick % QUANTUM)
            self._run_block(ctx[i:i + n], us[i:i + n], self._reward, out_bla[i:i + n], out_cea[i:i + n])
            i += n
        return {"B.bla": out_bla, "B.cea": out_cea}

    def _run_block(self, ctx, us, reward, out_bla, out_cea) -> None:
        """Run n ms in one BindsNET call and read back each ms's spikes from monitors."""
        n = ctx.shape[0]
        for layer in ("bla", "cea"):
            self._mon[layer].reset_state_variables()
        self.net.run(inputs={"ctx": ctx, "us": us}, time=n, reward=reward)
        out_bla[:] = self._mon["bla"].get("s").view(n, -1).float().numpy()
        out_cea[:] = self._mon["cea"].get("s").view(n, -1).float().numpy()

    # ------------------------------------------------------------------ persistence
    def _tensors(self) -> dict[str, torch.Tensor]:
        out = {}
        for name, layer in self.net.layers.items():
            for attr in ("v", "refrac_count", "s", "x"):
                if hasattr(layer, attr) and isinstance(getattr(layer, attr), torch.Tensor):
                    out[f"layer.{name}.{attr}"] = getattr(layer, attr)
        for (a, b), conn in self.net.connections.items():
            out[f"conn.{a}.{b}.w"] = conn.w
            rule = conn.update_rule
            for attr in RULE_STATE:
                if hasattr(rule, attr):
                    out[f"rule.{a}.{b}.{attr}"] = getattr(rule, attr)
        return out

    def state_dict(self) -> dict:
        t = {k: v.detach().cpu().numpy().copy() for k, v in self._tensors().items()}
        return {"tensors": t, "streams": self.streams.state_dict(), "us": self.us.copy(),
                "reward": self._reward.numpy().copy(),
                "scalars": np.array([self.da_tonic, self.nm_ticks], np.float64)}

    def _rule_shapes(self) -> dict[str, tuple[int, ...]]:
        """The shape of every learning-rule trace this network has or will make at its first step."""
        out = {}
        for (a, b), conn in self.net.connections.items():
            if isinstance(conn.update_rule, MSTDPET):
                n = self.net.batch_size
                shape = {"p_plus": (n, conn.source.n), "p_minus": (n, conn.target.n),
                         "eligibility": (n, *conn.w.shape), "eligibility_trace": (n, *conn.w.shape)}
                out.update({f"rule.{a}.{b}.{attr}": shape[attr] for attr in RULE_STATE})
        return out

    def check_state_dict(self, d: dict) -> None:
        """Every array this network holds, at its shape. A learning rule's traces may be absent (a save from before
        its first step), but only all together."""
        rules = self._rule_shapes()
        want = shapes_of(self.state_dict())
        want["tensors"] = {k: s for k, s in want["tensors"].items() if k not in rules}
        saved = d.get("tensors") if isinstance(d, dict) else None
        saved = saved if isinstance(saved, dict) else {}        # (check_shapes refuses a missing one)
        for rule in dict.fromkeys(k.rsplit(".", 1)[0] for k in rules):
            there = [f"{rule}.{attr}" in saved for attr in RULE_STATE]
            if any(there) and not all(there):
                raise ValueError(f"amygdala {rule}: some of its learning traces are missing")
            if all(there):
                want["tensors"].update({f"{rule}.{attr}": rules[f"{rule}.{attr}"] for attr in RULE_STATE})
        check_shapes("amygdala", d, want)
        check_streams("amygdala", d["streams"], "bn")

    def load_state_dict(self, d: dict) -> None:
        self.check_state_dict(d)
        self.streams = Streams.from_state(d["streams"])
        self.us = np.array(d["us"], np.float32)
        self._reward = torch.from_numpy(np.array(d["reward"], np.float32))
        sc = np.array(d["scalars"], np.float64)
        self.da_tonic, self.nm_ticks = float(sc[0]), int(sc[1])
        # A save from before the first step has no learning-rule traces yet: drop this instance's, so the rule starts
        # them afresh exactly as in a new instance instead of carrying on with an abandoned timeline's.
        for (a, b), conn in self.net.connections.items():
            for attr in RULE_STATE:
                if f"rule.{a}.{b}.{attr}" not in d["tensors"] and hasattr(conn.update_rule, attr):
                    delattr(conn.update_rule, attr)
        for key, arr in d["tensors"].items():
            kind, *path = key.split(".")
            if kind == "layer":
                obj, attr = self.net.layers[path[0]], path[1]
            elif kind == "conn":
                obj, attr = self.net.connections[(path[0], path[1])], path[2]
            else:
                obj, attr = self.net.connections[(path[0], path[1])].update_rule, path[2]
            val = torch.from_numpy(np.array(arr))    # keep the saved dtype: a run turns some bool spikes to float
            if attr == "w":
                obj.w.data.copy_(val)
            else:
                setattr(obj, attr, val)

    def digest_arrays(self) -> list[np.ndarray]:
        return [v.detach().cpu().numpy() for _, v in sorted(self._tensors().items())]

    def opinion(self) -> dict[str, float]:
        """Mean learned ctx -> BLA weight per assembly (dev telemetry only; the player never sees it)."""
        w = self.net.connections[("ctx", "bla")].w.detach()
        return {a: float(w[:, i * PER_ASM:(i + 1) * PER_ASM].mean()) for i, a in enumerate(ASSEMBLY)}
