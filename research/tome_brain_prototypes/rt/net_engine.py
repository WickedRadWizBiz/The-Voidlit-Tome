"""The game engine with the Director replaced by the Tome's hybrid network (design prototype, repo untouched)."""
from __future__ import annotations
import math, pickle, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/home/user/The-Voidlit-Tome/brain")
import cosim as CS
from voidlit.tome.engine import TomeEngine
from voidlit.cortex.features import BEATS
from voidlit.senses.state import Intent

INTENTS = list(Intent)
K_COMMIT = 6

class TomeNetEngine(TomeEngine):
    def __init__(self, *a, seed: int = 0, net: CS.Tome | None = None, **kw):
        super().__init__(*a, seed=seed, **kw)
        self.net = net or CS.Tome(seed=seed)
        self.net_rng = np.random.default_rng(seed + 11)
        self.net_log: list[dict] = []

    def _net_features(self, t, f) -> np.ndarray:
        """64 thalamic relay channels: Director features (18), intent (11), approach (4), affect channels (7),
        last outcome (4), NPC state (6), clocks/obstacles (4), padding."""
        x = np.zeros(64, np.float32)
        x[:18] = f.tensor().numpy()
        s = t.sensory
        if s is not None:
            x[18 + INTENTS.index(s.intent)] = 1.0
            x[29 + ["grit", "guile", "insight", "none"].index(s.effective_approach.value)] = 1.0
            x[33:40] = [s.aggression, s.deception, s.leverage, s.exposure, max(0, s.rapport), max(0, -s.rapport), s.boldness]
        if t.outcome is not None:
            x[40 + ["fail", "mixed", "success", "critical"].index(t.outcome.value)] = 1.0
        for k, nid in enumerate(sorted(self.brains)[:2]):
            b = self.brains[nid]
            x[44 + 3 * k] = float(nid in self.state.npcs_present)
            for j, role in enumerate(("ally", "betray")):
                r = b.spec.role_index(role)
                x[45 + 3 * k + j] = float(b.momentum[r]) if r is not None else 0.0
        return x

    def _beat(self, t):
        st = self.state
        allowed = self._allowed_beats(spoke=bool(t.npc_lines))
        feats = self._features(t)
        with torch.no_grad():
            p = self.director.probabilities(feats).numpy()            # habit pathway: the trained Director
        self.net.set_game_inputs(self._net_features(t, feats), st.lumen.fraction)
        mask = np.array([b in allowed for b in BEATS])
        gum = -np.log(-np.log(self.net_rng.random(8)))                  # trial-persistent MSN excitability
        score = np.where(mask, np.log(p + 1e-6) + gum, -np.inf)
        sal = np.zeros(CS.N_CH, np.float32)
        sal[:8] = np.where(mask, 0.05 + 0.03 * np.clip(score - score.max() + 4.0, 0, None), 0.0)
        M, I = self.net.sj.sizes["M"], self.net.sj.sizes["I"]
        het = 0.85 + 0.3 * torch.rand(CS.N_CH * M, generator=torch.Generator().manual_seed(1))
        self.net.sj.prior_bias = torch.repeat_interleave(torch.as_tensor(sal), M) * het
        nogo = np.zeros(CS.N_CH, np.float32); nogo[:8] = np.where(mask, 0.0, 0.5)   # rules fair-play mask
        self.net.sj.nogo = torch.repeat_interleave(torch.as_tensor(nogo), I)
        until = lambda c: c[:8].max() >= K_COMMIT
        with torch.no_grad():
            self.net.run_turn(ms=240, until=until, min_ms=120)
        c = self.net.turn_counts[:8] * mask
        k = int(np.argmax(c)) if c.max() > 0 else int(np.argmax(score))
        beat = BEATS[k]
        self.net_log.append({"beat": beat.value, "ms": self.net.turn_elapsed, "released": bool(c.max() > 0)})
        probs = {b.value: round(float(q), 4) for b, q in zip(BEATS, p)}
        orig = self.director.choose
        self.director.choose = lambda features, rng, allowed=None: (beat, probs)
        try:
            super()._beat(t)
        finally:
            self.director.choose = orig

    def save(self, directory):
        d = super().save(directory)
        (Path(directory) / "tome_net.pkl").write_bytes(pickle.dumps({"net": self.net.state(),
                                                                    "net_rng": self.net_rng.bit_generator.state}))
        return d

    def load(self, directory):
        super().load(directory)
        blob = pickle.loads((Path(directory) / "tome_net.pkl").read_bytes())
        self.net.load(blob["net"])
        self.net_rng.bit_generator.state = blob["net_rng"]
