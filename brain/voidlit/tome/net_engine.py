"""TomeNetEngine: the game run by the Tome's one network instead of the separate Director.

Only the beat choice changes. Every turn the network perceives the game (Phase A, before the rules resolve the
player's action), then its basal ganglia race to a commit (Phase B) under the rules' hard NOGO masks; the rules
apply the committed beat exactly as before, gated by the Tome's own check. The habit cortex inside the network is
the trained Director, run exactly as trained, so the designer's pacing is the network's resting point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..cortex.features import BEATS, Beat
from ..tomebrain import persist
from ..tomebrain.brain import TomeBrain
from .engine import TomeEngine, Turn

BRAIN_FILE = "tome_brain.pt"


class TomeNetEngine(TomeEngine):
    def __init__(self, *args, brain: TomeBrain | None = None, **kw) -> None:
        super().__init__(*args, **kw)
        self.brain = brain or TomeBrain(run_seed=self.state.seed, director=self.director.model)
        self.last_commit = None

    def _senses_vector(self, t: Turn) -> np.ndarray:
        s = t.sensory
        if s is None:
            return np.zeros(8, np.float32)
        return np.array([s.aggression, s.deception, s.leverage, s.exposure, max(0.0, s.rapport),
                         max(0.0, -s.rapport), s.boldness, 1.0 if s.target == "tome" else 0.0], np.float32)

    def _body_vector(self) -> np.ndarray:
        st, ch = self.state, self.state.character
        return np.array([st.lumen.fraction, ch.hull / max(1, ch.hull_max), ch.stress / 10.0], np.float32)

    def _before_act(self, t: Turn) -> None:
        self.brain.set_input(director_features=None, senses=self._senses_vector(t), body=self._body_vector(),
                             glucose=0.15 + 0.85 * self.state.lumen.fraction)
        self.brain.phase_a()

    def _choose_beat(self, t: Turn, allowed: set[Beat]) -> tuple[Beat, dict[str, float]]:
        feats = self._features(t).tensor().numpy()
        self.brain.set_input(director_features=feats, senses=self._senses_vector(t), body=self._body_vector())
        out = self.brain.phase_b(feats, {b.value for b in allowed})
        self.last_commit = out
        beat = Beat(out.pace.channel)
        assert beat in allowed, "the race committed a forbidden beat"
        return beat, {b.value: round(float(p), 4) for b, p in zip(BEATS, out.prior)}

    def save(self, directory: str | Path) -> Path:
        d = super().save(directory)
        persist.save(self.brain.cosim, Path(d) / BRAIN_FILE, extra={"turn": self.brain.turn,
                                                                   "fallbacks": self.brain.fallbacks})
        return d

    def load(self, directory: str | Path) -> None:
        super().load(directory)
        extra = persist.load(self.brain.cosim, Path(directory) / BRAIN_FILE)
        self.brain.turn = int(extra.get("turn", 0))
        self.brain.fallbacks = int(extra.get("fallbacks", 0))
