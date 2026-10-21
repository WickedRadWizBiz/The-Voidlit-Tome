"""TomeNetEngine: the game run by the Tome's one network instead of the separate Director.

The beat, the Tome's rhetoric (hold, disclose, suggest, probe), its frame and the fact it volunteers come from the
network; its voice's register comes from its readouts. Every turn the network perceives the game (Phase A, before the
rules resolve the player's action), then its basal ganglia race to a commit (Phase B) under the rules' hard NOGO masks;
the rules
apply the committed beat exactly as before, gated by the Tome's own check. The habit cortex inside the network is
the trained Director, run exactly as trained, so the designer's pacing is the network's resting point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..cortex.features import BEATS, Beat
from ..tomebrain import persist
from ..tomebrain.brain import TomeBrain
from ..tomebrain.partition import check_shapes
from ..tomebrain.readout import Mind
from ..voice.style import VoiceStyle, style_from_mind
from .engine import TomeEngine, Turn
from .frames import TomeFrame

BRAIN_FILE = "tome_brain.pt"


class TomeNetEngine(TomeEngine):
    def __init__(self, *args, brain: TomeBrain | None = None, **kw) -> None:
        super().__init__(*args, **kw)
        self.brain = brain or TomeBrain(run_seed=self.state.seed, director=self.director.model)
        self.last_commit = None
        self._slot_ids: list[str] = []          # the fact each P.fact assembly was driven with at Phase B
        self._us_sent: dict[str, float] = {}    # the stimuli Phase B received
        self.late_us: dict[str, float] = {}     # stimuli set after Phase B, for the next Phase A (saved)

    def _senses_vector(self, t: Turn) -> np.ndarray:
        s = t.sensory
        if s is None:
            return np.zeros(8, np.float32)
        return np.array([s.aggression, s.deception, s.leverage, s.exposure, max(0.0, s.rapport),
                         max(0.0, -s.rapport), s.boldness, 1.0 if s.target == "tome" else 0.0], np.float32)

    def _body_vector(self) -> np.ndarray:
        st, ch = self.state, self.state.character
        return np.array([st.lumen.fraction, ch.hull / max(1, ch.hull_max), ch.stress / 10.0], np.float32)

    def _debug_extra(self) -> dict:
        return {"mind": self.brain.mind.telemetry(), "opinion": self.brain.bn.opinion(),
                "fallbacks": self.brain.fallbacks}

    def tome_mods(self) -> dict[str, int]:
        """The Tome's own check modifiers come from its network (ATP, arousal, drive, serotonin; design §8.6)."""
        return self.brain.mods()

    def _before_act(self, t: Turn) -> None:
        self.brain.set_us(self.late_us)
        self.late_us, self._us_sent = {}, {}
        self.brain.set_input(director_features=None, senses=self._senses_vector(t), body=self._body_vector(),
                             glucose=0.15 + 0.85 * self.state.lumen.fraction)
        self.brain.phase_a()

    def _fact_salience(self) -> np.ndarray:
        """What each fact in the schemer's slots is worth volunteering to the Tome now (designed input, 0..1): the
        cheaper it is to tell and the more it serves a live agenda (light when the Lumen is low, driving off the
        NPCs when they are here), the stronger its slot is driven. The network decides; this only informs it."""
        st = self.state
        agenda = {"feed_light": 1.0 - st.lumen.fraction, "leave_npc": 1.0 if st.npcs_present else 0.0,
                  "go_lamp_room": 0.5}
        out = np.zeros(8, np.float32)
        slots = self.ledger.slots(self.chapter.id)
        self._slot_ids = [f.id for f in slots]
        for i, f in enumerate(slots):
            out[i] = min(1.0, 0.25 + 0.5 * agenda.get(f.agenda or "", 0.0) + 0.25 * (1.0 - f.costs_tome))
        return out

    def _choose_rhetoric(self, t: Turn, n_slots: int) -> tuple[str, str, int | None]:
        out = self.last_commit
        if out is None:
            return super()._choose_rhetoric(t, n_slots)
        # The winning slot means the fact it was driven with. This turn's reveal may since have told the player a
        # slot fact, which shifts the slots after it: find the chosen fact in the current list (gone if it was told).
        n = len(self._slot_ids)
        chosen = self._slot_ids[int(np.argmax(out.facts[:n]))] if n and out.facts[:n].max() > 0 else None
        now = [f.id for f in self.ledger.slots(self.chapter.id)]
        return out.rhetoric.channel, out.frame.channel, now.index(chosen) if chosen in now else None

    def _style(self) -> VoiceStyle:
        return style_from_mind(self.brain.mind)

    def _choose_beat(self, t: Turn, allowed: set[Beat]) -> tuple[Beat, dict[str, float]]:
        feats = self._features(t).tensor().numpy()
        ch = self.state.character
        if ch.hull <= max(2, ch.hull_max // 3):
            t.us["bearer_near_death"] = 1.0
        self._us_sent = dict(t.us)
        self.brain.set_us(self._us_sent)
        self.brain.set_input(director_features=feats, senses=self._senses_vector(t), body=self._body_vector(),
                             glucose=0.15 + 0.85 * self.state.lumen.fraction, facts=self._fact_salience())
        out = self.brain.phase_b(feats, {b.value for b in allowed})
        self.last_commit = out
        beat = Beat(out.pace.channel)
        assert beat in allowed, "the race committed a forbidden beat"
        return beat, {b.value: round(float(p), 4) for b, p in zip(BEATS, out.prior)}

    def _compose(self, t: Turn) -> TomeFrame:
        # What the rules set after Phase B (an NPC beat's extract tactic marks a loss) reaches the amygdala at the
        # next Phase A instead of being lost.
        self.late_us = {k: v for k, v in t.us.items() if self._us_sent.get(k) != v}
        return super()._compose(t)

    # ------------------------------------------------------------------ persistence
    def _brain_extra(self) -> dict:
        return {"turn": self.brain.turn, "fallbacks": self.brain.fallbacks, "mind": self.brain.mind.state_dict(),
                "late_us": dict(self.late_us)}

    def save(self, directory: str | Path) -> Path:
        d = super().save(directory)
        persist.save(self.brain.cosim, Path(d) / BRAIN_FILE, extra=self._brain_extra())
        return d

    def _snapshot(self) -> dict:
        return {**super()._snapshot(), "brain": persist.snapshot(self.brain.cosim, self._brain_extra())}

    def _read_save(self, d: Path) -> dict:
        # a slot this brain does not fit (a Director game's, which has none, or one from another window size or
        # connectome) is refused here, with nothing changed
        saved = super()._read_save(d)
        saved["brain"] = blob = persist.read(self.brain.cosim, d / BRAIN_FILE)
        extra = blob["extra"]
        check_shapes("brain extra", extra, {"turn": (), "fallbacks": (), "mind": None, "late_us": None})
        mind, fresh = Mind(), Mind()
        mind.load_state_dict(extra["mind"])         # its own reader (which takes older saves' readouts too)
        if any(set(getattr(mind, k)) != set(getattr(fresh, k)) for k in ("needs", "pad", "affect", "mods")):
            raise ValueError("the Tome's mind in this save does not fit this build")
        check_shapes("late stimuli", extra["late_us"], {k: () for k in extra["late_us"]})
        return saved

    def _take_save(self, saved: dict) -> None:
        super()._take_save(saved)
        extra = persist.apply(self.brain.cosim, saved["brain"])
        self.brain.turn, self.brain.fallbacks = int(extra["turn"]), int(extra["fallbacks"])
        self.brain.mind.load_state_dict(extra["mind"])
        self.late_us = {str(k): float(v) for k, v in extra["late_us"].items()}
