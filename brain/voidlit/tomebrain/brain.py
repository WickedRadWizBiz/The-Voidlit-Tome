"""TomeBrain: the Tome's one network as the game sees it.

It owns the co-simulation (the JAX partition plus the SpikingJelly basal ganglia on one clock), feeds the game's
state to the thalamus, and reads the Tome's choices from its VA/VL thalamus with the race-to-threshold rule:

    each loop commits to the first channel whose 4 VA/VL cells fire K = 6 spikes within any 20-ms window,
    counted from 120 ms after Phase B starts; Phase B ends when PACE and RHETORIC have committed, or at 240 ms,
    when the channel with the most VA/VL spikes wins and the fallback counter goes up.

Rules-forbidden channels get a hard NOGO in GPi and are never counted, so a forbidden commit cannot happen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .clock import Cosim
from .jaxpart import params as PR
from .partition import BusEdge, JaxPartition
from .sjpart import basal_ganglia as BG

PHASE_A_MS = 60
COMMIT_AFTER_MS = 120
CAP_MS = 240
K_SPIKES, WINDOW_MS = 6, 20
QUANTUM = 4             # Phase B ends on a 4-ms boundary after the commit, whatever the window size
CHANNEL_PA = 220.0
REST_DRIVE = 0.2
# VA/VL live under tonic GPi inhibition (multi-synaptic basket terminals, strong): they need the extra drive and the
# heavy GPi synapse so that a released channel fires its commit burst while the others stay silent. Both were fitted
# for race fidelity (gate G10a): 94% of 16 trials matched argmax(log p + Gumbel), 1 fallback, commit at 120 ms.
VAVL_DRIVE_PA = 220.0
GPI_VAVL_NS = 12.0
CTX = ["S.L5IT", "T.L5IT", "P.L5IT", "I.L5IT", "P.L5ET", "CA1.ctx", "P.MTX"]


@dataclass
class Commit:
    loop: str
    channel: str | None
    at_ms: int | None
    fallback: bool
    counts: dict[str, int] = field(default_factory=dict)


@dataclass
class PhaseBOut:
    pace: Commit
    rhetoric: Commit
    frame: Commit
    prior: np.ndarray                 # the habit prior p over the beats
    ms: int


def bus_edges() -> list[BusEdge]:
    edges = []
    for c in range(BG.N_CH):
        edges.append(BusEdge("S.gpi", BG.N_CH * BG.I, f"P.vavl{c}", BG.I, (PR.GABA_A,), PR.SOMA, GPI_VAVL_NS, 5,
                             src_range=(c * BG.I, (c + 1) * BG.I)))
    edges.append(BusEdge("S.vs", BG.N_VS, "RMTg", 4, (PR.GABA_A,), PR.SOMA, 1.0, 6))
    return edges


class TomeBrain:
    def __init__(self, run_seed: int = 0, director=None, W: int = 4) -> None:
        jax_part = JaxPartition(run_seed, edges=bus_edges(), outputs={"J.ctx": CTX, "J.pvt": "PVT", "J.vavl": "P.MTX"},
                                graded=("J.nm",))
        self.sj = BG.SJPartition(run_seed, director=director)
        self.cosim = Cosim(jax_part, [self.sj], W=W)
        self.jax = jax_part
        self.net = jax_part.net
        self.turn = 0
        self.fallbacks = 0
        mu = np.asarray(jax_part.state["mu"]).copy()
        mu[self.net.idx("P.MTX")] += VAVL_DRIVE_PA
        import jax.numpy as jnp
        jax_part.state["mu"] = jnp.asarray(mu)
        self._drive = self._rest_drive()
        jax_part.ext_fn = lambda t0, W: np.broadcast_to(self._drive, (W, self.net.n)).astype(np.float32)

    # ------------------------------------------------------------------ input
    def _rest_drive(self) -> np.ndarray:
        drive = np.zeros(self.net.n, np.float32)
        for c in PR.COLUMNS:
            sl = self.net.idx(f"{c}.TC")
            drive[sl] = REST_DRIVE * CHANNEL_PA
        return drive

    def set_input(self, director_features: np.ndarray | None = None, senses: np.ndarray | None = None,
                  body: np.ndarray | None = None, glucose: float | None = None) -> None:
        """Rate-code the game into thalamic channels (two TC cells each, 220 pA x f): the Director's 18 features to
        the S reader, the senses to the T mind-reader, the body (Lumen, Hull, Stress) to the I insula."""
        drive = self._rest_drive()
        for col, vec in (("S", director_features), ("T", senses), ("I", body)):
            if vec is None:
                continue
            sl = self.net.idx(f"{col}.TC")
            f = np.clip(np.asarray(vec, np.float32), 0.0, 1.0)
            n = min(len(f), (sl.stop - sl.start) // 2)
            drive[sl.start:sl.start + 2 * n] = np.repeat(f[:n], 2) * CHANNEL_PA
        self._drive = drive
        if glucose is not None:
            self.jax.glucose = float(glucose)

    # ------------------------------------------------------------------ phases
    def phase_a(self) -> None:
        """60 ms before the rules resolve the player's action: the network perceives the new input."""
        self.cosim.clock.phase = "A"
        self.cosim.run(PHASE_A_MS)

    def phase_b(self, director_features: np.ndarray, allowed: set[str] | None = None) -> PhaseBOut:
        clock = self.cosim.clock
        clock.phase = "B"
        prior = self.sj.begin_phase_b(director_features, clock.t_ms, self.turn, allowed)
        start = clock.t_ms
        rasters, gpi = [], []
        commits: dict[str, Commit] = {}
        W = self.cosim.W
        assert PHASE_A_MS % QUANTUM == 0 and CAP_MS % QUANTUM == 0
        if QUANTUM % W:
            raise ValueError(f"the window ({W} ms) must divide the {QUANTUM}-ms Phase B quantum")
        while clock.t_ms - start < CAP_MS:
            rec = self.cosim.run(QUANTUM, record=("J.vavl", "S.gpi"))
            rasters.append(rec["J.vavl"])
            gpi.append(rec["S.gpi"])
            if clock.t_ms - start >= COMMIT_AFTER_MS:
                commits = self._race(np.concatenate(rasters), allowed, final=False)
                if "PACE" in commits and "RHETORIC" in commits:
                    break
        raster = np.concatenate(rasters)
        gpi_per_ch = np.concatenate(gpi).reshape(-1, BG.N_CH, BG.I).sum((0, 2))
        final = self._race(raster, allowed, final=True, committed=commits, gpi=gpi_per_ch)
        self.sj.end_phase_b()
        self.sj.end_turn(clock.t_ms - start + PHASE_A_MS)
        self.turn += 1
        clock.turn = self.turn
        clock.phase = "idle"
        if final["PACE"].fallback:
            self.fallbacks += 1
        return PhaseBOut(final["PACE"], final["RHETORIC"], final["FRAME"], prior, clock.t_ms - start)

    def _race(self, raster: np.ndarray, allowed: set[str] | None, final: bool,
              committed: dict[str, Commit] | None = None, gpi: np.ndarray | None = None) -> dict[str, Commit]:
        per_ch = raster.reshape(raster.shape[0], BG.N_CH, 4).sum(2)        # [ms, channel]
        out = dict(committed or {})
        for loop in BG.LOOPS:
            if loop in out:
                continue
            chans = [c for c, (lp, name) in enumerate(BG.CHANNELS) if lp == loop
                     and (loop != "PACE" or allowed is None or name in allowed)]
            best = None
            for c in chans:
                cs = np.convolve(per_ch[:, c], np.ones(WINDOW_MS), mode="full")[:per_ch.shape[0]]
                hits = np.flatnonzero((cs >= K_SPIKES) & (np.arange(len(cs)) >= COMMIT_AFTER_MS))
                if len(hits):
                    # earliest wins; channels qualifying at the same ms (often all at the 120-ms floor) are ranked by
                    # their windowed count, then by their spikes over the whole phase, never by list order
                    key = (int(hits[0]), -int(cs[hits[0]]), -int(per_ch[:, c].sum()))
                    if best is None or key < best[1]:
                        best = (c, key)
            counts = {BG.CHANNELS[c][1]: int(per_ch[:, c].sum()) for c in chans}
            if best is not None:
                out[loop] = Commit(loop, BG.CHANNELS[best[0]][1], best[1][0], False, counts)
            elif final:
                # fallback: the most VA/VL spikes; if the thalamus stayed silent, the most-suppressed GPi channel
                g = gpi if gpi is not None else np.zeros(BG.N_CH)
                top = max(chans, key=lambda c: (per_ch[:, c].sum(), -g[c]))
                out[loop] = Commit(loop, BG.CHANNELS[top][1], None, True, counts)
        return out

    # ------------------------------------------------------------------ persistence
    def digest(self) -> str:
        return self.cosim.digest()
