"""The SpikingJelly partition: the habit cortex and the basal ganglia as one fused LIF population.

Three cortico-basal ganglia-thalamic loops choose what the Tome does each turn:

    PACE (8)      calm, escalate, clue, npc, hazard, respite, twist, advance
    RHETORIC (6)  hold, disclose, suggest, probe, corrupt_npc, bend
    FRAME (4)     help, necessity, fear, kinship

Per channel: D1 16, D2 16, GPe 6, STN 3, GPi 6; shared FSI 32 and a ventral-striatum critic of 16. GPi is tonic
(about 65 Hz) and crosses the bus to inhibit its channel's four VA/VL cells in the JAX P column. A channel wins when
its D1 cells silence its GPi and its VA/VL cells fire (the race commit, read by ``brain.TomeBrain``).

Learning is three-factor with synaptic tags (Frey & Morris): eligibility from cortical traces x MSN spikes is
captured into tags at the end of each turn, and phasic dopamine (read from the JAX VTA over the bus) turns tags into
weight changes, D1 up and D2 down for a positive prediction error, with a slow pull back to the initial weights
so the Director stays the resting point.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ...cortex.features import BEATS
from ..partition import Partition, check_scalars, check_shapes, check_streams, shapes_of
from ..rng import Streams
from .habit import PER_BEAT, RATE_MAX, Habit
from .nodes import PerNeuronLIF

LOOPS = {
    "PACE": tuple(b.value for b in BEATS),
    "RHETORIC": ("hold", "disclose", "suggest", "probe", "corrupt_npc", "bend"),
    "FRAME": ("help", "necessity", "fear", "kinship"),
}
CHANNELS = [(loop, name) for loop, names in LOOPS.items() for name in names]
N_CH = len(CHANNELS)                     # 18
LOOP_OF = np.array([list(LOOPS).index(loop) for loop, _ in CHANNELS])
M, E, S, I = 16, 6, 3, 6                 # D1 and D2 MSNs, GPe, STN, GPi per channel
N_FSI, N_VS = 32, 16
N_CTX = 328                              # cortical, hippocampal and thalamostriatal lines (bus J.ctx)
N_PVT = 8
N_LIM = 48                               # amygdala BLA lines (bus B.bla)
TAU = dict(d1=20.0, d2=20.0, fsi=5.0, gpe=10.0, stn=6.0, gpi=10.0, vs=20.0)
# The D2 MSNs' tonic drive is the up-state the network's sparse cortex cannot hold them in. Their only input was the
# cortical lines, about a twentieth of threshold, so they never fired, carried no tags and no dip could strengthen them
# (the design's 0.3 x 5-HT became 0.03 x its excursion from tonic, which is zero on average). At 0.04 they fire on the
# cortex's fluctuations at 0-1.5 Hz a turn (bot games: 0.3-0.6 Hz median, 0.4-1.0 Hz mean), all channels alike, which
# leaves GPe and the race as they were (gate G10a: 0.947 hits, 0 fallbacks), and a dip strengthens their synapses.
TONIC = dict(gpe=0.12, stn=0.15, gpi=0.13,   # GPe ~77 Hz, STN ~33 Hz, GPi ~67 Hz at rest (with noise)
             vs=0.04,                         # the critic idles at a few Hz, so it can learn what predicts reward
             d2=0.04)
W_HABIT = 0.012          # habit trace -> D1, per unit of filtered habit spikes (20-ms trace)
# One unit of log-probability moves a D1 cell's input by W_HABIT x 4 cells x 20 ms x RATE_MAX / SCALE = 0.024, so
# the Gumbel excitability uses the same gain: the race then samples argmax(log p + Gumbel), i.e. from p.
BETA_G = 0.024
HYPER = 0.5              # hyperdirect: pooled cortical salience of a loop -> its STN (fitted, G10a)
LATERAL = 0.5            # MSN collateral inhibition onto the D1 cells of the loop's other channels (winner-take-all)
NOISE = 0.04             # per-ms current noise (threshold units) on GPe, STN, GPi, FSI and VS
NOISE_MSN = 0.015        # ... and on the D1/D2 MSNs (less: their small drive differences carry the choice)
NOGO = 0.5               # rules-forbidden channel: extra GPi drive (hard NOGO) ...
NOGO_D1 = 0.2            # ... and its D1 cells held down, so a vetoed action cannot suppress its rivals
ETA, K_TAG, ANCHOR = 2e-4, 0.05, 0.002
TAG_TAU_MS = 60_000.0
QUIET_MS = 20_000.0
_K20 = math.exp(-1 / 20.0)
# RHETORIC and FRAME have no habit prior: their D1 cells had only the cortex's weak, non-selective input and their
# Gumbel excitability, so many of their races never released GPi (RHETORIC: 31% at init, 43% of bot-game turns), ran
# Phase B to its cap and were decided by the fallback rule, outside the basal ganglia. In Phase B their D1 cells get the
# drive a PACE beat gets at the top of its prior (habit cells at RATE_MAX, log p = log p_max), so they race as PACE does
# with a flat prior: argmax(Gumbel), tilted only by what the cortex and the amygdala have taught them. At init RHETORIC
# then commits in 100 of 100 races, 95 to its Gumbel argmax (FRAME: 90, 93); no Phase B reaches the cap.
FLAT_D1 = W_HABIT * PER_BEAT * RATE_MAX / (1 - _K20)      # 0.0984
TONIC_TAU_MS = 6_000.0   # tonic DA / 5-HT: running mean of the concentration (the DA's over the quiet interval too)
# The striatum reads dopamine against its tonic level, but never finer than DA_FLOOR (concentration units; the fed VTA
# sustains about 1.0). A starving VTA paces at under 1 Hz: its 24 cells' spikes arrive about 50 ms apart and each one
# moves the concentration by a quarter of its tonic level, so read against that level alone their shot noise was a
# stream of pauses and bursts. Against the floor, eventless starving turns read neither; fed, the floor is never met.
# So at Lumen 0 (tonic about 0.2) no dip teaches, a punishment's included: a silent VTA has no firing left to pause.
# Its cells fire about 0.3 Hz during a turn, most turns not at all, and their own gaps take the concentration as low
# as a punishment's 250-ms pause does (0.05 against a tonic of 0.2; 2 seeds x 16 eventless and 6 punished turns): read
# against the tonic alone, 17 of the 32 eventless starving turns would cross the dip threshold, as all 12 punished
# ones do. A starving Tome's punishments still teach its amygdala (the stimulus itself is the teacher there).
DA_FLOOR = 0.5
PHASIC = 0.3             # a dopamine burst: the slow dopamine at least 30% above tonic. Below that is the sparse VTA's
                         # shot noise and wobble, which must teach nothing: with no event it stays within -0.19..+0.23
                         # (200 turns, 2 seeds), while a gift reaches 0.44-0.57
PAUSE = 0.5              # a dopamine dip: the concentration itself half its tonic level below it (da_signal). The
                         # slow dopamine needs 220 ms of silence to fall 30%, longer than most turns (Phase B often
                         # commits at 124 ms), but the concentration halves after 139 ms. The VTA's pacemakers never
                         # all fall silent by chance (with no pause the concentration never went below -0.38 in 300
                         # eventless turns), and an eventless reward pauses them too briefly to halve it (-0.47 at most
                         # in 300). Its upper tail is shot noise (up to +0.84), so a burst is read from the slow DA.
WARMUP_MS = 500
# Dopamine acts twice: the fast phasic signal gates plasticity (below), while its modulation of MSN excitability runs
# through second messengers (cAMP/PKA) and follows it only slowly (the shot noise of a sparse VTA is SD ~0.45 of tonic
# per ms; read fast and applied to the prior alone, it rescaled the prior against the Gumbel term and broke the race).
DA_MOD_TAU_MS = 200.0


def _block(n_pre: int, n_post: int, w: float) -> torch.Tensor:
    pre = torch.arange(N_CH).repeat_interleave(n_pre)
    post = torch.arange(N_CH).repeat_interleave(n_post)
    return (pre[:, None] == post[None, :]).float() * w


class SJPartition(Partition):
    name = "sj"

    def __init__(self, run_seed: int, anatomy_seed: int = 0, director=None) -> None:
        torch.set_num_threads(1)
        self.streams = Streams(run_seed, "sj")
        self.habit = Habit(director, self.streams)
        g = torch.Generator().manual_seed(1000 + anatomy_seed)
        sizes = [N_CH * M, N_CH * M, N_FSI, N_CH * E, N_CH * S, N_CH * I, N_VS]
        names = ["d1", "d2", "fsi", "gpe", "stn", "gpi", "vs"]
        off = np.cumsum([0] + sizes)
        self.off = {k: (int(off[i]), int(off[i + 1])) for i, k in enumerate(names)}
        self.n = int(off[-1])
        tau = torch.cat([torch.full((n,), TAU[k]) for k, n in zip(names, sizes)])
        self.pop = PerNeuronLIF(tau)
        # plastic input weights (with their anchors) and fixed ones
        self.W = {
            # cortex -> action channels starts non-selective: every channel's cells get the same random draw, so
            # however active the cortex is, it cannot tilt the race at initialisation; selectivity is learned (G10a)
            "ctx_d1": torch.rand(N_CTX, M, generator=g).repeat(1, N_CH) * 0.004,
            "ctx_d2": torch.rand(N_CTX, M, generator=g).repeat(1, N_CH) * 0.004,
            "ctx_vs": torch.rand(N_CTX, N_VS, generator=g) * 0.004,
            "lim_d1": torch.rand(N_LIM, M, generator=g).repeat(1, N_CH) * 0.003,
        }
        self.W0 = {k: v.clone() for k, v in self.W.items()}
        self.w_ctx_fsi = torch.rand(N_CTX, N_FSI, generator=g) * 0.004
        self.w_pvt_vs = torch.rand(N_PVT, N_VS, generator=g) * 0.05
        self.w_lim_vs = torch.rand(N_LIM, N_VS, generator=g) * 0.05
        hab = torch.zeros(len(BEATS) * PER_BEAT, N_CH * M)
        for b in range(len(BEATS)):
            hab[b * PER_BEAT:(b + 1) * PER_BEAT, b * M:(b + 1) * M] = W_HABIT
        self.w_habit = hab
        # recurrent matrix (one CSR product per ms)
        R = torch.zeros(self.n, self.n)

        def put(src, dst, w):
            a, b = self.off[src]
            c, d = self.off[dst]
            R[a:b, c:d] = w
        put("fsi", "d1", -torch.rand(N_FSI, N_CH * M, generator=g) * 0.02)
        put("fsi", "d2", -torch.rand(N_FSI, N_CH * M, generator=g) * 0.02)
        put("d1", "gpi", -_block(M, I, 0.8))
        loop_d1 = torch.from_numpy(LOOP_OF[np.repeat(np.arange(N_CH), M)])
        ch_d1 = torch.arange(N_CH).repeat_interleave(M)
        put("d1", "d1", -LATERAL * ((loop_d1[:, None] == loop_d1[None, :]) & (ch_d1[:, None] != ch_d1[None, :])).float())
        put("d2", "gpe", -_block(M, E, 0.4))
        put("gpe", "stn", -_block(E, S, 0.05))
        put("stn", "gpe", _block(S, E, 0.2))
        put("gpe", "gpi", -_block(E, I, 0.1))
        same_loop = torch.as_tensor(LOOP_OF[np.repeat(np.arange(N_CH), S)][:, None]
                                    == LOOP_OF[np.repeat(np.arange(N_CH), I)][None, :]).float()
        put("stn", "gpi", same_loop * 0.06)            # diffuse hyperdirect, within the loop
        self.RT = R.T.contiguous().to_sparse_csr()
        self.tonic = torch.zeros(self.n)
        for k, v in TONIC.items():
            a, b = self.off[k]
            self.tonic[a:b] = v
        self.noise_scale = torch.full((self.n,), NOISE)
        self.noise_scale[: 2 * N_CH * M] = NOISE_MSN
        self.nogo = torch.zeros(N_CH * I)
        self.nogo_d1 = torch.zeros(N_CH * M)
        self.gumbel = torch.zeros(N_CH)
        no_habit = LOOP_OF != list(LOOPS).index("PACE")
        self.flat = torch.from_numpy(np.repeat(no_habit, M).astype(np.float32))       # D1 cells without a habit prior
        self.kappa_h = 1.0
        self.prev = torch.zeros(self.n)
        self.habit_trace = torch.zeros(len(BEATS) * PER_BEAT)
        self.ctx_trace = torch.zeros(N_CTX)
        self.lim_trace = torch.zeros(N_LIM)
        self.elig = {k: torch.zeros_like(v) for k, v in self.W.items()}
        self.tags = {k: torch.zeros_like(v) for k, v in self.W.items()}
        self.tag_gain = {k: torch.zeros_like(v) for k, v in self.W.items()}     # tanh(k T), fixed within a turn
        self.loop_onehot = torch.zeros(N_CH, len(LOOPS))
        self.loop_onehot[torch.arange(N_CH), torch.from_numpy(LOOP_OF)] = 1.0
        self.phase_b_tick = 0
        self.da_tonic = 0.0      # running means of the dopamine and serotonin concentrations
        self.da_mod = 0.0        # the slow excitability effect of dopamine on the MSNs
        self.ht_tonic = 0.0
        self.nm_ticks = 0
        self.in_phase_b = False
        self.outputs = {"S.gpi": N_CH * I, "S.vs": N_VS}

    # ------------------------------------------------------------------ turn control
    def begin_phase_b(self, features: np.ndarray, tick: int, turn: int, allowed: set[str] | None) -> np.ndarray:
        """Prime the habit prior, draw this turn's MSN excitabilities, and set the rules' hard NOGO."""
        p = self.habit.prime(features, tick)
        self.gumbel = torch.from_numpy(self.streams.gumbel("bg_gumbel", turn, N_CH))
        # every decision starts with the MSNs in their down-state: no head start for last turn's winner
        msn = slice(self.off["d1"][0], self.off["d2"][1])
        self.pop.v[msn] = 0.0
        self.prev[msn] = 0.0
        self.habit_trace = torch.zeros_like(self.habit_trace)
        self.nogo = torch.zeros(N_CH * I)
        self.nogo_d1 = torch.zeros(N_CH * M)
        if allowed is not None:
            for c, (loop, name) in enumerate(CHANNELS):
                if loop == "PACE" and name not in allowed:
                    self.nogo[c * I:(c + 1) * I] = NOGO
                    self.nogo_d1[c * M:(c + 1) * M] = NOGO_D1
        self.in_phase_b = True
        self.phase_b_tick = tick
        return p

    def end_phase_b(self) -> None:
        self.habit.quiet()
        self.gumbel = torch.zeros(N_CH)
        self.nogo = torch.zeros(N_CH * I)
        self.nogo_d1 = torch.zeros(N_CH * M)
        self.in_phase_b = False

    def da_signal(self, conc: float) -> float:
        """The phasic dopamine signal of a concentration: its excursion from tonic, in units of tonic (DA_FLOOR at
        least). -1 is no dopamine at all at a fed tonic level."""
        return (conc - self.da_tonic) / max(self.da_tonic, DA_FLOOR)

    def rest(self, da_turn: float, da_rest: float, ht: float, tau_post_ms: float) -> None:
        """The quiet interval for the tonic running means (design §5.6), integrated exactly over its 20 s. The dopamine
        concentration follows the VTA from the level its turn rate sustains back to the level its resting rate
        sustains, da_rest + (da_turn - da_rest) exp(-t / tau_post); the resting rate is the network's at the current
        Lumen, so the tonic level follows the slow change of a starving VTA (integrated over turn ticks alone, about
        20 turns to adapt, a dying light read as a pause on every turn). Serotonin is held at ``ht``, as the quiet
        holds it (over turn ticks alone, the tonic level lagged the concentration's climb over the first ten turns,
        and the D2 cells fired at 40 Hz on the difference)."""
        a = (da_turn - da_rest) * tau_post_ms / (tau_post_ms - TONIC_TAU_MS)
        keep = math.exp(-QUIET_MS / TONIC_TAU_MS)
        self.da_tonic = da_rest + a * math.exp(-QUIET_MS / tau_post_ms) + (self.da_tonic - da_rest - a) * keep
        self.ht_tonic = ht + (self.ht_tonic - ht) * keep
        self.nm_ticks += int(QUIET_MS)

    def end_turn(self, elapsed_ms: float) -> None:
        """Capture eligibility into tags; tags decay over the quiet interval; weights drift toward the anchor. The
        slow dopamine relaxes to tonic over the quiet's 20 s (the brain returns the concentration to its resting level,
        which ``rest`` takes the tonic level to), so a burst or dip that the turn's end cut short never pays the tags
        captured here."""
        decay = math.exp(-elapsed_ms / TAG_TAU_MS)
        for k in self.W:
            self.tags[k] = self.tags[k] * decay + self.elig[k]
            self.elig[k].zero_()
            self.tags[k] *= math.exp(-QUIET_MS / TAG_TAU_MS)
            self.tag_gain[k] = torch.tanh(K_TAG * self.tags[k])
            self.W[k] += ANCHOR * (self.W0[k] - self.W[k])
        self.da_mod = 0.0

    # ------------------------------------------------------------------ dynamics
    def step(self, bus, t0: int, W: int) -> dict[str, np.ndarray]:
        """Advance W ms. Everything is computed tick by tick in the same order whatever W is, so windowed and
        lockstep runs are bitwise identical (gate G6b); bus inputs are sparse, so they are summed by active line."""
        ctx = torch.from_numpy(bus.read("J.ctx", t0, W, 5))
        pvt = torch.from_numpy(bus.read("J.pvt", t0, W, 5))
        lim = torch.from_numpy(bus.read("B.bla", t0, W, 6)) if "B.bla" in bus.buf else torch.zeros(W, N_LIM)
        nm = bus.read("J.nm", t0, W, 6)                     # DA, NE, 5-HT, ACh concentrations
        o = self.off
        sl = {k: slice(*o[k]) for k in o}
        out_gpi = np.zeros((W, N_CH * I), np.float32)
        out_vs = np.zeros((W, N_VS), np.float32)
        # this turn's excitability, and the flat prior of the loops without a habit prior (both Phase B only)
        excite = BETA_G * self.gumbel.repeat_interleave(M) + FLAT_D1 * self.flat
        with torch.no_grad():
            for i in range(W):
                tick = t0 + i
                # running means: an exact average over the first ticks, then an EMA with TONIC_TAU_MS
                self.nm_ticks += 1
                k = 1.0 / min(self.nm_ticks, TONIC_TAU_MS)
                self.da_tonic += (float(nm[i, 0]) - self.da_tonic) * k
                self.ht_tonic += (float(nm[i, 2]) - self.ht_tonic) * k
                warm = self.nm_ticks >= WARMUP_MS               # no learning until the tonic level is known
                da = min(1.0, self.da_signal(float(nm[i, 0]))) if warm else 0.0      # (>= -1: no dopamine at all)
                ht = float(nm[i, 2]) / max(self.ht_tonic, 1e-3) - 1.0 if warm else 0.0
                self.da_mod += (da - self.da_mod) / DA_MOD_TAU_MS
                active = torch.nonzero(ctx[i]).flatten()
                sparse = lambda w: w[active].sum(0)        # noqa: E731
                lim_on = torch.nonzero(lim[i]).flatten()
                hs = self.habit.step()
                self.habit_trace = self.habit_trace * _K20 + hs
                I_in = self.tonic + torch.from_numpy(self.streams.normal("bg_noise", tick, (self.n,))) * self.noise_scale
                d1 = (sparse(self.W["ctx_d1"]) + self.W["lim_d1"][lim_on].sum(0)
                      + self.kappa_h * (self.habit_trace @ self.w_habit))
                # the excitability shows in the response to the habit input, so it ramps with the habit trace
                ramp = 1.0 - math.exp(-max(0, tick - self.phase_b_tick) / 20.0) if self.in_phase_b else 0.0
                # dopamine is a gain on the D1 cells' whole drive, excitability and flat prior included: it sets the
                # race's vigour but not its temperature, so the race keeps sampling argmax(log p + Gumbel) at any
                # dopamine level
                d1 = (d1 + ramp * excite) * (1 + 0.5 * self.da_mod) - self.nogo_d1
                I_in[sl["d1"]] += d1
                I_in[sl["d2"]] += sparse(self.W["ctx_d2"]) * (1 - 0.4 * self.da_mod) + 0.03 * ht
                I_in[sl["fsi"]] += sparse(self.w_ctx_fsi)
                I_in[sl["vs"]] += sparse(self.W["ctx_vs"]) + pvt[i] @ self.w_pvt_vs + self.w_lim_vs[lim_on].sum(0)
                # hyperdirect pathway: each loop's pooled cortical salience raises its STN (diffusely raising GPi)
                loop_sal = self.loop_onehot.T @ d1.reshape(N_CH, M).mean(1)
                I_in[sl["stn"]] += HYPER * (self.loop_onehot @ loop_sal).repeat_interleave(S)
                I_in[sl["gpi"]] += self.nogo
                s = self.pop(I_in + (self.RT @ self.prev[:, None])[:, 0])
                self.prev = s
                out_gpi[i] = s[sl["gpi"]].numpy()
                out_vs[i] = s[sl["vs"]].numpy()
                # eligibility: 20-ms presynaptic cortical trace x postsynaptic spikes (spiking columns only)
                self.ctx_trace = self.ctx_trace * _K20 + ctx[i]
                self.lim_trace = self.lim_trace * _K20 + lim[i]
                for k, post, trace in (("ctx_d1", "d1", self.ctx_trace), ("ctx_d2", "d2", self.ctx_trace),
                                       ("ctx_vs", "vs", self.ctx_trace), ("lim_d1", "d1", self.lim_trace)):
                    cols = torch.nonzero(s[sl[post]]).flatten()
                    if len(cols):
                        self.elig[k][:, cols] += trace[:, None]
                # dopamine bursts and dips pay tagged synapses: D1 and VS up, D2 down for a burst; the reverse for a
                # dip (Shen 2008), so an omitted reward unlearns what predicted it
                delta = self.da_mod if self.da_mod >= PHASIC else da if da <= -PAUSE else 0.0
                if delta:
                    for k, sign in (("ctx_d1", 1.0), ("ctx_d2", -1.0), ("ctx_vs", 1.0), ("lim_d1", 1.0)):
                        self.W[k] += (ETA * delta * sign) * self.tag_gain[k]
                        self.W[k].clamp_(0.0, 0.05)
        return {"S.gpi": out_gpi, "S.vs": out_vs}

    # ------------------------------------------------------------------ persistence
    def state_dict(self) -> dict:
        return {
            "streams": self.streams.state_dict(), "habit": self.habit.state_dict(),
            "v": self.pop.v.numpy().copy(), "prev": self.prev.numpy().copy(),
            "W": {k: v.numpy().copy() for k, v in self.W.items()},
            "elig": {k: v.numpy().copy() for k, v in self.elig.items()},
            "tags": {k: v.numpy().copy() for k, v in self.tags.items()},
            "nogo": self.nogo.numpy().copy(), "nogo_d1": self.nogo_d1.numpy().copy(),
            "gumbel": self.gumbel.numpy().copy(),
            "habit_trace": self.habit_trace.numpy().copy(), "ctx_trace": self.ctx_trace.numpy().copy(),
            "lim_trace": self.lim_trace.numpy().copy(),
            "scalars": {"kappa_h": np.array(self.kappa_h, np.float64), "da_tonic": np.array(self.da_tonic, np.float64),
                        "ht_tonic": np.array(self.ht_tonic, np.float64), "da_mod": np.array(self.da_mod, np.float64),
                        "in_phase_b": np.array(self.in_phase_b, np.bool_),
                        "nm_ticks": np.array(self.nm_ticks, np.int64),
                        "phase_b_tick": np.array(self.phase_b_tick, np.int64)},
        }

    def check_state_dict(self, d: dict) -> None:
        """Every array of the basal ganglia and the habit cortex, at its shape, and every scalar by name, of its kind
        (a tick count a whole number, the phase flag a bool)."""
        want = self.state_dict()
        check_shapes("basal ganglia", d, shapes_of(want))
        check_streams("basal ganglia", d["streams"], "sj")
        check_scalars("basal ganglia.scalars", d["scalars"], want["scalars"])

    def load_state_dict(self, d: dict) -> None:
        self.check_state_dict(d)
        t = lambda a: torch.from_numpy(np.array(a, np.float32))  # noqa: E731
        self.streams = Streams.from_state(d["streams"])
        self.habit.streams = self.streams
        self.habit.load_state_dict(d["habit"])
        self.pop.v, self.prev = t(d["v"]), t(d["prev"])
        self.W = {k: t(v) for k, v in d["W"].items()}
        self.elig = {k: t(v) for k, v in d["elig"].items()}
        self.tags = {k: t(v) for k, v in d["tags"].items()}
        self.tag_gain = {k: torch.tanh(K_TAG * v) for k, v in self.tags.items()}
        self.nogo, self.nogo_d1, self.gumbel = t(d["nogo"]), t(d["nogo_d1"]), t(d["gumbel"])
        self.habit_trace, self.ctx_trace = t(d["habit_trace"]), t(d["ctx_trace"])
        self.lim_trace = t(d["lim_trace"])
        sc = d["scalars"]
        self.kappa_h, self.da_tonic, self.ht_tonic = float(sc["kappa_h"]), float(sc["da_tonic"]), float(sc["ht_tonic"])
        self.da_mod, self.in_phase_b = float(sc["da_mod"]), bool(sc["in_phase_b"])
        self.nm_ticks, self.phase_b_tick = int(sc["nm_ticks"]), int(sc["phase_b_tick"])
