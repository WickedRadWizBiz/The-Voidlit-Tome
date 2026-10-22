"""G2e: the basal ganglia's three-factor rule pays tagged synapses for dopamine bursts and dips (design §6.11).

A burst potentiates the tagged D1 and VS-critic synapses and depresses the tagged D2 ones; a dip does the reverse
(Shen 2008), so a reward that was predicted and then omitted (the VTA pauses where it was due) unlearns what
predicted it. A burst is read from the slow dopamine, a dip from the concentration, which a pause halves within a
short turn. The sparse VTA's shot noise at tonic teaches nothing, and untagged synapses never move.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("jax")

from voidlit.tomebrain.bus import SpikeBus  # noqa: E402
from voidlit.tomebrain.jaxpart import network as NW  # noqa: E402
from voidlit.tomebrain.sjpart import basal_ganglia as BG  # noqa: E402

W = 4
VTA_N, TONIC_HZ = 24, 4.5      # the JAX VTA's pacemakers idle at about 4.5 Hz in Phase A on eventless turns
DA_PER_SPIKE = 0.05            # its volume transmission (jaxpart.network: dM/dt = -M/tau + 0.05 spikes)
WARM_MS, TAIL_MS = 1000, 600
TAGGED = {"ctx_d1": slice(0, 40), "ctx_d2": slice(0, 40), "ctx_vs": slice(0, 40), "lim_d1": slice(0, 12)}
SIGN = {"ctx_d1": 1.0, "ctx_d2": -1.0, "ctx_vs": 1.0, "lim_d1": 1.0}     # what a burst does to each
DECAY = math.exp(-1.0 / NW.NEUROMOD_TAU[0])
TONIC_DA = DA_PER_SPIKE * VTA_N * TONIC_HZ * 1e-3 / (1.0 - DECAY)       # the concentration the pacemakers sustain


def dopamine(event_hz: np.ndarray, seed: int = 0) -> np.ndarray:
    """DA concentration per ms: pacemaking VTA cells (4.5 Hz, ISIs jittered by 10%), then ``event_hz`` (Poisson,
    one rate per ms: 0 is a pause, 40 Hz a burst), then pacemaking again. A pause holds every cell below threshold:
    released, each fires again 15-30 ms later (as the brain's VTA does) and paces on from there."""
    rng = np.random.default_rng(seed)
    n = WARM_MS + len(event_hz) + TAIL_MS
    lo, hi = WARM_MS, WARM_MS + len(event_hz)
    spikes = np.zeros(n)
    period = 1000.0 / TONIC_HZ
    for _ in range(VTA_N):
        t = rng.uniform(0.0, period)
        while t < n:
            if lo <= t < hi and not np.any(event_hz):
                t = hi + rng.uniform(15.0, 30.0)
            spikes[int(t)] += 1
            t += period * (1.0 + 0.1 * rng.standard_normal())
    spikes[lo:hi] = rng.poisson(VTA_N * np.asarray(event_hz) * 1e-3)
    nm = TONIC_DA
    out = np.empty(n, np.float32)
    for t in range(n):
        nm = nm * DECAY + DA_PER_SPIKE * spikes[t]
        out[t] = nm
    return out


def tagged_partition() -> BG.SJPartition:
    """A partition whose cortical cue lines (and some amygdala lines) were active with the MSNs and the critic when
    the reward came last time: their synapses carry tags."""
    sj = BG.SJPartition(0)
    for k, rows in TAGGED.items():
        sj.elig[k][rows] = 5.0
    sj.end_turn(300.0)
    # a brain that has been alive a while: its tonic dopamine and serotonin are known
    sj.nm_ticks, sj.da_tonic, sj.ht_tonic = int(BG.TONIC_TAU_MS), TONIC_DA, 1.0
    return sj


def run(sj: BG.SJPartition, event_hz: np.ndarray) -> tuple[dict, dict]:
    """A second of tonic dopamine, then ``event_hz`` and 600 ms of tonic after it, with a silent cortex (so only
    what is already tagged can be paid). Returns the weights before and after the event."""
    return play(sj, dopamine(event_hz), WARM_MS)


def play(sj: BG.SJPartition, da: np.ndarray, from_ms: int = 2 * W, ctx_hz: float = 0.0) -> tuple[dict, dict]:
    """The partition reads the dopamine concentration ``da`` (per ms) with a silent cortex, or one whose lines fire
    Poisson spikes at ``ctx_hz``. Returns the weights at ``from_ms`` and at the end."""
    bus = SpikeBus({"J.ctx": BG.N_CTX, "J.pvt": BG.N_PVT, "J.nm": 4, "B.bla": BG.N_LIM}, W)
    silent = {src: np.zeros((W, n), np.float32) for src, n in (("J.ctx", BG.N_CTX), ("J.pvt", BG.N_PVT),
                                                               ("B.bla", BG.N_LIM))}
    rng = np.random.default_rng(0)

    def publish(t):
        nm = np.zeros((W, 4), np.float32)
        nm[:, 0], nm[:, 2] = da[t:t + W], 1.0
        if ctx_hz:
            silent["J.ctx"] = (rng.random((W, BG.N_CTX)) < ctx_hz * 1e-3).astype(np.float32)
        for src, rows in silent.items():
            bus.write(src, t, rows)
        bus.write("J.nm", t, nm)

    before = None
    for t in range(0, 2 * W, W):            # the striatum has seen tonic dopamine before it first reads the bus
        publish(t)
    for t in range(2 * W, len(da) - len(da) % W, W):
        if t == from_ms:
            before = {k: v.clone() for k, v in sj.W.items()}
        sj.step(bus, t, W)
        publish(t)
    return before, {k: v.clone() for k, v in sj.W.items()}


def change(before: dict, after: dict) -> dict:
    """Mean weight change of each matrix's tagged rows, and the largest change anywhere else."""
    out = {}
    for k, rows in TAGGED.items():
        d = after[k] - before[k]
        rest = torch.ones(d.shape[0], dtype=torch.bool)
        rest[rows] = False
        out[k] = (float(d[rows].mean()), float(d[rest].abs().max()))
    return out


@pytest.fixture(scope="module")
def burst():
    sj = tagged_partition()
    w0 = {k: v.clone() for k, v in sj.W.items()}
    before, after = run(sj, np.full(50, 40.0))            # a gift's burst: 40 Hz for the reward's 50 ms
    return w0, before, after


def test_g2e_shot_noise_at_tonic_teaches_nothing(burst):
    w0, before, _ = burst
    assert all(torch.equal(w0[k], before[k]) for k in w0)


def test_g2e_a_burst_pays_tags_d1_and_critic_up_d2_down(burst):
    _, before, after = burst
    for k, (tagged, untagged) in change(before, after).items():
        assert SIGN[k] * tagged > 1e-5, (k, tagged)
        assert untagged == 0.0, k


def test_g2e_an_omitted_reward_dips_and_unlearns_what_predicted_it():
    """The VTA pauses for 250 ms where the predicted reward was due: tagged D1 and critic synapses weaken, D2
    strengthens."""
    before, after = run(tagged_partition(), np.zeros(250))
    for k, (tagged, untagged) in change(before, after).items():
        assert -SIGN[k] * tagged > 1e-5, (k, tagged)
        assert untagged == 0.0, k


def test_g2e_a_pause_within_a_short_turn_unlearns():
    """Phase B often commits at 124 ms, so a turn may last 184 ms. A 150-ms pause from the start of Phase A cannot
    take the slow dopamine (200 ms) 30% below tonic, even left to run on (it reaches about -0.27), but the
    concentration halves after 139 ms, and that is where a dip is read."""
    before, after = run(tagged_partition(), np.zeros(150))
    for k, (tagged, untagged) in change(before, after).items():
        assert -SIGN[k] * tagged > 1e-5, (k, tagged)
        assert untagged == 0.0, k


def test_g2e_a_pause_of_eventless_size_teaches_nothing():
    """A 90-ms pause (r = -0.035, twice the SD of the eventless reward noise): the concentration falls about 40%."""
    before, after = run(tagged_partition(), np.zeros(90))
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_g2e_mutant_a_dip_needing_an_empty_synapse_misses_the_short_pause(monkeypatch):
    """Read where a turn-length pause cannot reach, the dip teaches nothing: the test above detects the gate."""
    monkeypatch.setattr(BG, "PAUSE", 1.0)
    before, after = run(tagged_partition(), np.zeros(150))
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_g2e_the_d2_msns_fire_at_a_low_rate():
    """With no drive but the sparse cortical lines (about a twentieth of threshold) the D2 MSNs never fired in the
    brain, so their synapses carried no tags and no dip could strengthen them. Their tonic drive (the up-state the
    small network's cortex cannot sustain) has them fire at a low rate on the brain's cortical input (its 328 lines
    at about 4 Hz)."""
    sj, spikes = tagged_partition(), []
    fire = sj.pop.forward

    def counted(x):
        s = fire(x)
        spikes.append(float(s[slice(*sj.off["d2"])].sum()))
        return s
    sj.pop.forward = counted
    play(sj, dopamine(np.zeros(0)), ctx_hz=4.0)
    hz = sum(spikes) / (BG.N_CH * BG.M) / len(spikes) * 1000.0
    assert 0.1 < hz < 5.0, hz


STARVING_HZ = 0.6       # a starving brain's VTA: 0.85 Hz at rest at Lumen 0, 0-1.4 Hz in its turns


def level(hz: float) -> float:
    """The DA concentration a VTA pacing at ``hz`` sustains."""
    return DA_PER_SPIKE * VTA_N * hz * 1e-3 / (1.0 - DECAY)


def starving_dopamine(ms: int, seed: int = 0) -> np.ndarray:
    """DA concentration per ms from a VTA whose cells fire few, irregular (Poisson) spikes at STARVING_HZ."""
    spikes = np.random.default_rng(seed).poisson(VTA_N * STARVING_HZ * 1e-3, ms)
    nm, out = level(STARVING_HZ), np.empty(ms, np.float32)
    for t in range(ms):
        nm = nm * DECAY + DA_PER_SPIKE * spikes[t]
        out[t] = nm
    return out


def test_g2e_the_tonic_dopamine_integrates_the_quiet():
    """Over the quiet's 20 s the dopamine follows the VTA from its turn rate back to its resting rate (with the quiet's
    tau_post of 2 s) and the tonic level integrates it: the closed form agrees with the running mean stepped through
    every ms, and a fed tonic level goes nine tenths of the way to a starving VTA's resting level in one quiet."""
    sj = tagged_partition()
    da_turn, da_rest, tau_post = 1.3 * TONIC_DA, level(STARVING_HZ), NW.POST_TAU_MS
    tonic = sj.da_tonic
    for t in range(int(BG.QUIET_MS)):
        tonic += (da_rest + (da_turn - da_rest) * math.exp(-t / tau_post) - tonic) / BG.TONIC_TAU_MS
    sj.rest(da_turn, da_rest, 1.0, tau_post)
    assert sj.da_tonic == pytest.approx(tonic, rel=1e-3)
    assert sj.da_tonic - da_rest < 0.1 * (TONIC_DA - da_rest), (sj.da_tonic, da_rest)


@pytest.mark.parametrize("seed", range(2))
def test_g2e_a_starving_vta_teaches_nothing(seed):
    """At Lumen 0 the VTA paces at under 1 Hz. Once the tonic level has followed it over the quiet, its few, irregular
    spikes read as neither pauses nor bursts (each moves the concentration by a quarter of that level, so the
    striatum reads them against DA_FLOOR), and 3 s of them pay none of the tagged synapses."""
    sj = tagged_partition()
    sj.rest(TONIC_DA, level(STARVING_HZ), 1.0, NW.POST_TAU_MS)
    before, after = play(sj, starving_dopamine(3000, seed))
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_g2e_mutant_a_tonic_level_left_fed_reads_starvation_as_a_dip():
    """Integrated over turn ticks alone, the tonic level stays near where the fed brain left it for about 20 turns:
    the starving VTA reads as one long pause and unlearns every tagged synapse (in the brain, dips on 19 of 30 eventless
    starving turns). The test above detects a tonic level that does not follow the VTA's resting rate."""
    before, after = play(tagged_partition(), starving_dopamine(3000))
    for k, (tagged, _) in change(before, after).items():
        assert -SIGN[k] * tagged > 1e-5, (k, tagged)


def test_g2e_mutant_read_against_the_starving_tonic_alone_its_shot_noise_teaches(monkeypatch):
    """Without DA_FLOOR, the starving VTA's gaps between spikes read as pauses: the test above detects the floor."""
    monkeypatch.setattr(BG, "DA_FLOOR", 1e-3)
    sj = tagged_partition()
    sj.rest(TONIC_DA, level(STARVING_HZ), 1.0, NW.POST_TAU_MS)
    before, after = play(sj, starving_dopamine(3000))
    assert not all(torch.equal(before[k], after[k]) for k in before)
