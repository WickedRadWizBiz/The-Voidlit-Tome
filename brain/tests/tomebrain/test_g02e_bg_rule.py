"""G2e: the basal ganglia's three-factor rule pays tagged synapses for dopamine bursts and dips (design §6.11).

A burst potentiates the tagged D1 and VS-critic synapses and depresses the tagged D2 ones; a dip does the reverse
(Shen 2008), so a reward that was predicted and then omitted (the VTA pauses where it was due) unlearns what
predicted it. The sparse VTA's shot noise at tonic teaches nothing, and untagged synapses never move.
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
    one rate per ms: 0 is a pause, 40 Hz a burst), then pacemaking again."""
    rng = np.random.default_rng(seed)
    n = WARM_MS + len(event_hz) + TAIL_MS
    spikes = np.zeros(n)
    period = 1000.0 / TONIC_HZ
    for _ in range(VTA_N):
        t = rng.uniform(0.0, period)
        while t < n:
            spikes[int(t)] += 1
            t += period * (1.0 + 0.1 * rng.standard_normal())
    spikes[WARM_MS:WARM_MS + len(event_hz)] = rng.poisson(VTA_N * np.asarray(event_hz) * 1e-3)
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
    da = dopamine(event_hz)
    bus = SpikeBus({"J.ctx": BG.N_CTX, "J.pvt": BG.N_PVT, "J.nm": 4, "B.bla": BG.N_LIM}, W)
    silent = {src: np.zeros((W, n), np.float32) for src, n in (("J.ctx", BG.N_CTX), ("J.pvt", BG.N_PVT),
                                                               ("B.bla", BG.N_LIM))}
    before = None
    for t in range(0, len(da) - len(da) % W, W):
        if t == WARM_MS:
            before = {k: v.clone() for k, v in sj.W.items()}
        sj.step(bus, t, W)
        nm = np.zeros((W, 4), np.float32)
        nm[:, 0], nm[:, 2] = da[t:t + W], 1.0
        for src, rows in silent.items():
            bus.write(src, t, rows)
        bus.write("J.nm", t, nm)
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


def test_g2e_mutant_a_threshold_dips_cannot_reach_teaches_nothing_on_omission(monkeypatch):
    """The old gate (a 50% excursion of the slow dopamine) is out of a dip's reach: the same omission changes no
    weight, so the test above detects it."""
    monkeypatch.setattr(BG, "PHASIC", 0.5)
    before, after = run(tagged_partition(), np.zeros(250))
    assert all(torch.equal(before[k], after[k]) for k in before)
