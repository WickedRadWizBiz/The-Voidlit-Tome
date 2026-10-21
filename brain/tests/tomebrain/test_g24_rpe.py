"""G24: dopamine is a reward-prediction error.

The Tome's intrinsic reward (drive reduction, design §8.2) reaches the VTA as a lateral-hypothalamic current at the
next Phase A. An unpredicted gift must burst the VTA far above the same brain's turn without the gift, and the burst
must teach (the slow dopamine crosses PHASIC) while eventless turns teach nothing. A negative reward must dip it: the
gift's fading pays a negative reward into the following Phase A, which lowers the VTA below what the very same brain
state does there with no reward (mutant: a reward clipped at 0 never dips).

Deferred, not yet met (design §15.4): the Schultz transfer clause (after 10 predicted gifts the burst at the gift
falls >= 50%; omitting a predicted gift dips) and its VS -> RMTg-cut mutant. Measured on seed 11 with 12 cue -> gift
pairings: the critic does learn the cue (VS at the reward 9 -> 17 Hz, ctx -> VS weights +150%), but VS -> RMTg -> VTA
is too weak to subtract that prediction from a 150-200 pA reward current. The burst fell 39 -> 35 Hz intact and
39 -> 33 Hz with the edge cut (the reward itself shrinks as the bond settles), and 5x the edge's weight still gave
no transfer. It needs the critic's path to the VTA recalibrated, which these gates will then hold.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")

from voidlit.cortex.teacher import sample_states  # noqa: E402
from voidlit.tomebrain import persist  # noqa: E402
from voidlit.tomebrain.brain import REWARD_MS, TomeBrain  # noqa: E402
from voidlit.tomebrain.sjpart import basal_ganglia as BG  # noqa: E402

pytestmark = pytest.mark.slow


def states() -> np.ndarray:
    return sample_states(12, torch.Generator().manual_seed(9)).numpy()


def window_rate(brain) -> float:
    """VTA rate (Hz) over the reward window at the start of the last Phase A."""
    return float(brain.vta_phase_a[:REWARD_MS].mean() * 1000.0)


class Recorder:
    """A TomeBrain whose basal ganglia report the slow dopamine's extremes per turn."""

    def __init__(self, seed: int) -> None:
        self.brain = TomeBrain(seed)
        self.w0 = {k: v.clone() for k, v in self.brain.sj.W.items()}
        self.da: list[float] = []
        step = self.brain.sj.step

        def recorded(bus, t0, W):
            out = step(bus, t0, W)
            self.da.append(self.brain.sj.da_mod)
            return out
        self.brain.sj.step = recorded

    def play(self, features, us=None) -> dict:
        """One turn. The reward is read before Phase B computes the next one: it is what this Phase A received."""
        brain = self.brain
        self.da.clear()
        r_in = brain.mind.reward
        brain.set_us({})
        brain.set_input(director_features=features)
        brain.phase_a()
        rate = window_rate(brain)
        brain.set_us(us or {})
        brain.phase_b(features)
        return dict(vta=rate, reward=r_in, da_max=max(self.da), da_min=min(self.da))

    def learned(self) -> bool:
        return any(not torch.equal(self.w0[k], self.brain.sj.W[k]) for k in self.w0)


def protocol(gift: bool) -> tuple[list[dict], bool]:
    rec, X = Recorder(11), states()
    turns = [rec.play(X[i], {"gift": 1.0} if gift and i == 6 else None) for i in range(9)]
    return turns, rec.learned()


@pytest.fixture(scope="module")
def gift_and_control():
    return protocol(True), protocol(False)        # the same brain, with and without the gift


def test_g24_an_unpredicted_gift_bursts_far_above_the_same_turn_without_it(gift_and_control):
    (gift, _), (control, _) = gift_and_control
    base = np.mean([t["vta"] for t in gift[:6]])
    burst = gift[7]
    assert burst["reward"] > 0.05, burst                      # the gift's reward is what reached this Phase A
    assert burst["vta"] >= base + 5.0, (burst, base)
    assert burst["vta"] >= control[7]["vta"] + 20.0, (burst, control[7])


def test_g24_the_burst_teaches_and_eventless_turns_teach_nothing(gift_and_control):
    """With no stimulus the VTA stays near tonic and the slow dopamine below PHASIC on every turn, so no weight
    moves; the gift's burst crosses PHASIC and pays the tags."""
    (gift, gift_learned), (control, control_learned) = gift_and_control
    assert gift[7]["da_max"] >= BG.PHASIC and gift_learned
    assert all(t["vta"] < 20.0 for t in control), [t["vta"] for t in control]
    assert all(max(t["da_max"], -t["da_min"]) < BG.PHASIC for t in control)
    assert not control_learned


def paired_dip(seed: int, path) -> dict:
    """The gift protocol up to the dip turn, whose Phase A then runs three times from one saved state: with the reward
    the brain computed after the burst turn (what this Phase A receives), with no reward, and in the mutant brain that
    injects only positive reward. VTA spikes over Phase A and the dopamine concentration at its end."""
    rec, X = Recorder(seed), states()
    for i in range(8):
        rec.play(X[i], {"gift": 1.0} if i == 6 else None)
    brain = rec.brain
    r = brain.mind.reward
    brain.set_us({})
    brain.set_input(director_features=X[8])
    snap = persist.save(brain.cosim, path)
    ext = brain.jax.ext_fn

    def clipped(t0, W):
        brain.mind.reward = max(0.0, r)
        try:
            return ext(t0, W)
        finally:
            brain.mind.reward = r
    out = {"reward": r}
    for name, value, fn in (("real", r, ext), ("zero", 0.0, ext), ("clipped", r, clipped)):
        persist.load(brain.cosim, snap)
        brain.mind.reward, brain.jax.ext_fn = value, fn
        brain.phase_a()
        out[name] = (int(brain.vta_phase_a.sum()), float(np.asarray(brain.jax.state["nm"])[0]))
    return out


@pytest.fixture(scope="module")
def fading(tmp_path_factory):
    d = tmp_path_factory.mktemp("g24")
    return [paired_dip(seed, d / f"{seed}.pt") for seed in range(11, 17)]


def lowered(runs: list[dict], name: str = "real") -> int:
    """VTA spikes the reward removed from the dip turn's Phase A, summed over the seeds: after a burst the 24 cells
    fire only a few times in 60 ms, so one seed alone is mostly shot noise."""
    return sum(r["zero"][0] - r[name][0] for r in runs)


def test_g24_the_gift_fading_dips_the_vta_below_the_same_state_without_it(fading):
    for run in fading:
        assert run["reward"] < 0.0, run               # read when it was injected: the dip turn's Phase A
        assert run["real"][0] <= run["zero"][0] and run["real"][1] <= run["zero"][1], run
    assert lowered(fading) >= 4, fading


def test_g24_mutant_a_reward_clipped_at_zero_never_dips(fading):
    """A VTA that only ever receives positive reward runs the dip turn exactly as with no reward: the test above
    fails."""
    assert lowered(fading, "clipped") == 0 and all(r["clipped"] == r["zero"] for r in fading)
