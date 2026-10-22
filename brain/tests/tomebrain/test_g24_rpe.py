"""G24: dopamine is a reward-prediction error.

The Tome's intrinsic reward (drive reduction, design §8.2) reaches the VTA as a lateral-hypothalamic current at the
next Phase A: a positive one bursts it for 50 ms, a negative one holds it silent, longer the more negative it is. An
unpredicted gift must burst the VTA far above the same brain's turn without the gift, and the burst must teach (the
slow dopamine crosses PHASIC) within its own turn, while eventless turns teach nothing. The gift's fading is a
negative reward: it must take the dopamine well below what the very same brain state does with no reward (mutant:
the old negative current, proportional to the reward for 50 ms, barely does). And a realistic punishment must teach
in play: tagged D1 and critic synapses weaken and D2 synapses strengthen, and the end of the pause pays nothing back
on the next turn: the quiet returns the VTA's pacemakers to where the pause found them (without that they rebound
together at the next Phase A) and the dopamine concentration to its resting level (without that a pause cut short by
the turn's end goes on, and dips, in the next one). A starving brain, whose VTA nearly falls silent, teaches nothing
without an event.

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
from voidlit.tomebrain import brain as B  # noqa: E402
from voidlit.tomebrain import persist  # noqa: E402
from voidlit.tomebrain.brain import REWARD_MS, TomeBrain  # noqa: E402
from voidlit.tomebrain.sjpart import basal_ganglia as BG  # noqa: E402

pytestmark = pytest.mark.slow
PAID = ("ctx_d1", "ctx_vs", "ctx_d2")     # the cortical inputs that carry tags in the brain
SIGN = {"ctx_d1": 1.0, "ctx_vs": 1.0, "ctx_d2": -1.0}    # what a burst does to each, and a dip the reverse


def states() -> np.ndarray:
    return sample_states(12, torch.Generator().manual_seed(9)).numpy()


def window_rate(brain) -> float:
    """VTA rate (Hz) over the reward window at the start of the last Phase A."""
    return float(brain.vta_phase_a[:REWARD_MS].mean() * 1000.0)


class Recorder:
    """A TomeBrain whose basal ganglia report, per turn, the slow dopamine's extremes and the lowest dopamine
    concentration the striatum read (relative to tonic)."""

    def __init__(self, seed: int) -> None:
        self.brain = TomeBrain(seed)
        self.w0 = {k: v.clone() for k, v in self.brain.sj.W.items()}
        self.da: list[float] = []
        self.conc: list[float] = []
        step = self.brain.sj.step

        def recorded(bus, t0, W):
            out = step(bus, t0, W)
            sj = self.brain.sj
            self.da.append(sj.da_mod)
            if sj.nm_ticks >= BG.WARMUP_MS:       # as the striatum reads it once the tonic level is known
                self.conc.append(sj.da_signal(float(bus.read("J.nm", t0, W, 6)[:, 0].min())))
            return out
        self.brain.sj.step = recorded

    def play(self, features, us=None, reward=None) -> dict:
        """One turn. The reward is read before Phase B computes the next one: it is what this Phase A received."""
        brain = self.brain
        self.da.clear()
        self.conc.clear()
        if reward is not None:
            brain.mind.reward = reward
        r_in = brain.mind.reward
        w = {k: brain.sj.W[k].clone() for k in PAID}
        brain.set_us({})
        brain.set_input(director_features=features)
        brain.phase_a()
        rate = window_rate(brain)
        brain.set_us(us or {})
        brain.phase_b(features)
        # the change of each paid matrix, the anchor's pull toward its initial weights taken out
        dw = {k: float((brain.sj.W[k] - w[k] - BG.ANCHOR * (brain.sj.W0[k] - w[k])).mean()) for k in PAID}
        return dict(vta=rate, reward=r_in, da_max=max(self.da), da_min=min(self.da),
                    conc_min=min(self.conc, default=0.0), dw=dw)

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
    """With no stimulus the VTA stays near tonic, the slow dopamine below PHASIC and the concentration above the
    dip's level on every turn, so no weight moves; the gift's burst crosses PHASIC and pays the tags."""
    (gift, gift_learned), (control, control_learned) = gift_and_control
    assert gift[7]["da_max"] >= BG.PHASIC and gift_learned
    assert all(t["vta"] < 20.0 for t in control), [t["vta"] for t in control]
    assert all(t["da_max"] < BG.PHASIC and t["conc_min"] > -BG.PAUSE for t in control), control
    assert not control_learned


def test_g24_the_burst_ends_with_its_turn(gift_and_control):
    """The quiet's 20 s end the transient: the turn after the burst starts from tonic dopamine. Frozen across the
    quiet, the burst's tail went on paying in that turn, onto tags that now held the burst turn's own activity."""
    (gift, _), _ = gift_and_control
    assert gift[8]["da_max"] < BG.PHASIC, gift[8]


def fading_dip(seed: int, path) -> dict:
    """The gift protocol up to the dip turn, whose turn then runs three times from one saved state: with the reward
    the brain computed after the burst turn (what this Phase A receives), with no reward, and with that reward
    injected the old way (REWARD_PA x r for the first 50 ms). The slow dopamine's minimum over the turn for each."""
    rec, X = Recorder(seed), states()
    for i in range(8):
        rec.play(X[i], {"gift": 1.0} if i == 6 else None)
    brain = rec.brain
    r = brain.mind.reward
    brain.set_us({})
    brain.set_input(director_features=X[8])
    snap, mind, turn = persist.save(brain.cosim, path), brain.mind.state_dict(), brain.turn
    out = {"reward": r}
    for name, value, old_way in (("real", r, False), ("zero", 0.0, False), ("old", r, True)):
        persist.load(brain.cosim, snap)
        brain.mind.load_state_dict(mind)
        brain.turn = turn
        with pytest.MonkeyPatch.context() as mp:
            if old_way:
                mp.setattr(B, "PAUSE_PA", -B.REWARD_PA * r)
                mp.setattr(B, "PAUSE_MS_PER_UNIT", B.REWARD_MS / -r)
            out[name] = rec.play(X[8], reward=value)["da_min"]
    return out


@pytest.fixture(scope="module")
def fading(tmp_path_factory):
    d = tmp_path_factory.mktemp("g24")
    return [fading_dip(seed, d / f"{seed}.pt") for seed in range(11, 17)]


def deepened(runs: list[dict], name: str) -> float:
    """How much lower the slow dopamine fell with the reward than in the same state without it, summed over the
    seeds (after a burst the VTA is quiet anyway, so one seed alone is mostly its noise)."""
    return sum(r["zero"] - r[name] for r in runs)


def test_g24_the_gift_fading_dips_the_dopamine_below_the_same_state_without_it(fading):
    """Measured: 0.42 over the six seeds; the old 50-ms current gave 0.08."""
    assert all(run["reward"] < 0.0 for run in fading), fading       # the fading is a negative prediction error
    assert deepened(fading, "real") >= 0.25, fading


def test_g24_mutant_the_old_50ms_current_barely_dips(fading):
    """The negative reward as it was injected before (a current proportional to it, for the reward's 50 ms): the
    same fading leaves the slow dopamine almost where no reward leaves it, so the test above fails."""
    assert deepened(fading, "old") < 0.25, fading


@pytest.fixture(scope="module")
def punishments(tmp_path_factory):
    """Per seed, ten eventless turns, then from that one state: a turn that receives a reward r and the turn after it,
    which receives none (so whatever it learns is the punishment's leftover), for r = 0, -0.06, -0.1 and -1, and for
    -0.1 injected the old way ("old")."""
    d, X, out = tmp_path_factory.mktemp("g24p"), states(), {}
    for seed in (4, 5, 7):
        rec = Recorder(seed)
        for i in range(10):
            rec.play(X[i])
        brain = rec.brain
        snap, mind, turn = persist.save(brain.cosim, d / f"{seed}.pt"), brain.mind.state_dict(), brain.turn
        for r, old_way in ((0.0, False), (-0.06, False), (-0.1, False), (-1.0, False), (-0.1, True)):
            persist.load(brain.cosim, snap)
            brain.mind.load_state_dict(mind)
            brain.turn = turn
            with pytest.MonkeyPatch.context() as mp:
                if old_way:
                    mp.setattr(B, "PAUSE_PA", -B.REWARD_PA * r)
                    mp.setattr(B, "PAUSE_MS_PER_UNIT", B.REWARD_MS / -r)
                dip = rec.play(X[10], reward=r)
            out[seed, r, "old" if old_way else "pause"] = dip, rec.play(X[11], reward=0.0)
    return out


def test_g24_a_punishment_teaches_in_play(punishments):
    """A negative reward of -0.06 or -0.1 (harm, a dying light, a gift's fading) pauses the VTA long enough that the
    dopamine concentration halves in the turn it arrives, however short: tagged D1 and critic synapses weaken, D2
    synapses strengthen (Shen 2008; the D2 MSNs were silent before, so they carried no tags)."""
    for (seed, r, how), (dip, _) in punishments.items():
        if how == "pause" and -1.0 < r < 0.0:
            assert dip["conc_min"] <= -BG.PAUSE, (seed, r, dip)
            assert all(SIGN[k] * v < -1e-7 for k, v in dip["dw"].items()), (seed, r, dip)


def test_g24_the_pause_pays_nothing_back_on_the_next_turn(punishments):
    """The turn after a punishment learns nothing, and its slow dopamine peaks no higher than after no reward at all.
    Without the pacemakers' return over the quiet, a -0.1 pause's rebound took the next Phase A's VTA to 7-13 Hz and
    the slow dopamine to +0.16..+0.27 (+0.00..+0.04 after no reward; seeds 4, 5, 7, 8); without the concentration's
    return to rest, the pause went on into the next turn and dipped there (to -0.58..-0.68 on seeds 5 and 7)."""
    for (seed, r, how), (_, after) in punishments.items():
        if how == "pause" and r < 0.0:
            control = punishments[seed, 0.0, "pause"][1]
            assert all(abs(v) < 1e-9 for v in after["dw"].values()), (seed, r, after)
            assert after["da_max"] < control["da_max"] + 0.1, (seed, r, after, control)


def test_g24_mutant_the_old_50ms_current_teaches_no_dip(punishments):
    """Injected the old way, the same punishment leaves every paid weight where the anchor puts it: the test above
    detects that a dip does not teach."""
    for (seed, r, how), (dip, _) in punishments.items():
        if how == "old":
            assert dip["conc_min"] > -BG.PAUSE and all(abs(v) < 1e-9 for v in dip["dw"].values()), (seed, dip)


def test_g24_a_starving_brain_teaches_nothing_without_an_event():
    """Twelve fed turns, then thirty at Lumen 0 with no stimulus: the VTA falls to under 1 Hz, and the striatum's tonic
    level follows it over the quiet instead of reading every starving turn as a pause (before: dips on 19 of the 30
    turns, some with a positive reward)."""
    brain, X = TomeBrain(0), sample_states(42, torch.Generator().manual_seed(3)).numpy()
    learned = []
    for i, x in enumerate(X):
        lumen = 6 if i < 12 else 0
        w = brain.sj.W["ctx_d1"].clone()
        brain.set_us({})
        brain.set_input(director_features=x, body=np.array([lumen / 6, 1.0, 0.0], np.float32),
                        glucose=0.15 + 0.85 * lumen / 6)
        brain.phase_a()
        brain.phase_b(x)
        dw = brain.sj.W["ctx_d1"] - w - BG.ANCHOR * (brain.sj.W0["ctx_d1"] - w)
        if i >= 12 and float(dw.abs().max()) > 1e-8:          # (float32 rounding of the anchor's pull: about 1e-10)
            learned.append(i)
    assert not learned, learned
