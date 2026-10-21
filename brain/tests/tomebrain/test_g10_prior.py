"""Gate G10: the network keeps the Director's pacing as its prior.

G10a  the basal-ganglia race samples the habit prior: the winner is argmax(log p + Gumbel) in >= 85% of trials,
      the chosen beat's mean log p matches exact sampling within 0.15, fallback <= 5%, no forbidden commit.
      Mutant: without the Gumbel excitability the race just takes the argmax.
G10b  pacing at initialisation: KL(network beat histogram || teacher) <= 0.15.
Also: the real brain is window-size invariant (W = 4 equals W = 1 bitwise, the mind read from it included), and
TomeNetEngine plays and reloads.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")

from voidlit.cortex.features import BEATS  # noqa: E402
from voidlit.cortex.teacher import sample_states, teacher_distribution  # noqa: E402
from voidlit.tomebrain.brain import TomeBrain  # noqa: E402
from voidlit.tomebrain.sjpart import basal_ganglia as BG  # noqa: E402

PACE = list(BG.LOOPS["PACE"])


def race_trials(n, seed=0, masks=False, beta=None, monkeypatch=None):
    if beta is not None:
        monkeypatch.setattr(BG, "BETA_G", beta)
    brain = TomeBrain(seed)
    X = sample_states(n, torch.Generator().manual_seed(100 + seed)).numpy()
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        allowed = None
        if masks:
            allowed = {b for b in PACE if rng.random() < 0.7} or {"calm"}
        brain.set_input(director_features=X[i])
        brain.phase_a()
        out = brain.phase_b(X[i], allowed)
        g = brain.sj.streams.gumbel("bg_gumbel", brain.turn - 1, BG.N_CH)[:8]
        logp = np.log(out.prior + 1e-9)
        ok = np.array([allowed is None or b in allowed for b in PACE])
        score = np.where(ok, logp + g, -np.inf)
        rows.append((out.pace.channel, PACE[int(np.argmax(score))], out.pace.fallback, allowed, out.prior, ok))
    return rows


def _stats(rows):
    hits = np.mean([c == best for c, best, *_ in rows])
    fb = np.mean([f for _, _, f, *_ in rows])
    lp_chosen = np.mean([np.log(p[PACE.index(c)]) for c, _, _, _, p, _ in rows])
    exact = np.mean([float((p[ok] / p[ok].sum() * np.log(p[ok])).sum()) for *_, p, ok in rows])
    return hits, fb, lp_chosen, exact


@pytest.mark.slow
def test_g10a_race_samples_the_prior():
    rows = race_trials(150)
    hits, fb, lp_chosen, exact = _stats(rows)
    assert hits >= 0.85, hits
    assert fb <= 0.05, fb
    assert abs(lp_chosen - exact) < 0.15, (lp_chosen, exact)


@pytest.mark.slow
def test_g10a_masks_are_never_violated():
    rows = race_trials(60, seed=1, masks=True)
    assert all(c in allowed for c, _, _, allowed, *_ in rows)
    hits, fb, *_ = _stats(rows)
    assert hits >= 0.8 and fb <= 0.1


@pytest.mark.slow
def test_g10a_mutant_without_gumbel_takes_the_argmax(monkeypatch):
    rows = race_trials(60, beta=0.0, monkeypatch=monkeypatch)
    _, _, lp_chosen, exact = _stats(rows)
    assert lp_chosen - exact > 0.15


@pytest.mark.slow
def test_g10b_pacing_matches_the_teacher():
    n = 300
    rows = race_trials(n, seed=2)
    hist = np.array([sum(c == b for c, *_ in rows) for b in PACE], float) / n
    X = sample_states(n, torch.Generator().manual_seed(102))
    teacher = teacher_distribution(X).mean(0).numpy()
    kl = float(np.sum(np.where(hist > 0, hist * np.log(hist / teacher), 0.0)))
    assert kl <= 0.15, (kl, hist.round(3), teacher.round(3))


def test_the_real_brain_is_window_size_invariant():
    """G6b on the real brain, with the amygdala's stimuli and the P fact slots driven: the network and the mind read
    from it (needs, drive, reward, the Tome's modifiers) agree bitwise every turn. The Mind pays no reward before its
    fifth turn, so the last two turns receive one by hand: a burst, then a pause that outlasts Phase A (and the
    pacemakers' return over the quiet)."""
    X = sample_states(4, torch.Generator().manual_seed(5)).numpy()
    us = [{"gift": 1.0}, {"harm": 1.0, "lie": 0.5}, {}, {"loss": 1.0, "bearer_near_death": 0.6}]
    rewards = [None, None, 0.12, -0.09]
    a, b = TomeBrain(3, W=4), TomeBrain(3, W=1)
    for i, x in enumerate(X):
        facts = np.roll(np.linspace(0.2, 0.9, 8, dtype=np.float32), i)
        outs = []
        for brain in (a, b):
            if rewards[i] is not None:
                brain.mind.reward = rewards[i]
            brain.set_us(us[i])
            brain.set_input(director_features=x, senses=np.full(8, 0.3, np.float32),
                            body=np.array([0.7, 0.9, 0.2], np.float32), facts=facts)
            brain.phase_a()
            outs.append(brain.phase_b(x))
        assert outs[0].pace == outs[1].pace and outs[0].ms == outs[1].ms
        assert np.array_equal(outs[0].facts, outs[1].facts)
        assert np.array_equal(a.jax.probe_sum, b.jax.probe_sum)
        assert a.mind.state_dict() == b.mind.state_dict(), i
        assert a.digest() == b.digest()


def test_net_engine_plays_saves_and_resumes(tmp_path):
    from voidlit.playtest import bot_command
    from voidlit.tome.net_engine import TomeNetEngine

    a = TomeNetEngine(seed=4)
    a.start()
    for _ in range(3):
        f = a.turn(bot_command(a))
        assert f.debug["beat"] in [b.value for b in BEATS]
    a.save(tmp_path / "slot")
    b = TomeNetEngine(seed=77)
    b.load(tmp_path / "slot")
    assert a.brain.digest() == b.brain.digest()
    for _ in range(2):
        cmd = bot_command(a)
        fa, fb = a.turn(cmd), b.turn(cmd)
        assert fa.to_dict(include_debug=True) == fb.to_dict(include_debug=True)
