import random

import torch

from voidlit.cortex import (BEATS, DRAWS, N_FEATURES, NARRATIVE, Beat, DirectorFeatures, teacher_distribution,
                            train_director)
from voidlit.cortex.teacher import sample_states


def test_features_are_unit_interval():
    f = DirectorFeatures(lumen=2.0, stress=-1.0)  # clamped
    t = f.tensor()
    assert t.shape == (N_FEATURES,) and float(t.min()) >= 0 and float(t.max()) <= 1


def test_choose_respects_the_rules_mask(director):
    rng = random.Random(0)
    allowed = {Beat.CALM, Beat.CLUE}
    for _ in range(30):
        beat, probs = director.choose(DirectorFeatures(), rng, allowed)
        assert beat in allowed
        assert abs(sum(probs.values()) - 1.0) < 1e-3
        assert all(probs[b.value] == 0 for b in BEATS if b not in allowed)


def _mean_probs(director, features, n=40):
    return torch.stack([director.probabilities(features) for _ in range(n)]).mean(0)


def _best_of(p, allowed):
    return max(allowed, key=lambda b: float(p[BEATS.index(b)]))


def test_beats_split_into_draws_and_narration():
    assert DRAWS | NARRATIVE == set(BEATS) and not DRAWS & NARRATIVE


def test_director_offers_a_struggling_bearer_a_choice(director):
    struggling = DirectorFeatures(grit=0.1, stress=0.9, wounds=0.66, tension=0.8, progress=0.3)
    assert _best_of(_mean_probs(director, struggling), DRAWS) is Beat.SHAPES


def test_director_meets_a_comfortable_late_trial_with_the_dark(director):
    quiet = DirectorFeatures(grit=1.0, stress=0.0, tension=0.05, progress=0.9, boldness=0.8, hunger=0.6)
    assert _best_of(_mean_probs(director, quiet), DRAWS) is Beat.DARK


def test_director_lets_a_hostile_person_speak(director):
    tense = DirectorFeatures(npc_present=1.0, npc_hostility=1.0, in_chapter=1.0, tension=0.5, progress=0.3)
    assert _best_of(_mean_probs(director, tense), NARRATIVE) is Beat.NPC


def test_trained_director_tracks_teacher(director_model):
    gen = torch.Generator().manual_seed(123)
    x = sample_states(1000, gen)
    with torch.no_grad():
        p = torch.softmax(director_model(x, gen), dim=1)
    kl = torch.sum(teacher_distribution(x) * (teacher_distribution(x).log() - p.log()), dim=1).mean()
    assert float(kl) < 0.2


def test_surrogate_gradient_training_reduces_loss():
    _, losses = train_director(steps=100, batch=128, seed=1)
    assert sum(losses[-10:]) / 10 < 0.9 * sum(losses[:10]) / 10
