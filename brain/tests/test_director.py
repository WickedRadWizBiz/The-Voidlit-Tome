import random

import torch

from voidlit.cortex import BEATS, N_FEATURES, Beat, DirectorFeatures, teacher_distribution, train_director
from voidlit.cortex.teacher import sample_states


def test_features_are_unit_interval():
    f = DirectorFeatures(lumen=2.0, stress=-1.0)
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


def test_director_gives_a_struggling_player_air(director):
    struggling = DirectorFeatures(hull=0.1, stress=0.9, tension=0.8, progress=0.5, since_respite=1.0)
    p = _mean_probs(director, struggling)
    assert BEATS[int(p.argmax())] is Beat.RESPITE


def test_director_escalates_a_quiet_late_game(director):
    quiet = DirectorFeatures(hull=1.0, stress=0.0, tension=0.05, progress=0.9, since_threat=1.0, boldness=0.8,
                             lumen=0.5)
    p = _mean_probs(director, quiet)
    top2 = {BEATS[i] for i in torch.topk(p, 2).indices.tolist()}
    assert top2 & {Beat.ESCALATE, Beat.HAZARD}


def test_trained_director_tracks_teacher(director_model):
    gen = torch.Generator().manual_seed(123)
    x = sample_states(1000, gen)
    with torch.no_grad():
        p = torch.softmax(director_model(x, gen), dim=1)
    kl = torch.sum(teacher_distribution(x) * (teacher_distribution(x).log() - p.log()), dim=1).mean()
    assert float(kl) < 0.2


def test_surrogate_gradient_training_reduces_loss():
    _, losses = train_director(steps=40, batch=128, seed=1)
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5
