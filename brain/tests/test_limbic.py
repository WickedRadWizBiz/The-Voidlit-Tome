import torch

from voidlit.limbic import NPCBrain, gossip
from voidlit.limbic.npc_brain import PRUNE_VOLUME, W_MAX, W_MIN
from voidlit.senses import SensoryState

from .conftest import make_spec

WARM = SensoryState(rapport=0.8, leverage=0.3)
HOSTILE = SensoryState(aggression=0.9, exposure=0.8, rapport=-0.7)


def test_winner_take_all_follows_the_player(spec):
    brain = NPCBrain(spec, seed=1)
    for _ in range(4):
        d = brain.perceive(WARM)
    assert d.dominant_plot == 0 and d.plot_spikes[0] > 0 and d.plot_spikes[2] == 0
    for _ in range(5):
        d = brain.perceive(HOSTILE)
    assert d.dominant_plot == 2, "sustained hostility should hand the NPC to its betrayal plot"
    assert d.plot_spikes[2] > d.plot_spikes[0]


def test_reward_arrives_one_turn_late(spec):
    brain = NPCBrain(spec, seed=2)
    first = brain.perceive(WARM)
    assert first.reward is None and first.rpe is None
    second = brain.perceive(WARM)
    assert second.reward is not None and second.reward > 0


def test_rstdp_weights_converge_with_reward_sign():
    """Spec checklist (2): an A/B of the three-factor rule. Both brains see identical spikes (same seed, same
    input); only the sign of the dopamine differs. Reward potentiates the plot->tactic synapse that acted,
    punishment depresses it, and both stay inside the soft bounds."""
    pos_spec, neg_spec = make_spec(), make_spec()
    pos_spec.reward = {"rapport": 0.6}
    neg_spec.reward = {"rapport": -0.6}
    pos, neg = NPCBrain(pos_spec, seed=3), NPCBrain(neg_spec, seed=3)
    w0 = float(pos.c_tactics.w[0, 0])
    for _ in range(12):
        pos.perceive(WARM)
        neg.perceive(WARM)
    assert float(pos.c_tactics.w[0, 0]) > w0 + 0.03
    assert float(neg.c_tactics.w[0, 0]) < w0 - 0.03
    for w in (pos.c_tactics.w, neg.c_tactics.w):
        assert float(w.min()) >= W_MIN - 1e-6 and float(w.max()) <= W_MAX + 1e-6


def test_spine_mask_is_respected(spec):
    brain = NPCBrain(spec, seed=4)
    for s in (WARM, HOSTILE, WARM):
        brain.perceive(s)
    w = brain.c_spines.w
    assert torch.all(w[~brain.spine_mask] == 0)
    assert torch.all(w[brain.spine_mask] >= W_MIN - 1e-6)


def test_prune_split_and_trauma(spec):
    brain = NPCBrain(spec, seed=5)
    codes = brain.slot_of(2)
    with torch.no_grad():
        brain.c_spines.w[:, codes] *= 0.0
        brain.c_spines.w[brain.spine_mask[:, codes], codes] = W_MIN
    brain.meta[codes].volume = PRUNE_VOLUME
    shelter = brain.slot_of(1)
    brain.branches.calcium[shelter] = 0.95
    d = brain.perceive(HOSTILE)
    assert "pruned branch 2" in d.events
    split = [e for e in d.events if e.startswith("branch 1 split")]
    assert split and "toward betrayal" in split[0]
    child = brain.meta[brain.slot_of(int(split[0].split()[4]))]
    assert child.parent_lineage_id == 1 and child.origin_role == "ally" and child.target_plot == 2
    assert any(e.startswith("trauma sprouted") for e in d.events)
    d2 = brain.perceive(HOSTILE)
    assert not any(e.startswith("trauma sprouted") for e in d2.events), "trauma has a cooldown"


def test_save_load_round_trip_is_exact(spec, tmp_path):
    """Spec checklist (3): serialization round-trips, including RNG state, so the futures match."""
    a = NPCBrain(spec, seed=6)
    for s in (WARM, HOSTILE, WARM):
        a.perceive(s)
    a.save(tmp_path / "elara.pt")
    b = NPCBrain(spec, seed=999)
    b.load(tmp_path / "elara.pt")
    assert torch.equal(a.c_spines.w, b.c_spines.w) and torch.equal(a.c_tactics.w, b.c_tactics.w)
    assert [m and m.to_dict() for m in a.meta] == [m and m.to_dict() for m in b.meta]
    da, db = a.perceive(HOSTILE), b.perceive(HOSTILE)
    assert da.to_dict() == db.to_dict()


def test_gossip_resonates_and_grafts():
    """Spec checklist (4): off-screen dendritic grafting carries provenance."""
    speaker = NPCBrain(make_spec("elara", 1), seed=7)
    listener = NPCBrain(make_spec("marcus", 2), seed=8)
    for _ in range(5):
        speaker.perceive(HOSTILE)
    before = float(listener.momentum[2])
    res = gossip(speaker, listener, trust=0.8)
    assert float(listener.momentum[2]) > before
    assert res.grafted_branch is not None
    meta = listener.meta[listener.slot_of(res.grafted_branch)]
    assert meta.source_npc == 1 and listener.spec.plots[meta.target_plot].role == "betray"


def test_hated_rival_inverts_betrayal_into_warning():
    speaker = NPCBrain(make_spec("elara", 1), seed=9)
    listener = NPCBrain(make_spec("marcus", 2), seed=10)
    for _ in range(5):
        speaker.perceive(HOSTILE)
    slot = max(speaker.active_slots(),
               key=lambda i: float(speaker.branches.calcium[i]) + 0.5 * speaker.meta[i].volume)
    assert speaker.spec.plots[speaker.meta[slot].target_plot].role == "betray"
    res = gossip(speaker, listener, trust=-0.8)
    assert res.inverted
    meta = listener.meta[listener.slot_of(res.grafted_branch)]
    assert listener.spec.plots[meta.target_plot].role == "ally"


def test_weak_trust_does_not_graft():
    speaker = NPCBrain(make_spec("elara", 1), seed=11)
    listener = NPCBrain(make_spec("marcus", 2), seed=12)
    speaker.perceive(HOSTILE)
    assert gossip(speaker, listener, trust=0.1).grafted_branch is None
