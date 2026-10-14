"""Limbic system: per-NPC BindsNET brains with online reward-modulated plasticity."""

from .contagion import GossipResult, gossip
from .dendrites import BranchMeta, DendriticBranchNodes
from .npc_brain import BrainDecision, BranchSeed, NPCBrain, NPCSpec, PlotSpec, TacticSpec

__all__ = [
    "GossipResult", "gossip", "BranchMeta", "DendriticBranchNodes", "BrainDecision", "BranchSeed",
    "NPCBrain", "NPCSpec", "PlotSpec", "TacticSpec",
]
