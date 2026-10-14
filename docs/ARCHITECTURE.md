# Architecture: the Tome's split brain

The Voidlit Tome is a Dungeon Master built from three kinds of machinery, each doing only what it is good at:

| Layer | Library | Learns | Decides | Never does |
|---|---|---|---|---|
| **Rules** (`voidlit/rules`) | plain Python | — | every outcome: dice, harm, stress, light, movement | improvise |
| **Limbic system** (`voidlit/limbic`) | **BindsNET** | online, every turn (R-STDP, structural plasticity) | each NPC's motive and tactic | write words |
| **Cortex** (`voidlit/cortex`) | **SpikingJelly** | offline (surrogate-gradient BPTT) | the DM's pacing beat | touch the dice |
| **Transducer / Voice** (`voidlit/voice`) | **llama.cpp** (`llama-server`) | — | how things are *said* | decide what happens |

The body is a separate process: an **Unreal Engine 5.8** plugin (`unreal/VoidlitTome`) that renders the Tome's voxel
surface and talks to the brain over a localhost WebSocket ([PROTOCOL.md](PROTOCOL.md)).

```
 player keystrokes                                                        voxel surface
        │                                                                       ▲
        ▼                                                                       │
 ┌──────────────┐   ws://127.0.0.1:8765   ┌──────────────────────────────────────┴───────┐
 │ Unreal 5.8   │ ──── input ───────────▶ │ voidlit serve  (Python sidecar)              │
 │ ATomeMonolith│ ◀─── frame ──────────── │                                              │
 └──────────────┘                         │ 1 senses   llama.cpp GBNF pass → SensoryState│
                                          │ 2 rules    dice pools, obstacles, items      │
                                          │ 3 limbic   BindsNET NPC brains (R-STDP)      │
                                          │ 4 pressure panic, the Lumen gutters          │
                                          │ 5 cortex   SpikingJelly Director → beat      │
                                          │ 6 voice    llama.cpp narration + dialogue    │
                                          │ 7 frame    semantic blocks → Unreal          │
                                          └──────────────────────────────────────────────┘
```

## Why a Python sidecar instead of the C++ SNN in the original spec

The reference spec (summarised below) proposed a hand-written C++20 spiking network. You asked for **SpikingJelly
and BindsNET**, which are PyTorch libraries, so the mind runs as a Python process next to the game:

* Both libraries are used for what each does best: BindsNET for biologically plausible *online* plasticity,
  SpikingJelly for *trainable* deep SNNs with surrogate gradients and CUDA/CuPy backends.
* The game can iterate on the brain (retune, retrain, hot-swap content) without recompiling Unreal.
* The brain can run on a GPU, a second machine, or headless for automated playtesting (`tests/test_engine.py`
  plays whole runs).
* Shipping: bundle the sidecar with PyInstaller (or an embedded CPython) and have `ATomeMonolith` spawn it. The
  protocol is plain JSON, so a future C++ port of the hot path stays possible.

## The limbic system: one BindsNET network per NPC (`limbic/npc_brain.py`)

```
senses (Input, 6 ch) ──MSTDPET, masked spines──▶ branches (DendriticBranchNodes, 32 slots)
                                                        │ fixed routing (one plot per branch)
                                                        ▼
                                         plots (LIF, winner-take-all via −0.45 lateral inhibition)
                                                        │ MSTDPET
                                                        ▼
                                                 tactics (LIF) ──▶ the NPC's move this turn
```

* **Dendritic branches** are a custom BindsNET `Nodes` subclass. Spine synapses sum into a compartment voltage, and
  crossing θ fires a 5 ms *NMDA plateau*, a super-linear burst into the branch's plot. Each plateau deposits
  slow **calcium**, which survives between turns.
* **Causal reward timing:** BindsNET's `MSTDPET` rule accumulates eligibility traces during the 100 ms turn with
  `reward=0`. On the *next* turn, before simulating, `consolidate()` computes dopamine
  `R = clamp(0.6·rapport − 0.5·deception − 0.4·exposure)` (per-NPC weights), the prediction error `R − R̄`, and applies
  `Δw = η·(R−R̄)·tanh(k·e)·(0.2 + w(1−w))`. The NPC is paid for last turn's tactic by this turn's reaction.
* **Structural plasticity:** branch volume `V = 0.92V + 0.20Ca + 0.08w̄`. A branch is pruned when `V < 0.15` and
  split when `Ca > 0.85`; a split under hostility turns the child toward betrayal but keeps its `origin_role`.
  A shock above 0.85 sprouts a trauma branch (3-turn cooldown).
* **Gossip** (`limbic/contagion.py`): between chapters, NPCs who have met the player talk. Plot momentum resonates
  by `0.25·C·trust`, and the speaker's most salient branch is *grafted* into the listener, attenuated by
  `|trust|` and tagged with `source_npc`. Coming from a hated rival, a betrayal branch inverts into a warning.
* **Mechanical teeth:** an NPC whose dominant plot is *alliance* lends +1 die to the player's rolls. Attacking them
  drives them off. The brains matter to survival, not just to flavour.
* **Prompt compilation** (`voice/prompts.py`): dominant plot, subtext plot, tactic, lineage ("turning on you
  because you broke their trust") and provenance ("Marcus told you things about them") become a system prompt.
  Offline, the tactic's authored lines are used instead.

## The cortex: the Director (`cortex/director.py`)

A SpikingJelly multi-step network: Poisson-encoded 18-feature game state → `Linear→LIF→Linear→LIF→Linear`, averaged
over 16 timesteps. It is trained with surrogate gradients (ATan) to imitate a hand-written pacing teacher
(`cortex/teacher.py`), with KL ≈ 0.09. The engine masks the beats the rules forbid right now, then samples one:

`calm · escalate · clue · npc · hazard · respite · twist · advance`

Each beat has a concrete rules effect: clocks tick, hazards spawn, an obstacle adapts, an NPC acts unprompted,
and so on. The teacher encodes the designer's instincts: target tension rises with the act, a struggling player
gets air, a stalled scene gets a push. Playtest logs can fine-tune the network later without touching rules code.
`python -m voidlit train-director` retrains it in about 30 seconds on a CPU.

## The voice: llama.cpp as a transducer, never a decider

* **Pass 1, senses** (`voice/llama_bridge.py: LlamaTransducer`): the player's text goes in with a GBNF grammar
  rebuilt every turn (`senses/grammar.py`), so `target` can only be an entity actually in the scene. Keys and
  punctuation are literals; the model samples only enums and two-decimal floats at temperature 0.1.
* **Pass 2, voice** (`LlamaVoice`): narration gets a list of *already-resolved facts*, the outcome, the beat, and
  strict instructions never to invent results. NPC dialogue gets the compiled limbic prompt.
* **Offline fallback for everything** (`senses/heuristic.py`, `voice/narrator.py`): lexicon senses and authored
  phrase banks. With no model running the game still plays, and every test runs without a GPU.

## Mapping from the original spec

| Spec item | Where it lives now |
|---|---|
| §3 Stage 1 sensory extraction (5 floats) | `senses/state.py`, extended with intent, approach, target and boldness for a DM |
| §4 GBNF grammar | `senses/grammar.py`, dynamic per scene; `grammars/sensory.gbnf` is the static reference |
| §3 Stage 2 retroactive R-STDP | `NPCBrain.consolidate`, using BindsNET `MSTDPET` traces |
| §3 Stage 3 Poisson / NMDA plateau / WTA / tactics | `NPCBrain.simulate`, `DendriticBranchNodes`, LIF plots with lateral inhibition |
| §3 Stage 4 calcium, prune, split, trauma | `NPCBrain.restructure` |
| §3 Stage 5 lineage and provenance prompts | `voice/prompts.py: compile_npc_prompt` |
| §3 Stage 6 vocal generation | `LlamaVoice.speak` |
| §5.2 off-screen contagion and grafting | `limbic/contagion.py: gossip` |
| `.snnb` binary save | `NPCBrain.state_dict` via `torch.save` (versioned, includes RNG state); `TomeEngine.save` |
| §7 tests (1)–(4) | `tests/test_senses.py`, `tests/test_limbic.py` (R-STDP A/B, round-trip, grafting) |

Tuning constants differ from the spec where calibration demanded it (plateau threshold, plateau length, tactic
threshold, eligibility gain). The rationale is in the header of `limbic/npc_brain.py`.

## Balance snapshot

`tests/test_engine.py` contains two scripted players. Over 40 seeded runs of *The Hollow Lantern*, a sensible bot
escapes about 60% of the time and a naive bot about 7%. Most deaths come at the final boss. Re-measure after
content changes (see `task_state.md`).

## Roadmap

1. Compile and play-test the Unreal plugin in a 5.8 project, then author the material with
   `Content/Python/create_tome_material.py`.
2. Spawn and supervise the sidecar from `ATomeMonolith`, and package it with PyInstaller.
3. Fine-tune the Director on playtest logs (designer-labelled good and bad beats).
4. More sectors (TOML only), NPC-to-NPC scenes on screen, and multiple crew members (Escape the Dark Sector style).
