# Architecture: the Tome's split brain

The Voidlit Tome is a Dungeon Master built from three kinds of machinery, each doing only what it is good at:

| Layer | Library | Learns | Decides | Never does |
|---|---|---|---|---|
| **Rules** (`voidlit/rules`) | plain Python | — | every outcome: dice, harm, stress, light, movement | improvise |
| **Limbic system** (`voidlit/limbic`) | **BindsNET** | online, every turn (R-STDP, structural plasticity) | each NPC's motive and tactic | write words |
| **Cortex** (`voidlit/cortex`) | **SpikingJelly** | offline (surrogate-gradient BPTT) | the DM's pacing beat | touch the dice |
| **Transducer / Voice** (`voidlit/voice`) | **llama.cpp**, in-process (llama-cpp-python, local `.gguf`) | — | how things are *said* | decide what happens |

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
                                          │ 2 rules    Core Rules 0.2: cast, Leaf, Bite  │
                                          │ 3 limbic   BindsNET NPC brains (R-STDP)      │
                                          │ 4 pressure Stress, Panic, the Lumen dims     │
                                          │ 5 cortex   SpikingJelly Director → draw/beat │
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
* The brain can run on a GPU, or headless for automated playtesting (`tests/test_engine.py`
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
* **Mechanical teeth:** NPCs live in place packs and appear on Witness Leaves. One whose dominant plot is
  *alliance* lends +1 die to the player's casts (an app rule, offered to the Core Rules). Attacking them drives them
  off. The brains matter to survival, not just to flavour.
* **Prompt compilation** (`voice/prompts.py`): dominant plot, subtext plot, tactic, lineage ("turning on you
  because you broke their trust") and provenance ("Marcus told you things about them") become a system prompt.
  Offline, the tactic's authored lines are used instead.

## The cortex: the Director (`cortex/director.py`)

A SpikingJelly multi-step network: Poisson-encoded 17-feature game state → `Linear→LIF→Linear→LIF→Linear`, averaged
over 16 timesteps. It is trained with surrogate gradients (ATan) to imitate a hand-written pacing teacher
(`cortex/teacher.py`), with KL ≈ 0.06. The Core Rules already escalate on their own (Depth, Abyssal Leaves, the
Lumen, the Crucible), so the Director never changes a roll, a Leaf or the deck. It decides only presentation:

* **how the next draw is offered**: `dot` (one pulsing point), `shapes` (two or three; the player picks one and the
  rest are lost), or `dark` (nothing appears and the TOME waits);
* **the narrative beat** of a turn: `calm`, `foreshadow`, `clue` (a truth of this place), `npc` (someone present
  acts on their own agenda).

The teacher's instincts: tension should climb through a trial, a struggling Bearer is offered the shapes, a
comfortable one is met with the dark, and success invites a revealed truth. Playtest logs can fine-tune the network
later. `python -m voidlit train-director` retrains it in about 30 seconds on a CPU.

## The voice: llama.cpp as a transducer, never a decider

* **Pass 1, senses** (`voice/llama_local.py: LocalTransducer`): the player's text goes in with a GBNF grammar
  rebuilt every turn (`senses/grammar.py`), so `target` can only be an entity actually in the scene. Keys and
  punctuation are literals; the model samples only enums and two-decimal floats at temperature 0.1.
* **Pass 2, voice** (`LocalVoice`): narration gets a list of *already-resolved facts*, the outcome, the beat, and
  strict instructions never to invent results. NPC dialogue gets the compiled limbic prompt.

### Local only

The language model never leaves this machine, and nothing reaches the network to use one:

* llama.cpp runs **inside the brain's process** through llama-cpp-python (`LocalLlama`). There is no
  `llama-server`, no HTTP and no socket between the game and the model.
* `--model` / `--senses-model` must be **existing local `.gguf` files**. URLs, `hf:` and Hugging Face references
  are refused (`resolve_model_path`). `from_pretrained`, the bindings' download helper, is never called.
* The bindings load only llama.cpp's core libraries (`libllama`, `libggml*`), which contain no network code.
  llama.cpp's own model downloader lives in a separate library, `libllama-common` (cpp-httplib + OpenSSL), which
  llama-cpp-python builds and installs but never loads. A test asserts it is not mapped into the process after a
  model turn. (The `LLAMA_CURL` build flag is not a safeguard: llama.cpp now ignores it.)
* The one socket in the system, the WebSocket to Unreal, **only binds loopback** (127.0.0.1, ::1 or localhost);
  `ensure_loopback` refuses anything else.
* `tests/test_llama_local.py` guards this. A static scan fails if any brain module imports a common outbound
  client (urllib, http, requests, httpx, aiohttp, socket, ssl and others), a model hub, or llama-cpp-python's
  bundled HTTP server, or calls `from_pretrained`. It is a regression tripwire, not a sandbox. The same file runs
  real llama.cpp against tiny random-weight GGUFs generated on the spot, so the grammar path and strict chat
  templates are tested without downloading anything.
* Every prompt is laid out as one optional system turn plus one user turn (context goes in the user text, never as
  a fake assistant turn). If a model's chat template rejects a system role (Gemma-2, some Mistral releases), that
  is detected at load and the system text is merged into the user turn. Otherwise such models would silently fall
  back to the offline voice.
* Each loaded model is shared across sessions and guarded by a lock (a llama.cpp context is not thread-safe).
  Every grammar is parsed by llama.cpp before sampling. A grammar that fails to parse would otherwise crash the
  process inside the bindings; this check found exactly that bug in the original spec's multi-line grammar.
* **Offline fallback for everything** (`senses/heuristic.py`, `voice/narrator.py`): lexicon senses and authored
  phrase banks. With no model running the game still plays, and every test runs without a GPU.

## Mapping from the original spec

| Spec item | Where it lives now |
|---|---|
| §3 Stage 1 sensory extraction (5 floats) | `senses/state.py`, extended with intent, stat, target and boldness for a DM |
| §4 GBNF grammar | `senses/grammar.py`, dynamic per scene; `grammars/sensory.gbnf` is the static reference. The spec's multi-line rule does not parse in llama.cpp (a rule ends at a newline outside parentheses), so the body is wrapped in `( … )`. |
| §3 Stage 2 retroactive R-STDP | `NPCBrain.consolidate`, using BindsNET `MSTDPET` traces |
| §3 Stage 3 Poisson / NMDA plateau / WTA / tactics | `NPCBrain.simulate`, `DendriticBranchNodes`, LIF plots with lateral inhibition |
| §3 Stage 4 calcium, prune, split, trauma | `NPCBrain.restructure` |
| §3 Stage 5 lineage and provenance prompts | `voice/prompts.py: compile_npc_prompt` |
| §3 Stage 6 vocal generation | `LocalVoice.speak` (in-process llama.cpp) |
| §5.2 off-screen contagion and grafting | `limbic/contagion.py: gossip` |
| `.snnb` binary save | `NPCBrain.state_dict` via `torch.save` (versioned, includes RNG state); `TomeEngine.save` |
| `llama-server` over localhost HTTP | replaced by in-process llama.cpp on local files (see *Local only*) |
| §7 tests (1)–(4) | `tests/test_senses.py`, `tests/test_limbic.py` (R-STDP A/B, round-trip, grafting), `tests/test_llama_local.py` (real llama.cpp grammar pass) |

Tuning constants differ from the spec where calibration demanded it (plateau threshold, plateau length, tactic
threshold, eligibility gain). The rationale is in the header of `limbic/npc_brain.py`.

## Balance snapshot

`tools/balance.py` plays seeded short trials per class through the whole app. With a sensible bot, short-trial
survival is about 87–100% by class (30 seeds each); a bot that never pushes and ignores its best stat survives
about 20%. The project's playtest simulator reports 93–97% for the same rules; the main known difference is how
Panic 7 in the Crucible is read (see `docs/RULES.md`). Re-measure after content changes.

## Roadmap

1. Compile and play-test the Unreal plugin in a 5.8 project, then author the material with
   `Content/Python/create_tome_material.py`.
2. Spawn and supervise the sidecar from `ATomeMonolith`, and package it with PyInstaller (bundling llama.cpp).
3. Fine-tune the Director on playtest logs (designer-labelled good and bad beats).
4. More place packs (TOML only), NPC-to-NPC scenes on screen, and group play from the Core Rules.
