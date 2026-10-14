# The Voidlit Tome

A text-first sci-fi horror RPG whose game master is **the TOME**: an ancient slab of matte black stone, shaped
like a book, that opens with no seams and no pages. Its surface is a field of voxels. Flush, they vanish into the
black; raised or sunken a millimetre, they glow in vivid colour and become words, dice and the things hunting you.
You are its Bearer, and it runs you through trials.

The Tome's mind is a **split brain**:

* **BindsNET** gives every NPC a limbic system: dendritic plateaus, winner-take-all motives, and reward-modulated
  STDP that learns which manipulation works on *you*. NPCs grow and prune their own schemes and gossip about you
  behind your back.
* **SpikingJelly** runs the cortex: a Director SNN, trained with surrogate gradients, that paces the game the way a
  good DM does.
* **llama.cpp** is only a transducer and a voice, running **in-process from a local `.gguf` file**: no server,
  no URLs, no downloads. A GBNF-locked pass reads your intent and tone; a second pass gives the Tome and its NPCs
  words. It never decides what happens: the dice do.

The rules are **TOME: Core Rules (Draft 0.2)**, the project's rules document, in the spirit of **Voidlight**,
**Mothership** and **Escape the Dark Sector**: d6 pools where 5–6 hit, Stress and Panic, a hidden deck of Leaves
that ends in the Crucible, and the **Lumen**, a dying light that is both the trial's clock and the brightness of
the voxels. See [docs/RULES.md](docs/RULES.md) for what the app implements and adds.

## Quick start (terminal, no GPU or LLM needed)

```bash
cd brain
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"
python -m voidlit play --class breacher --name Ash
python -m voidlit play --place hollow_lantern --debug   # watch the NPC brains and the Director think
```

With a local language model (llama.cpp compiled into the brain's own process by llama-cpp-python):

```bash
pip install -e ".[llm]"                                     # GPU offload: set CMAKE_ARGS for your backend first
python -m voidlit play --model /path/to/model.gguf          # --senses-model small.gguf  --gpu-layers 99
```

Model arguments must be existing local `.gguf` files: URLs and model-hub names are refused, and nothing is ever
fetched. The game also runs with no model at all (heuristic senses and authored prose).

For Unreal Engine 5.8:

```bash
python -m voidlit serve --model /path/to/model.gguf         # ws://127.0.0.1:8765, loopback only
```

Then copy `unreal/VoidlitTome` into your project's `Plugins/` folder (see [unreal/README.md](unreal/README.md)).

## Layout

```
brain/                      Python sidecar: the Tome's mind
  voidlit/rules/            Core Rules 0.2: casting, the Bearer, the deck, Panic, Lumen, Pages, campaign
  voidlit/senses/           SensoryState, per-scene GBNF grammar, offline heuristic transducer
  voidlit/limbic/           BindsNET NPC brains, dendritic nodes, R-STDP, structural plasticity, gossip
  voidlit/cortex/           SpikingJelly Director (+ trained weights) and its pacing teacher
  voidlit/voice/            local in-process llama.cpp (llama-cpp-python), prompt compilers, offline narrator
  voidlit/tome/             the turn engine, render frames, content loader
  voidlit/content/tome/     the TOME Forge's content (JSON snapshot; refresh with tools/sync_content.py)
  voidlit/content/places/   place packs as TOML ("The Hollow Lantern")
  tests/                    rules, senses, BindsNET plasticity, Director, full trials and campaigns, WebSocket,
                            local-only guards, real llama.cpp on a generated tiny GGUF
  tools/balance.py          seeded bot trials → survival by class
  tools/sync_content.py     copy the Forge's content into the app
unreal/VoidlitTome/         UE 5.8 plugin: voxel surface, glyph typesetter, brain client, monolith actor
docs/                       ARCHITECTURE.md · RULES.md · PROTOCOL.md
grammars/sensory.gbnf       reference sensory grammar
task_state.md               cross-session handoff (current objective, decisions, next step)
```

## Tests

```bash
cd brain && python -m pytest -q          # ~10 s on a CPU
python tools/balance.py --runs 40         # balance report (also runs in CI)
```
