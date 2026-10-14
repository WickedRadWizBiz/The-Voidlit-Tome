# Task State — The Voidlit Tome

> The handoff file for cross-session work. Read this first after `/clear`; update it after every sub-task.

## Current objective
Build the first runnable slice of **The Voidlit Tome**, a text-first sci-fi horror RPG where a
monolithic voxel Tome acts as the Dungeon Master. The work covers:
1. The rules engine (an original d6-pool system inspired by Voidlight, Mothership and Escape the Dark Sector).
2. The "split brain" Tome mind:
   - **Limbic (BindsNET):** per-NPC online R-STDP brains with dendritic plateaus, winner-take-all plots, structural plasticity and gossip grafting.
   - **Cortex (SpikingJelly):** a Director SNN trained with surrogate gradients that picks the DM's pacing beats.
   - **Transducer/Voice (llama.cpp):** a GBNF-constrained sensory pass and a narration/dialogue pass, with an offline fallback.
3. A WebSocket protocol and an Unreal Engine 5.8 plugin that renders the Tome's voxel surface.

## Decisions (do not re-litigate)
- The brain is a **Python sidecar** (`brain/`) because SpikingJelly and BindsNET are PyTorch libraries. Unreal is only the body and talks to it over a localhost WebSocket. This departs from the Gemini spec, which called for a C++ SNN.
- The LLM never decides mechanics. Dice and the rules decide; the SNNs choose strategy and pacing; the LLM only writes the words.
- The full game must run offline (heuristic senses + template narrator) so tests run without a GPU or LLM.
- Mechanics and text are original. Inspirations are credited in `docs/RULES.md`.

## Environment notes
- download.pytorch.org is blocked by the network policy, so torch comes from PyPI (CUDA build, large).
- The scratch venv lives in the session scratchpad and is not committed. CI installs from `brain/pyproject.toml`.
- farirpgs.com and figcat.com are blocked; the Voidlight and FigCat details came from web search summaries.

## Steps
- [ ] Repo skeleton, docs, CI
- [x] rules/ (dice pools, character, lumen, clocks, panic, sector deck, obstacles)
- [x] senses/ (SensoryState, dynamic GBNF, parser, heuristic extractor)
- [x] limbic/ (BindsNET NPC brain, dendritic nodes, R-STDP consolidation, structural plasticity, contagion, persistence). Constants were calibrated by a scripted run; see the npc_brain.py header.
- [x] cortex/ (SpikingJelly Director, teacher, training). Weights are in voidlit/cortex/director.pt (KL ~0.09). Retrain after any teacher change.
- [x] voice/ (llama-server bridge, prompt compilers, offline narrator with variants)
- [x] tome/ (engine turn loop, frames protocol, hollow_lantern.toml). A scripted offline playthrough works.
- [ ] server + CLI
- [ ] tests passing
- [ ] Unreal plugin (voxel surface, glyph font, typesetter, brain client, monolith actor, material script)
- [ ] commit + push

## Next immediate action
Write server.py (websockets), cli.py (ANSI terminal play), and __main__.py, then the pytest suite.
