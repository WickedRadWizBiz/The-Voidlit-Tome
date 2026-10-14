# Task State — The Voidlit Tome

> The handoff file for cross-session work. Read this first after `/clear`; update it after every sub-task.

## Current objective
First runnable slice of **The Voidlit Tome**, a text-first sci-fi horror RPG whose Dungeon Master is a voxel
monolith. **Status: v0.1 done and pushed** on branch `claude/determined-maxwell-5j79pa`.

## Decisions (do not re-litigate)
- The brain is a **Python sidecar** (`brain/`) because SpikingJelly and BindsNET are PyTorch libraries; Unreal is
  the body over a localhost WebSocket (`docs/PROTOCOL.md`). This departs from the Gemini spec's C++ SNN. The
  mapping is in `docs/ARCHITECTURE.md`.
- **BindsNET = limbic** (per-NPC online R-STDP, dendritic plateaus, structural plasticity, gossip).
  **SpikingJelly = cortex** (the Director, trained offline with surrogate gradients).
- The LLM never decides mechanics; there is a full offline fallback (heuristic senses, phrase banks).
- Mechanics and text are original; inspirations are credited in `docs/RULES.md`.
- Voxel glyphs come from DejaVu Sans Mono Bold (permissive license). Unifont was rejected because the installed
  package is GPL-2+.

## Done
- [x] rules/: d6 pools, Grit/Guile/Insight, obstacles as approach pips (off-approach: 2 hits = 1 pip), Stress/Panic
      (2d6 + Insight vs Stress; panicking discharges 2), Lumen (interval 5), clocks, sector deck, Last Light
- [x] senses/: SensoryState, per-scene GBNF, strict parser, heuristic transducer
- [x] limbic/: BindsNET NPC brain (custom DendriticBranchNodes, LIF WTA plots, MSTDPET + retroactive consolidation),
      prune/split/trauma, gossip grafting, save/load. An ally NPC gives +1 die (the brains matter mechanically).
- [x] cortex/: Director SNN plus weights (`voidlit/cortex/director.pt`, KL ~0.09). Retrain after teacher changes:
      `python -m voidlit train-director`.
- [x] voice/: llama-server bridge (grammar pass, narration pass), prompt compilers, offline narrator with variants
- [x] tome/: engine turn loop, frames, `content/hollow_lantern.toml`; DM fair-play masks (hazards only on a clear
      scene with a cooldown, one twist per obstacle)
- [x] server + CLI (`python -m voidlit play|serve|train-director|gbnf`), `tools/balance.py`
- [x] 50 pytest tests passing (~8 s on CPU); CI workflow `.github/workflows/brain-tests.yml` with a balance report
- [x] Unreal 5.8 plugin `unreal/VoidlitTome` (voxel ISM surface, GPU-animated via TomeVoxel.ush, typesetter,
      glyphs, WebSocket client, monolith actor, Slate keyboard capture, material-building editor script)

## Balance snapshot (40 seeded runs)
- Smart bot escapes ~60%; naive bot ~7%. Most deaths are at the final boss (the Conductor).

## Known gaps / next steps (priority order)
1. **Compile the Unreal plugin** in a real UE 5.8 project. It has never been built (no UE in CI); expect small
   API fixes. Then run `Content/Python/create_tome_material.py`.
2. Package the sidecar (PyInstaller) and have `ATomeMonolithActor` spawn it by default.
3. Fine-tune the Director on playtest logs; add more sectors (TOML only).
4. Try the LLM path against a real `llama-server` (only mocked in tests so far).

## Environment notes
- download.pytorch.org is blocked here; torch came from PyPI (CUDA build, ~6 GB venv in the scratchpad, not
  committed). CI uses CPU torch from download.pytorch.org.
- farirpgs.com and figcat.com are blocked; Voidlight and FigCat details came from web search summaries.

## Next immediate action
None pending in this session. Next session: step 1 above (needs a machine with UE 5.8).
