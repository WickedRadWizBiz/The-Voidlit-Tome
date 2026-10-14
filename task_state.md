# Task State — The Voidlit Tome

> The handoff file for cross-session work. Read this first after `/clear`; update it after every sub-task.

## Current objective (2026-10-14, Steven in the "TOME - Game Mechanics" project)
Reconcile the app with the project's design. **The project wins every conflict**; the app keeps only what the
project did not already have. Sources of truth (read-only from here):
- TOME: Core Rules (Draft 0.2), https://claude.ai/code/artifact/09d100ff-71c2-4696-8d3b-da93391a2dbe
- Game content: TOME Forge JSON (`/mnt/project-files/tome/content/`, Forge format "tome-content")
- Lore: TOME: Lore Workshop + `/mnt/project-files/tome/lore/content/`
- Balance reference: `/mnt/project-files/tome/playtest/tome_sim.py` + `final_cfg.json`
Branch: `claude/reconcile-core-rules` (from `claude/determined-maxwell-5j79pa`), PR into `main`.

## Plan
1. [x] rules/: replace Grit/Guile/Insight, Hull, pips, clocks, act deck with Core Rules 0.2
2. [x] content: ship a Forge snapshot (brain/voidlit/content/tome/*.json) + tools/sync_content.py; places (Hollow Lantern) load from the Forge
3. [x] engine: draw prompts, Leaves, Push, Bargain, Breather/Echo/Intervention, Crucible, Haven, Pages, oracle
4. [x] keep + rewire: senses (4 stats), NPC brains on Witness Leaves, Director re-scoped to draw mode + beats
5. [x] voice: Stress registers; palette = voxel color language
6. [x] tests (84 pass), balance report, docs, Unreal protocol v2 fields (Grit, class, place)
7. [ ] export Hollow Lantern + app-only rules to /mnt/project-files/tome/from-app/ for the coordinator
8. [ ] PR into main, drive CI green

## Rules settled with the rules thread (2026-10-14, pending Steven's card there)
- Panic 7 in the Crucible is a lost action; the Crucible is Depth 8, Damage 2; Vessel panics there (or with the
  Lumen out) cost its next action. Ally die, gear helps, Talent coverage and Director scope are now Core Rules.
- Places come from the Forge (places, people, Leaves with a place); the app's TOML copy is gone.

## Not implemented yet
Group play, Vigil mode, recovering a Debt, Insight for acting on a Bond or Debt.

## Next immediate action
Step 7, then step 8. After merge: re-run tools/sync_content.py whenever the Forge changes; compile the UE plugin
on Steven's machine (needs UE 5.8); PyInstaller packaging; playtest with a real local model.
