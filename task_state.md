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
- **llama.cpp is local only (user requirement):** it runs in-process through llama-cpp-python from local `.gguf`
  files (`--model`, `--senses-model`). There is no llama-server, no HTTP, no URLs or hub names (refused), and no
  downloads. The bindings never load llama.cpp's downloader library (libllama-common); a test checks this. The
  `LLAMA_CURL` flag is ignored by current llama.cpp, so it is not a safeguard. The Unreal WebSocket binds loopback
  only. `tests/test_llama_local.py` guards all of this (static import tripwire, /proc maps, loopback,
  strict chat templates).
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
- [x] voice/: local in-process llama.cpp (`llama_local.py`: LocalLlama, LocalTransducer, LocalVoice), prompt
      compilers, offline narrator with variants. GBNF root wrapped in parentheses (llama.cpp ends rules at
      newlines); grammars are pre-parsed by llama.cpp to avoid a NULL-sampler segfault.
- [x] tome/: engine turn loop, frames, `content/hollow_lantern.toml`; DM fair-play masks (hazards only on a clear
      scene with a cooldown, one twist per obstacle)
- [x] server + CLI (`python -m voidlit play|serve|train-director|gbnf`), `tools/balance.py`
- [x] 73 pytest tests passing (~9 s on CPU), including real llama.cpp against a tiny random GGUF generated on the
      fly (`tests/tiny_gguf.py`). CI has two jobs: `test` (with a balance report) and `local-llm`.
- [x] Unreal 5.8 plugin `unreal/VoidlitTome` (voxel ISM surface, GPU-animated via TomeVoxel.ush, typesetter,
      glyphs, WebSocket client, monolith actor, Slate keyboard capture, material-building editor script)

## Balance snapshot (40 seeded runs)
- Smart bot escapes ~60%; naive bot ~7%. Most deaths are at the final boss (the Conductor).

## Known gaps / next steps (priority order)
1. **Compile the Unreal plugin** in a real UE 5.8 project. It has never been built (no UE in CI); expect small
   API fixes. Then run `Content/Python/create_tome_material.py`.
2. Package the sidecar (PyInstaller) and have `ATomeMonolithActor` spawn it by default.
3. Fine-tune the Director on playtest logs; add more sectors (TOML only).
4. Play-test with a real local instruct model (.gguf) to tune the prompts; grammar output is already verified
   with real llama.cpp.

## Environment notes
- download.pytorch.org is blocked here; torch comes from PyPI (`pip install torch`, CUDA build), then
  `pip install -e "brain[dev]"`. CI uses CPU torch from download.pytorch.org. spikingjelly is pinned to 0.0.0.0.14.
- farirpgs.com and figcat.com are blocked; Voidlight and FigCat details came from web search summaries.

## Current sub-task (in progress): the adversarial narrator merge
The user sent two sources: the "AURA-X" affective engine (HTML/JS demo: PSI needs, PAD/OCC affect, Theory of Mind, BT/GOAP,
"Z3" gate) and a JAX + FastAPI "biomimetic human brain" spec with validation gates 0-4. The request: merge them into an even
more adversarial narrator, keep current behavior wherever it does not conflict, and **ask the user about every
contradiction**. Copies are in the session scratchpad (`upload/aura_engine.js`, `upload/biomimetic-human-brain/`); they are not
committed.
- [x] Read both sources; wrote the spec package to the scratchpad.
- [ ] Reconciliation workflow (run the spec gates, audit AURA-X, map against the build, skeptic pass).
- [x] User decisions (2026-10-21):
  - **Truth:** the Tome may *omit* important details until the player asks (lying by omission) and may make
    self-serving suggestions framed as helping the player, or as having no other choice. It still never states a
    falsehood about a resolved fact. (My reading; confirm if a case is unclear.)
  - **Truth reading confirmed** by the user.
  - **Voice:** AURA-X *is* the Tome; drop the name. **No readout at all** of the Tome's opinion or mood: they leak only
    through wording in narration and dialogue. Frames must not carry them (debug mode excepted).
  - **Brain:** the Tome is **one singular hybrid neural network**. The user's biomimetic brain spec is part of its core
    logic, made as real as possible: it must feel, have emotions and needs, and be cunning. It replaces the separate
    Director.
  - **NPCs (secondary story):** Elara and Marcus are *both* their own minds *and* corrupted by the Tome. Most of the
    time they act as themselves (their own BindsNET brains); at times they are mildly influenced by the Tome, more so
    as the Tome grows desperate. Design: a corruption projection from the Tome's network into their brains, gated by
    the Tome's desperation. It must stay subtle and never be displayed.
  - **Player corruption:** only after an extended time playing without reaching the end. The Tome bends the player's
    *choices* at the moment they are made, into a slight deviation that benefits it. Example: the Tome wants the player
    to take a weapon **Rune** to force a sealed door.
    - Tier 1: the player chooses another way, but the scene loops back, and they look at the sealed door again, unable
      to leave without reconsidering.
    - Tier 2 (higher contempt + desperation): after a second decline, the player must **tap the screen suddenly** (a
      timed resist prompt) to stop themselves accepting the Rune.
    - Narration stays truthful: it describes the loop or slip as it happens; dice still decide outcomes. Needs a protocol
      message for the timed resist prompt, plus handling in Unreal (tap or any key) and the CLI (Enter within the window).
    - Assumption: exposure accumulates across runs in a save profile and resets on escape; the thresholds are
      configurable.
  - **Stack:** "a little bit of each" (SpikingJelly + BindsNET + JAX), following my suggestion, all inside that one network.
- [x] User decisions (2026-10-21, round 3):
  - **Every action by any character** (player, NPCs, the Tome) gets an invisible attribute + die check relevant to it.
  - **Dice are fully invisible.** No dice are ever shown or disclosed, even when asked; the player should not know
    dice exist. Outcomes come only through narration and the status bar, so dice blocks, obstacle pips, clock
    counters and roll/event text are removed from frames.
  - **Hybrid dice:**
    - d6 for passive events that are not a direct action by a character (Lumen gutter, panic, environment, passive
      notice).
    - d20 + attribute modifier for a direct action by a character.
    - A bonus +d6 when the acting character has a significant upper hand on the Tome and is directly opposing it.
    - Corruption is a check (the Tome's d20 against the target's Resolve) layered on top of exposure tracking.
  - **Resolve** is a 4th attribute (will): it resists corruption and panic. Insight keeps perception and analysis,
    including passive notice of omissions.
  - **Puppets may lie:** a corrupted Elara or Marcus may state falsehoods in the Tome's interest. The Tome's own voice
    never does.
  - **Difficulty: brutal.** Target about 20% escapes for the sensible bot.
  - Settled without asking: the Tome may make self-serving claims about the future but never false claims about
    present or past facts; the player can address the Tome and it answers in the first person; NPC brains keep current
    behavior, and the biomimetic machinery is the Tome's alone (plus the corruption pathway).
- [x] Reconcile audit: the spec's gates fail 3 of 5 as written (Gate 1 NaN, Gate 2 frequency test, Gate 3 E/I ratio), and
      several checks are tautological. Build the real models (stable integrators, Graupner-Brunel theta_p = 1.3, ...)
      with honest tests. Evidence is in scratchpad `reconcile/`.
- [x] Design workflow `wf_b488a148-81d` done. Both judges chose the "biophysical" design; grafts from the other two
      applied. Final spec committed as **`docs/TOME_BRAIN_DESIGN.md`** (the source of truth for the Tome's network,
      checks, corruption, truth guard and gates G1-G24). Prototypes stay in the scratchpad (`design/`), not committed.
- [x] §19 open decisions use the design defaults (no further questions): cross-run memory **keep**; Tier 2
      `access_mult` 1.0 with the setting available; Elara serves via offering/misdirect, Marcus via escort/orders;
      Rune +4 vs seals, hidden cost 1 Hull -> 1 Lumen per use and +30 exposure, found in the Chapel of Static;
      attribute scores shown only on the `welcome` sheet; no NPC seize mechanic; NPC corruption cap (itself >= 60%).
- [x] **M2 rules (done, pushed):** `rules/attributes.py`, `rules/checks.py` (d20 degrees m>=10 crit, >=0 success,
      >=-5 mixed; nat 20/1 shift one step; adapter to pool hits), `rules/relevance.py` (action table + no-check
      list), Resolve on backgrounds (2,2,1,2,1,2), NPC attributes (Elara 1/3/2/2, Marcus 3/1/2/3), panic
      d6+Resolve+3>Stress, Last Light d6+Grit>=5+overflow, DC_ob = 12+2*act (+1 dark), Tome beat checks with
      provisional mods (Grit_T from Lumen) until the network supplies them, frames v2, protocol 2.
- [x] **M0 JAX partition (done):** `brain/voidlit/tomebrain/jaxpart/` + `calib/calib_rates.py` +
      `connectome_calibrated.toml`. Gates G1a-c, G2a-c, G3a, G4a-b, G5 pass with mutants
      (`brain/tests/tomebrain/`; slow ones run with `-m slow`, CI job `network-gates`). Departures logged in design
      §20 (CV2, fitted background SD, v_ER 2.0, Vogels eta 0.01, stable chaos, chunked delivery).
- [x] **M1 (done):** `tomebrain/bus.py`, `rng.py`, `partition.py`, `clock.py`, `persist.py`; G6a-e, G7, G22 pass.
      The anatomy seed is fixed (`ANATOMY_SEED`); the run seed keys noise only.
- [x] **M3 (done):** `tomebrain/sjpart/` (habit = Director prior, fused BG, three-factor + tags), `brain.py`
      (TomeBrain: Phase A/B, race commit), `tome/net_engine.py` (TomeNetEngine, now the default `--brain net`).
      G10a/b pass; W-invariance on the real brain; ~293 ms/turn median. Departures in design §20.
- [x] **M4 (done):** BindsNET amygdala, needs/drive/desperation/PAD readouts, LH reward current, K_ATP energy,
      quiet interval, network-driven Tome modifiers. G3b, G23, G24 pass. Slow gates then caught race regressions
      (G10a 0.79): fixed in the BG (dopamine as a gain on the whole D1 drive; slow 200-ms DA for excitability;
      plasticity needs a 50% slow-DA excursion; channel-balanced initial weights) -> 0.88/0.87. Design §20 M4.
- [x] **M5 (done):** `tome/ledger.py` (facts/topics/suggestions/probes from TOML, DisclosurePlan, TruthVerifier),
      senses asks/claims/addressed_tome (+ GBNF), `voice/style.py` (hidden registers), guard + coverage check on LLM
      lines, P fact slots + RHETORIC/FRAME drive the network Tome's speech. G11, G12 pass (fast); design §20 M5.
- [x] Full slow suite after the BG fixes: 29/29 pass (G10c, G11 200 games, G12 net, G23, G10a/b).
- [ ] Next: M6 cerebellum / imagination / upper-hand belief (G20), Daw arbitration kappa_h; M7 ToM / Witness /
      counters (G18, G19, G15).
- [ ] M8 corruption (NPC + player tiers, Rune, `resist` message, Unreal tap + CLI Enter); G11 puppet-claim clause.

- [x] **User decision (2026-10-21): "Authored only".** Narration and the Tome's own words always come from the rules'
      authored text (offline voice, verbatim ledger acts, authored wrappers and banks). The local LLM does only
      perception (sensory pass) and the NPCs' own dialogue. Reason: adversarial tests showed model prose can
      contradict resolved outcomes even under strict shape filters ("Then everything mends, bright and whole").

## Review of M4+M5 (2026-10-21)
Workflow review-m4-m5 confirmed 33 findings (4 refuted); details in scratchpad `review_full.json`, split into
`fix_truth.json` (18), `fix_persist.json` (7), `fix_net.json` (8). Headline: a regex guard cannot keep a free-writing
LLM truthful -> new rule "the LLM decorates; only the rules assert" (verbatim acts and core sentences, shape-filtered
framing/colour). Fix workflow `fix-review-findings` (run wf_9315a302-27e) implements them on branches fix/truth,
fix/persist, fix/network (worktrees), each adversarially verified.

## M6 plan
Design workflow `design-m6` (run wf_35f966cd-abe) produced a judged plan: scratchpad `m6_design.json` (`final`).
Key facts: p95 latency already 502-561 ms (> G17's 500); RHETORIC falls back on 34-37% of turns (runs to the cap).
Step 1 fixes RHETORIC commits (flat D1 drive) before the cerebellum. Zones 7-9 become an isolated microcomplex (CB-U)
fed only by a rules-side opposition code, on a Phase-A-local clock, so G20's bit-identity holds.

- [x] Review rounds complete (2026-10-22): rounds 2-4 + routing rounds 5-8 merged at fe82ea1 (fast suite 2,908
      passed). Routing ends with a closed grammar for consequential acts and a 1,874-case corpus; residual
      low/medium phrasings are documented in scratchpad round8/r8_verify.md and design §20.

## Next immediate action
Slow suite on 50f5c8f: 42/42 slow + balance gates pass (29 min, incl. G10c, G11 200 games). Now M6 per the judged plan in scratchpad
`m6_design.json` (`final.steps`), staged: step 0 (baseline + latency probe), step 1 (RHETORIC commit reliability:
34-37% fallback -> flat D1 drive; p95 latency <= 440 ms gate), step 2 (SJ persistence/digest hardening), then the
cerebellum module (step 3) and onward. Run each stage as an isolated implementer + adversarial verifier workflow;
keep a safety-net check-in armed against session limits.

## Pause (2026-10-22)
Stage A (steps 0-2) workflow running as run wf_33e6dbe2-1f7 (branch m6/stage-a, base 0c4fd46). On completion: review,
merge only if no serious regression, suites, design §20, push. Then STOP: the user paused work until Tuesday night after
their limit resets. Resume scheduled for 2026-10-28T04:00Z (trig_0194TY9WjtjFBPsCkCnG4Vu6) -> stage B (step 3, cerebellum).
Stage A prep notes (from the first, aborted run): persist VERSION 1->2 (engine SAVE_VERSION is 5, bump only if campaign
format changes); wall-clock timing must be opt-in (18 tests compare debug dicts); `fallbacks: int` in
net_engine.BRAIN_EXTRA_TYPES needs a type change with the version bump for a RHETORIC count.
