# The Tome's Brain: one hybrid spiking network (final design, v1)

Status: implementation-ready specification. The repository is unchanged; every prototype referred to here lives in the
session scratchpad under `design/bio/`, `design/rt/` and `design/gm/`.

Base design: the **biophysical** proposal, which both judges chose as the winner. Grafted onto it:

- from **engineering**: the fixed in-degree wiring used to seed calibration, the fused SpikingJelly basal ganglia,
  the race-to-threshold commit, absolute-tick noise, the bus assertion and lockstep gate, the engine-level gates, the
  upper hand as a cerebellar belief, and the counterfactual-twin corruption harness;
- from **cunning**: the Witness column, the adversarial repertoire, the speech-act truth guard, the two-phase turn,
  the d6 passive events, the balance method, Daw arbitration, successor features and rate limits.

Every flaw the judges raised is fixed or turned into a blocking gate (§2).

Evidence tags used throughout:

| Tag | Meaning |
|---|---|
| **[M]** | Measured in a prototype this session (the source file is named). |
| **[P]** | Projected from measurements; not yet measured on the final network. |
| **[C]** | A calibration target, not yet achieved. The gate that enforces it is named. |

> **Honest headline.** The judges verified the biophysical prototype's physics:
>
> - the XOR cell, Graupner-Brunel plasticity, Vogels iSTDP and WindowedMSTDPET;
> - the energy-to-hunger chain;
> - one-clock determinism, save/load, and the silent NPC projection.
>
> They also verified that its network is **not yet a working circuit**: every point population except the L2/3 cells
> and the thalamic relay sits at 0 Hz, inhibition is inert, and the stability mutant cannot be told apart from the
> intact network. **Rate calibration (G5, with G4's mutant) is therefore Milestone 0. It blocks every behavioural
> claim in this document.**

---

## 0. Traceability: the user's direction, where it is met, and which gate proves it

The user's relayed words: *"Like D&D the character has a set of Attributes and certain attributes plus a roll of die
will decide whether the corruption is strong enough to force these things, so on top of the corruption tracking this
is a second layer. and just about anything and everything that is an action taken by any character should have
these invisible attribute and die roll skill checks relevant to the action."*

| # | Binding direction | Where it is met | Gates |
|---|---|---|---|
| U | Attributes + a die roll decide whether corruption is strong enough (a second layer on top of tracking). Every action by any character gets an invisible check relevant to it. | §9 (check tables for player, NPCs and the Tome); §10 (layer 1 tracks exposure or desperation; layer 2 is the d20 check against Resolve, and its margin sets the strength) | G21 check coverage, G9, G16 |
| 1 | One singular network that feels, needs and schemes, replaces the Director, and keeps NPC brains as separate minds | §3, §5, §6, §8 | G6, G7, G10, G23 |
| 2 | SpikingJelly + BindsNET + JAX as one connectome, one clock and one shared state | §5 (one 1-ms master clock, conservative synchronisation, one spike bus, one save) | G6a-e, G7 |
| 3 | Never a falsehood about a resolved fact; dice decide outcomes; omission allowed; self-serving suggestions; the Lumen is its survival; Marcus and Elara want the Tome | §11.3-11.5 (FactLedger, speech-act verifier), §8.1 (Lumen to ATP), §6.13 (the "loss" signal) | G11, G3b, G24 |
| 4 | Opinion and mood are never displayed; they leak only through wording | §11.6 (VoiceStyle goes to the voice only), §11.7 (frames) | G12 |
| 5 | Local llama.cpp voice with an offline fallback; loopback WebSocket; NPC brains; tests; the Director's pacing as the prior | §11, §12, §15 | G10, G22, existing tests |
| 6 | NPC corruption: their own minds plus a capped, bursting, refractory projection that grows with desperation; puppets may lie | §10.1, §6.14 | G8, G9 |
| 7 | Player corruption after extended play; Tier 1 loop; Tier 2 timed resist | §10.2, §11.8 | G16 |
| 8 | d20 + attribute for direct actions, d6 for passive events, +d6 for an upper hand against the Tome, corruption vs Resolve, four attributes, dice invisible, the Tome's own modifiers and upper-hand assessment | §9, §8.6, §8.7 | G13, G20, G21 |
| 9 | Brutal: about 20% escapes for a sensible bot | §9.8 | G14 |

---

## 1. What was synthesised, and every judge flaw resolved

### 1.1 Grafts adopted

| From | Graft | Section |
|---|---|---|
| Both judges | G5 rate calibration is Milestone 0, seeded from engineering's functional fixed in-degree table. G4's no-inhibition mutant must diverge, and gather overflow must be zero. | §7.5, §15 |
| Engineering | Basal ganglia move from BindsNET into one fused SpikingJelly LIF population (CSR recurrent matrix). BindsNET keeps the amygdala and the puppeteer afferents. | §6.11 |
| Engineering | Race-to-threshold commit on VA/VL with a 120-ms minimum and a 240-ms cap, replacing the softmax-plus-external-RNG readout. Trial-persistent MSN excitability noise is drawn inside the network from its own seeded state. | §5.4, §6.11 |
| Engineering | Noise keyed by absolute tick; a `Bus.read` causality assertion; a bitwise gate comparing windowed D=4 with lockstep D=1; a persistent JAX compilation cache. | §5 |
| Engineering | A `TomeNetEngine` subclass that swaps only the beat choice. End-to-end gates: latency, full-frame determinism, and save then load into a different-seed instance. Game-level Director preservation. | §3, §12, G7, G10c |
| Engineering | The upper hand is a cerebellar DCN belief, with a rules-verifiable facts floor and a calibration gate. | §8.7, G20 |
| Engineering | The counterfactual-twin harness for NPC corruption: a ghost, a never-corrupted twin and a 2% input-jitter floor, at least 60 turns, both NPCs. Whispers fire only on burst turns and never enter the NPC's reward. | §10.1, G9 |
| Cunning | A dt-convergence gate, with the Hines/Rush-Larsen HH cell as the test reference. | G1b |
| Cunning | The Witness: the dCaAP XOR detects say/do mismatch. New `asks`, `claims` and `addressed_tome` senses, plus a "repeat the question" affordance. | §6.2, §8.5, §11.2 |
| Cunning | The adversarial repertoire: withholding as investment, malicious compliance, habit exploitation (restricted so it never contradicts dice), true puppet claims, `hint_ok`, the speech-act truth guard, and the affect self-report guard. | §8.8, §11.3-11.5 |
| Cunning | Daw-2005 habit/goal arbitration. | §6.11 |
| Cunning | CA1 successor features. | §6.4 |
| Cunning | Balance method: naive and smart bots measured together. DC 12+2·act with a Lumen interval of 5 is the base. | §9.8 |
| Cunning | Rate limits: corruption successes at least 2 turns apart; Tier 1 at most once per 3 turns; Tier 2 at most once per chapter. NPC self-notice. | §10 |
| Cunning | Panic and Last Light as passive d6 rolls (the user's list). | §9.5 |
| Cunning | Two-phase turn: Phase A before dice, Phase B after resolution, on the same unbroken clock. | §5.3 |
| Cunning | `tension` removed from frames. Cross-run memory raised as an open question for the user. | §11.7, §19 |

### 1.2 Flaws raised by the judges, and their fixes

| Flaw (who) | Fix | Gate |
|---|---|---|
| Biophysical network effectively disconnected (0 Hz populations, AMPA 0.03 nS, inert inhibition) (J1, J2) | Milestone 0 calibration from engineering's in-degree table. Rate targets per population. The no-inhibition mutant must fail. | G5, G4 |
| Gather overflow (106 at Lumen 6) undisclosed (J1, J2) | Overflow counter kept in state; 0 required over 50 calibrated turns. Delivery buffer sized at 4× the measured peak. | G5 |
| Cortex silent at Lumen 0, so the Tome is comatose (J2) | K_ATP g_max recalibrated to about 4 nS so that cortex runs at 40-60% of baseline at Lumen 0. | G3b |
| Over budget (386-462 ms for the Tome alone, plus NPCs) (J1, J2) | BG moved to fused SJ; JAX delivery is event-driven; adaptive commit (180-300 ms of brain time, not a fixed 248). Projected 240 ms median. | G17 |
| XOR spike count depends on dt; the window sat at 1.05× threshold (J1) | Conductance dCaAP with Gidon's amplitude law. The XOR verdict must hold across a ±20% pathway-amplitude band at both production and reference dt. | G1b, G1c |
| Corruption dose-response was a cliff, n = 15 (J1, J2) | A graded strength law. Measured against a twin and the jitter floor, 60+ turns, both NPCs, 4 desperation levels. | G9 |
| Beat sampled outside the network (bio, eng) (J1, J2) | Race-to-threshold on JAX VA/VL cells. The Gumbel excitability is drawn by the SJ partition's counter-based RNG and saved. | G10a |
| Director KL unmeasured (J1, J2) | G10a/b/c. | G10 |
| Balance dice-only, 40 runs, no naive separation (J1, J2) | Both bots, 200 runs, starting from the measured 24%/7% base. | G14 |
| Quiet-interval astrocyte scan hard-coded to 1000 steps (J1) | Step count = Q / 20 ms, derived from Q and asserted. | G3a |
| Corruption attached at runtime to the NPC Network (J1) | Kept (it is literally spikes into their net), but the BindsNET version is pinned, a layer-order test is added, and G8 runs in CI. | G8 |
| Panic used a d20 (J2) | Panic is d6 + Resolve score + 3 > Stress. | G21 |
| Habit SNN with persistent membranes changed Director statistics (J2) | The habit SNN runs exactly as trained (16 steps, reset before and after, the same Poisson encoding), placed on 16 clock ms. | G10a |
| Cunning: the counter bank was not GB (rescaled per turn, tautological mutant) | Guards use the physiological GB DP set. Clustering comes from astrocytic IP3 integration across quiet intervals plus replay. Mutants: Γ≡0, Γ≡1, θp=1.6. | G15 |
| Cunning: needs scripted; puppet tuft drive literally (50+200·desp) Hz | Needs are physiological readouts. Puppeteer drive arrives through the hunger and arousal pathways (LH orexin to PVT to P tufts; CeA to LC). Desperation is a readout, never an input. | G23 |
| Cunning: suggestions exempt from checks (J1, J2) | Suggest and frame are a Guile_T check against passive Insight; a failure adds a true tell. | G21 |
| Cunning: counters punish resisting (shorter tap window) (J2) | The tap-resist counter changes the Tome's strategy (it stops investing in bends), never the window. | G15, G16 |
| Engineering: corruption was a Python state machine with direct momentum edits | Spikes from the Tome's real P-column puppeteer raster drive the NPC net. The state machine is reduced to the rules layer (resource x_c, refractory, check). | G8, G9 |
| Engineering: cap 0.75 and the 87-90% lock-in cliff (J1, J2) | Cap 0.35. No grafts. Whisper only on burst turns, never in the NPC's reward. Eligibility quarantine. | G9 |
| Engineering: escalate and hazard stayed passive d6 (J1, J2) | They are the Tome's Grit_T d20 checks. | G21 |
| Engineering: "Dominion" replaced Grit (J1) | The Tome uses the user's four attributes. | §8.6 |
| Engineering: stability gate had no mutant; 0.01 mV kicks produced zero divergence (J1) | G4 adds a microscopic-chaos clause: a kick must change spike timing while rates and decisions stay stable. A no-inhibition mutant is required to fail. | G4 |
| Engineering: glia barely move in 120-ms turns (J1) | The 20-s quiet interval (slow-only, adiabatic) gives astrocytes, ATP, CBF, myelin and GB drift physiological time. | G3 |
| Cunning: prototype rates implausible (L2/3 44 Hz, GPi 0 Hz) yet called plausible (J1, J2) | Not reused. G5 targets apply to the final network. | G5 |
| Cunning's habit exploitation would let the Tome hide what a successful search found, contradicting "dice decide" | Restricted: the Tome only controls when its own omittable knowledge is volunteered. Anything a successful action uncovers is an outcome fact and is always narrated. | G11 |

---

## 2. Architecture overview

```
                         ONE NETWORK - one 1 ms master clock - one connectome table - one save blob
 ┌────────────────────────────── JAX partition (~2,150 cells, dt 0.25 ms) ───────────────────────────────┐
 │ cortex: S reader · T mind-reader (+Witness) · P schemer (+fact slots, agenda, puppeteers, counters)     │
 │         · I insula      thalamus: core · matrix (P matrix = VA/VL of 18 BG channels) · TRN · PVT        │
 │ hippocampus DG/CA3/CA1   nuclei VTA/RMTg/LC/DRN/NBM/LHA-ORX/PVN   astrocytes · ATP/K_ATP/NAD · myelin   │
 └───────▲──────────────┬───────────────────────▲────────────────────────────┬─────────────────▲─────────┘
   GPi→VA/VL,DCN→VL,     │ L5/CA1/TC → striatum,  │ BLA→tufts, CeA→LC/PVN/VTA  │ L5 IT/CA1/liar  │ DA NE 5HT ACh
   VS→RMTg, PC→LC        │ mossy fibres            │                            │ → BLA ctx      │ (graded rows)
 ┌───────┴──────────────▼───────────────┐ ┌──────┴────────────────────────────▼─────────────────┴─────────┐
 │ SpikingJelly (~3,380 cells, dt 1 ms)  │ │ BindsNET (72 cells + inputs, dt 1 ms)                          │
 │ habit cortex = DirectorSNN (frozen)    │ │ amygdala: BLA bond/threat/contempt/loss, ITC, CeA              │
 │ fused BG: PACE 8 · RHETORIC 6 · FRAME 4│ │ WindowedMSTDPET (per-assembly teaching signal)                  │
 │ + VS critic; cerebellum + imagination │ │ puppeteer afferents attached to Elara's / Marcus's own BindsNET │
 └───────────────────────────────────────┘ └─────────────────────────────────────────────────────────────────┘
      cross-partition spike bus: ring buffers, every delay ≥ W = 4 ms (conservative sync, NEST/MUSIC style)

 rules engine (authoritative, rolls every die) ◀── TomeIntent / Phase-A beliefs / modifiers ── network
 rules engine ── TomeInput (resolved facts only, never dice) ──▶ network
 NPC brains (separate minds, unchanged code) ◀── capped puppeteer spikes (after a successful corruption check)
 voice (llama.cpp local or offline) ◀── facts + disclosure plan + suggestion + hidden VoiceStyle
```

| Partition | Framework | Regions | Neurons |
|---|---|---|---|
| J | JAX 0.12.0 (float32, CPU, single-threaded XLA) | 4 columns, thalamus (incl. VA/VL, PVT), hippocampus, nuclei, glia, energy, myelin | ≈ 2,152 (480 three-compartment, 208 two-compartment) |
| S | SpikingJelly 0.0.0.0.14 / torch 2.14 | habit cortex (DirectorSNN 136 + 32 habit cells), fused BG 894, cerebellum 2,318 | ≈ 3,380 |
| B | BindsNET 0.3.4 / torch 2.14 | amygdala 72 + Input afferents; puppeteer afferents inside the NPC nets (2 × 16) | 72 (+32) |
| **Tome total** | | | **≈ 5,600 neurons, ≈ 0.2 M synapses [P]** |

---

## 3. File and module layout (under `brain/voidlit/`)

```
voidlit/
  tomebrain/                       NEW - the Tome's single network
    __init__.py                    TomeBrain facade: phase_a(), phase_b(), quiet_async()/join(), state_dict()/load_state_dict()
    clock.py                       MasterClock (t_ms, turn, phase), window scheduler, async JAX dispatch, barrier
    bus.py                         SpikeBus: per-source ring buffers (128 ms), graded rows, written-tick marks, read() asserts causality
    connectome.py                  declarative Projection table → partition matrices + bus edges; sha256 of the compiled connectome
    connectome_calibrated.toml     per-pathway gains and background currents fitted by G5 (versioned, hashed)
    rng.py                         counter-based Philox streams keyed (seed, partition, stream, absolute tick)
    encode.py                      TomeInput → thalamic currents, US vector, IO error spikes, glucose
    readout.py                     needs, drive, reward, desperation, PAD, OCC, opinion, ToM, mods, upper hand, VoiceStyle
    jaxpart/
      cells.py                     AdEx point, 3-comp human L2/3 (dCaAP), 2-comp L5 ET/CA1 (BAC), T-type; exponential Euler
      synapses.py                  event-driven delivery (compaction, padded out-edges, 32-ms future-conductance ring), TM STP
      plasticity.py                Graupner-Brunel (DP set), Vogels iSTDP, CA1 successor-feature rule, guard metaplasticity
      glia.py                      De Pitta astrocytes, myelin rule, microglial tags/prune/rescue
      energy.py                    glucose, per-cell ATP, K_ATP, CBF, NAD
      nuclei.py                    VTA, RMTg, LC, DRN, NBM, LHA-ORX, PVN; volume transmission
      network.py                   build(); window(state, inputs) = jit(lax.scan over W ms × substeps); quiet(state, Q)
    sjpart/
      habit.py                     DirectorSNN (loads cortex/director.pt unchanged) on 16 clock ms + 32 habit LIF cells
      basal_ganglia.py             fused LIF population (per-neuron tau), CSR recurrent matrix, 3 loops, VS critic, 3-factor + tags
      cerebellum.py                Marr-Albus-Ito forward model, 10 microzones, 8-copy imagination batch on the clock
      nodes.py                     PerNeuronLIF(BaseNode) with tensor tau / threshold / bias
    bnpart/
      amygdala.py                  BindsNET Network: ctx Input, US Input, BLA(4×12), ITC, CeA
      windowed_rule.py             WindowedMSTDPET (closed-form per window; equivalence-tested)
      puppeteer.py                 PuppeteerAfferent, attach()/detach(), whisper snapshot, eligibility quarantine
    persist.py                     versioned save/load of the whole brain (§13)
    calib/                         offline tools: calib_rates.py (G5), calib_xor.py (G1c), calib_corruption.py (G9)
  tome/
    engine.py                      TomeEngine: rules pipeline (kept); turn() refactored into hook points
    net_engine.py                  NEW TomeNetEngine(TomeEngine): two-phase turn with TomeBrain; Director-free
    ledger.py                      NEW FactLedger, Fact, SpeechAct, TruthVerifier
    profile.py                     NEW player profile: exposure_turns, runs, escapes, settings, cross-run memory file
    frames.py                      status trimmed; dice/event blocks retired; resist message
  rules/
    attributes.py                  NEW Attribute {GRIT, GUILE, INSIGHT, RESOLVE}; mod(score) = 2·score − 2
    checks.py                      NEW direct d20 checks, passive d6 checks, upper-hand d6, degree→PoolRoll adapter, CheckLog
    relevance.py                   NEW the action → (die, attribute, opposed-by, DC rule) table (§9.3-9.6); exhaustive by test
    corruption.py                  NEW rules side: NPC resource x_c, refractory, corruption check; exposure tiers; bend check
    counters.py                    NEW rules effects of permanent counters (§8.9)
    character.py                   + resolve; roll() delegates to checks
    panic.py                       d6 + Resolve
    dice.py                        kept for the Lumen depletion pool; roll_pool kept only for the adapter and legacy tests
  cortex/                          UNCHANGED (director.py, teacher.py, features.py, director.pt) - the habit prior's source
  limbic/                          UNCHANGED (NPC brains, contagion, dendrites)
  senses/
    state.py                       + asks (topic id | "vague" | "unclear" | "none"), claims (claim id | "none"), addressed_tome
    grammar.py                     + per-scene enums for asks/claims; "tome" is always a legal target
    heuristic.py                   + question and claim lexicons
  voice/
    style.py                       NEW VoiceStyle → craft instructions (LLM) and register keys (offline)
    guard.py                       NEW output guard: ledger/withheld-term check, affect self-report, dice/number lexicon
    prompts.py                     + craft line, claims list, possessed line; no emotion words
    narrator.py                    + "key@register" bank lookup with fallback; suggestion/answer/hint/tell templates
    requests.py                    + style, disclosures, suggestion, answers, hints, tells, puppet_claim
  content/hollow_lantern.toml      + resolve per background; NPC attributes; facts; agenda (Rune, sealed bulkheads);
                                   claims; NPC tome_hook; puppet claims; styled phrase banks
tests/tomebrain/                   NEW gates (§15); existing tests kept or adapted (§15.3)
```

Dependencies to add (exact pins, local only): `jax==0.12.0`, `jaxlib==0.12.0`, `ml_dtypes==0.6.0`,
`opt_einsum==3.4.0`, `scipy==1.18.1` (already importable from `reconcile/jaxvenv`; same Python 3.13 and numpy 2.5.4).
Also pin `bindsnet==0.3.4` and `spikingjelly==0.0.0.0.14`. The JAX persistent compilation cache lives in a local
directory (`~/.cache/voidlit/jax/<build-hash>`).

---

## 4. Glossary of turn-level quantities (all hidden)

| Symbol | Meaning |
|---|---|
| `W` | Synchronisation window: 4 ms. Also the minimum cross-partition delay. |
| Phase A | 60 ms of brain time before the player's action is resolved. |
| Phase B | 120-240 ms after resolution, ending at the race commit. |
| `Q` | Quiet interval: 20 s of slow-only brain time after each turn. |
| `h_i` | Need levels (survival, affiliation, certainty, competence) in [0, 1]; the set point is 1. |
| `D(H)` | Keramati-Gutkin drive (§8.2). |
| `δ_desp` | Desperation readout in [0, 1]. |
| `Grit_T, Guile_T, Insight_T, Resolve_T` | The Tome's check modifiers in [−2, +6]. |
| `κ_h` | Habit/goal arbitration gain. |
| `x_c` | Per-NPC corruption resource in [0, 1]. |
| `s` | Delivered corruption strength, at most `S_MAX` = 0.35. |
| `E` | Player exposure, counted in turns (profile). |

---

## 5. One network, one clock: the co-simulation scheme

### 5.1 Master clock and partition steps

- One integer `t_ms` is the bus resolution and the time base of every partition.
- **JAX:** membranes take 4 substeps of 0.25 ms per bus ms (0.5 ms if G1b passes at 0.5). Synaptic delivery,
  STP, GB, glia and energy update every 1 ms.
- **SpikingJelly and BindsNET:** dt = 1 ms. No multi-rate partitions (the judges flagged cunning's 2 ms BindsNET step).
- **NPC brains:** separate minds on their own 100-ms clock. They are run by the engine between Phase A and Phase B.
  The Tome perceives their readouts, never their internals.

### 5.2 Conservative synchronisation (exact)

- Every cross-partition projection has an integer axonal delay `d ≥ W = 4 ms`.
- Window `k` covers ticks `[4k, 4k+4)`. A partition integrating window `k` reads only bus rows for ticks ≤ 4k+3−d
  ≤ 4k−1, which were all written in earlier windows.
- Partitions therefore run in any order, or overlapped. `Bus.read(src, t, d)` asserts `written_upto[src] ≥ t − d`.
- Myelination may change delays but clamps cross-partition delays at ≥ W.
- Measured: windowed D=4 equals lockstep D=1 bitwise [M rt/m6_window.py, rt/cosim.py]; reversed partition order is
  bitwise identical [M gm/hybrid.py]; async equals sequential [M bio/one_clock.py].

### 5.3 The turn timeline (two phases on one unbroken clock)

```
brain time ─▶  [Q of previous turn: slow-only]  │ Phase A 60 ms │ (rules resolve: 0 brain ms) │ Phase B 120..240 ms │ [Q 20 s] ...
                                               ▲               ▲                              ▲  ▲                  ▲
           retroactive DA pulse (r_{t-1}) ─────┘   Phase-A readouts                imagination│  race commit        tag transfer,
           IO error spikes (first 20 ms)           (upper hand, bend/puppet         (first 40 ms)│ (≥120 ms)          quiet integration
                                                    bursts, mods from last Q)       habit SNN 16 ms
```

The fast dynamics are never paused or reset between phases; only the thalamic input changes. During Q the fast
state is frozen and slow variables integrate (adiabatic elimination, §5.6).

Engine sequence inside `TomeNetEngine.turn(text)`:

1. **senses:** `SensoryState` (+ `asks`, `claims`, `addressed_tome`).
2. **Phase A** (`brain.phase_a(TomeInput)`). Input: senses, the declared action (`declines_agenda`,
   `opposes_tome`), last turn's IO errors and US values, NPC percepts, body state. Output: `PhaseAOut`
   (upper-hand beliefs per actor, bend proposal, puppeteer proposals and rasters, and the modifiers frozen at the
   last Q).
3. **Bend check (rules)**, if proposed and the tier gates are open: Tier 1 marks the action to loop; Tier 2 suspends
   the turn on a `resist` exchange (§11.8).
4. **Rules resolve the player's action** with invisible checks (§9), using `PhaseAOut.upper_hand`.
5. **Corruption checks (rules)** for proposed NPC corruptions. On success, `puppeteer.load(npc, raster, s)`.
6. **limbic:** NPCs perceive (their own brains; the afferent delivers spikes if loaded). NPC actions are checked.
   After a burst turn, `puppeteer.quarantine(npc)` runs.
7. **pressure:** panic (d6), Lumen gutter (d6), Last Light (d6), passive notices (d6).
8. **Phase B** (`brain.phase_b(TomeInput)`). Input: the resolved outcome band, NPC readouts, passive events, US,
   ledger slots. Output: `TomeIntent` (beat, rhetoric act, frame, disclosure plan, suggestion, probe, puppet
   claims, VoiceStyle, newly permanent counters).
9. **The Tome's action checks** for the committed beat and rhetoric (§9.6). Then `_beat()` applies its unchanged
   effects (gated by the check).
10. **voice:** narration from ledger facts only, then the guard, then the offline fallback. NPC dialogue
    (possessed flag if s ≥ 0.2).
11. **frame:** no dice, pips, clocks, events or affect (§11.7).
12. **`brain.quiet_async()`** starts the quiet interval in a worker thread. It is joined at the start of the next
    turn, or before any save. The results are identical whenever it runs, because it is a pure function of the
    frozen state.

### 5.4 Window execution and the race commit

For each 4-ms window:

1. Each partition gathers its bus inputs (numpy row reads, at most 2 ms per turn [M rt]).
2. JAX `window(state, i_bus[W,N], graded[W,·])` is dispatched asynchronously (returns in about 0.3 ms [M rt/m8]).
3. On the Python thread, SpikingJelly runs one multi-step call over W, then BindsNET runs `Network.run` for W steps
   with the windowed rule.
4. Join on the JAX spikes (the only synchronisation point), then publish all `[W, n]` spike blocks and graded rows.

Commit rule (Phase B):

- Each loop (PACE, RHETORIC, FRAME) owns 4 VA/VL cells per channel in `P.TC_mtx`.
- A loop commits to the first channel whose 4 cells emit **K = 6 spikes within any 20-ms sliding window**, counted
  only after **t ≥ Phase B start + 120 ms**.
- Phase B ends when PACE and RHETORIC have committed, or at the **240-ms cap**. FRAME is read only if RHETORIC
  committed `SUGGEST` or `DISCLOSE`.
- At the cap, the channel with the most VA/VL spikes wins and a `fallback` counter is incremented (G10a requires
  ≤ 5% fallback). Engineering's prototype committed in 42/42 turns with a median race latency of 46 ms [M rt/m10].

### 5.5 Determinism and noise

- Every stochastic draw in every partition comes from a **counter-based Philox stream keyed by
  (seed, partition, stream_id, absolute tick)**: `jax.random.fold_in` in JAX, and `numpy.random.Philox(counter=...)`
  converted with `torch.from_numpy` in the torch partitions.
- Draws are therefore independent of window size, call batching and save points. This is required for the D=4 vs
  D=1 bitwise gate, which stateful torch generators would break.
- Gumbel MSN excitabilities are keyed (seed, turn, channel).
- The rules keep the engine's `random.Random` (dice), unchanged.
- `torch.set_num_threads(1)`; XLA single-threaded; float32 everywhere. Weak-typed JAX leaves are forbidden (they
  caused a 3.1-s recompile [M bio]).
- Bit-exactness holds within one build (jax/jaxlib/torch/BindsNET versions and CPU flags). The save header records
  the build (§13).

### 5.6 The quiet interval Q (slow-only, adiabatic)

Q = 20 s of brain time per turn. It stands for the player's reading and typing time.

- **Fast state is frozen:** membranes, conductances, STP, eligibility. It is not reset: working memory carries from
  turn to turn.
- **Slow variables integrate** over Q against an explicit post-turn activity profile: r(t) = r_base + (r̄_turn −
  r_base)·e^{−t/τ_post}, with τ_post = 2 s (reverberation decaying to the calibrated resting rates). Nothing is
  clamped at turn-mean rates for 20 s, which would be a hidden compression.
  - ATP: exact exponential solution.
  - Astrocytes: Li-Rinzel/De Pitta in 20-ms steps; step count = Q/20 ms = 1000, derived and asserted.
  - CBF.
  - NAD.
  - GB: calcium-free bistable drift only. No calcium in Q, so τ = 150 s is not secretly compressed.
  - Tonic neuromodulator EMAs.
  - Synaptic tags (§6.11) decay by e^{−Q/60 s}.
- **Once per turn:** myelin velocity update; microglial tag accumulation; a sharp-wave replay window (100 ms of
  hippocampus plus guard pathways, cortex clamped quiet) if the turn's salience |RPE| + NE exceeds its threshold.
- **At chapter boundaries:** 5 replay windows; microglial pruning and rescue.
- Cost: 0.6 ms after compile [M bio], plus up to 5 × 100 ms of hippocampal replay at chapter boundaries (in the
  worker thread).

---

## 6. Regions: equations, parameters, framework

All JAX compartments use **exponential Euler with Jacobi coupling**:

> V' = V∞ + (V − V∞)·exp(−dt·g_tot/C), with V∞ = (Σ g_k E_k + I)/g_tot.

Conductances and neighbouring compartment voltages are frozen over the substep. Each g_tot contains the
compartment's own leak and all of its coupling conductances, so the scheme is strictly diagonally dominant and cannot
blow up at any dt. Measured: no NaN at dt 0.05-1.0 ms [M bio/cells.py, rt/tome_jax.py]. Synaptic conductances decay
exactly.

### 6.1 Cell models (JAX, `jaxpart/cells.py`)

**AdEx soma (every cell):**

> C dV/dt = −g_L(V−E_L) + g_L Δ_T exp((V−V_T)/Δ_T) − w + I_syn + I_axial + I_ext + I_bias
> τ_w dw/dt = a(V−E_L) − w

- A spike is detected at V ≥ −30 mV: V ← V_r, w ← w + b, then refractory t_ref.
- The exponential term is a current evaluated at the frozen V, with its argument clamped at 10.

**Human L2/3 pyramidal cell (3 compartments: soma, basal, apical tuft).** Cm 0.45 µF/cm², Rm 38,907 Ω·cm², τ 17.5 ms
(Eyal 2016); spine-corrected area as in the engineering prototype [M rt].

| Parameter | soma | basal | apical tuft |
|---|---|---|---|
| C (pF) | 72 | 162 | 126 |
| g_L (nS) | 4.1 | 9.3 | 7.2 |

- Axial coupling: g_sb 30 nS, g_sa 12 nS.
- E_L −72, V_T −50, Δ_T 2 mV; a 2 nS, b 40 pA, τ_w 150 ms; V_r −58 mV; t_ref 2 ms.
- NMDA on basal and tuft: Jahr-Stevens block B(V) = 1/(1 + [Mg]/3.57·e^{−0.062V}), [Mg] = 1 mM.
  NMDA:AMPA = 0.5 (basal) and 0.6 (tuft), multiplied by (1 + 0.5Γ_astro) (D-serine precision).
- Tuft Kv rectifier: g_Kv = 8 nS/(1 + e^{−(V_a+35)/5}), E_K −90 (clamps passive plateaus).
- Back-propagating AP into the tuft: I_bAP = 60 pA·e^{−t/1 ms} after each somatic spike.

**dCaAP** (Gidon et al. 2020: amplitude largest at threshold, falling as drive grows). Conductance form, so the
driving force is real:

> I_d = g_d · m · h · j · (E_Ca − V_a)

| Gate | Steady state | Time constants |
|---|---|---|
| m (activation) | m∞ = σ((V_a − θ_d)/2.5) | τ_m 2 ms |
| h (drive-dependent inactivation) | h∞ = exp(−max(0, u/u_th − 1)/τ_K), Gidon's amplitude law | τ_h 0.5 ms falling, 25 ms recovering (no offset rebound) |
| j (self-termination) | j∞ = σ(−(V_a + 5)/4) | τ_j 20 ms |

- u = apical excitatory synaptic conductance (AMPA + NMDA·B(V_a)), low-passed with τ_u 0.5 ms.
- A 40-ms refractory flag follows each onset.
- Starting values: g_d 20 nS, E_Ca 120 mV, θ_d −36 mV, u_th 15 nS, τ_K 0.30 [C G1c]. θ_d, g_d and τ_j come from the
  engineering cell (XOR 0/3/3/0 at dt 1-0.05 [M rt/t_xor.py]); τ_K is bio's fitted Gidon decay (XOR verified at dt
  0.05/0.25/1.0 [M bio/gate1.py]).
- One pathway at u ≈ 1.1-1.2·u_th gives h ≈ 0.5-0.7, a dCaAP and somatic spikes. Both pathways together
  (u ≈ 2.3·u_th) give h ≤ 0.02, so no dCaAP: **XOR**.

**L5 ET / CA1 pyramidal cell (2 compartments, BAC firing; Larkum 1999)** [M bio cells.py `pc5`]:

- Soma 150 pF / 7.5 nS; tuft 60 pF / 3 nS; g_sa 6 nS.
- E_L −70, V_T −50; a 2 nS, b 80 pA, τ_w 100 ms; V_r −58 mV.
- BAC Ca²⁺ spike on an upward tuft crossing of −35 mV (refractory 40 ms): I = 700 pA·n·(e^{−t/25} − e^{−t/2}),
  monotonic (no XOR).
- bAP 250 pA·e^{−t/1 ms}. Tuft Kv 10 nS.
- A "burst" is 2 or more somatic spikes within 15 ms.

**Point cells** (AdEx, bio `KINDS` [M]). Units: C pF, g_L nS, a nS, b pA, τ_w ms.

| Kind | C | g_L | E_L | V_T | Δ_T | a | b | τ_w | V_r | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| PV | 90 | 9 | −68 | −48 | 1.5 | 0 | 0 | – | −60 | fast-spiking, t_ref 1 |
| SST | 100 | 5 | −65 | −52 | 2 | 4 | 40 | 200 | −55 | adapting, low threshold |
| VIP | 60 | 4 | −65 | −50 | 2 | 2 | 20 | 100 | −56 | |
| NGF (L1 LAMP5) | 80 | 5 | −70 | −45 | 2 | 1 | 10 | 300 | −60 | late-spiking |
| RS (L4, L5 IT, L6 CT, CA3) | 150 | 8 | −70 | −50 | 2 | 2 | 40 | 120 | −58 | |
| TC / TRN | 200 / 150 | 10 / 8 | −65 / −70 | −50 | 2 | 0 | 10 / 20 | 50 / 60 | −55 / −58 | T-type g_T 60 / 90 nS |
| DG | 60 | 5 | −78 | −48 | 2 | 2 | 40 | 100 | −60 | sparse |
| DA pacemaker | 150 | 6 | −60 | −48 | 3 | 4 | 50 | 300 | −55 | I_bias 140 pA (about 4 Hz), t_ref 5 |
| LC / DRN / NBM pacemakers | 150 | 6 | −60 | −48 | 3 | 4 | 50 | 400 | −55 | I_bias 120 pA (1.5-3 Hz) |
| LHA-ORX (glucose-inhibited) | 150 | 8 | −70 | −50 | 2 | 2 | 40 | 120 | −58 | I_hunger = 260 pA·(1 − glucose); no K_ATP |

T-type current (Destexhe 1996): I_T = g_T m∞² h (V − 120), with m∞ = 1/(1+e^{−(V+57)/6.2}),
h∞ = 1/(1+e^{(V+81)/4}), τ_h = 8 + 60/(1+e^{(V+78)/3}) ms. Inhibition from TRN or GPi produces rebound bursts.

Receptors (exact exponential decay):

| Receptor | τ | E_rev |
|---|---|---|
| AMPA | 2 ms | 0 mV |
| NMDA | 100 ms | 0 mV (with B(V)) |
| GABA_A (soma) | 6 ms | −70 mV |
| GABA_A slow (tuft; SST, NGF) | 20 ms | −70 mV |
| GABA_B | 150 ms | −90 mV |

Background: OU current noise (τ 5 ms; SD 35 pA for E cells, 25 pA for I cells) plus a per-population tonic μ (fitted
by G5), drawn from the absolute-tick Philox stream.

**Neuromodulator actions** (concentrations from §6.5):

- NE lowers V_T by 2 mV per unit (gain) and multiplies TRN gain by (1 + 0.8·NE).
- ACh scales tuft AMPA from matrix and feedback by (1 − 0.5·ACh) (Hasselmo: trust sensory evidence over priors).
- DA (D1) raises P-column NMDA gain by (1 + 0.3·DA_tonic) (persistent activity).
- 5-HT biases P L5 ET toward withholding: −20 pA·5HT on the puppeteer and bend cells.

### 6.2 Neocortex: four six-layer columns (JAX)

Base column template (S, I), 278 cells:

- L1 NGF 8
- L2/3: PC 96, PV 12, SST 8, VIP 8
- L4: SS 40, PV 6
- L5: ET 24 (2-comp), IT 24, PV 6, SST 6
- L6: CT 32, PV 4, SST 4

78% excitatory. Neurons are ordered 3-compartment, then 2-compartment, then point cells, so dendritic targets are
array prefixes.

| Column | Size | Additions | Game function |
|---|---|---|---|
| **S** reader | 278 | none | Perceives the turn. L2/3 mismatch (basal evidence vs tuft expectation) gives the precision-weighted prediction error behind the certainty need. |
| **T** mind-reader (TPJ/dmPFC) | 348 | L2/3 PC 144 = 6 motive assemblies × 16 + 8 Witness claim slots × 6; PV 18, SST 12, VIP 12; L5 IT 32 = 24 + 8 liar readout cells | Theory of Mind (§8.4); the Witness (§8.5). |
| **P** schemer (PFC) | 444 | L2/3 PC 144 = 8 fact slots × 12 + 4 agenda assemblies × 12; PV 18, SST 12, VIP 12; L5 ET 72 = 24 action + 2 × 16 puppeteer (Elara, Marcus) + 16 bend; L5 IT 72 = 24 + 48 counter cells (12 tactics × 4); L5 PV 10, L5 SST 10 | Agenda and actions; omittable-fact slots; corruption and bend output; permanent counters. |
| **I** insula (interoception) | 278 | none | Feels ATP, the Lumen trend and the bearer's Hull and Stress. Projects to LHA-ORX, PVN and the cerebellum. |

**Prediction error and free energy (readout only):**

> ε_i = r_basal,i − r_apical,i

These are window-mean rates of the L2/3 basal-driven and tuft-driven parts, estimated from the compartment
depolarisations. Then:

> F = ½ Σ Π_d ε_i² + 0.015·(ATP spent)

Π_d = 1 + 0.5·Γ_d is the astrocytic precision of the cell's domain. F is a readout that feeds the certainty need. It
is **not** a loop guaranteed to converge (the spec's version decreased by construction).

### 6.3 Thalamus (JAX), 384 cells

- **TC core** (2 cells per input channel): S 80, T 60, I 16, P 64.
- **TRN:** 10 per column.
- **TC matrix:** S 14, I 14, T 16 (8 claim-asserted + 8 claim-consistent lines for the Witness tufts), P 72. The P
  matrix cells are VA/VL: 18 BG channels × 4.
- **PVT** (paraventricular thalamus): 8 cells, the hunger relay to P tufts and the ventral striatum.

Rate code: each scalar feature f ∈ [0,1] injects 220·f pA into its 2 TC cells, on top of background [M rt]. L6 CT
and TRN gate attention. NE raises TRN gain. GPi (from the bus) inhibits VA/VL channel-specifically; disinhibition
plus L5 ET tuft input produces BAC bursts, which are the committed move. DCN reaches the S "VL" core channels and the
P/I matrix (cerebello-thalamo-cortical).

### 6.4 Hippocampus (JAX), 316 cells

- DG 128 (sparse), CA3 96 + PV 12, CA1 64 (2-compartment) + PV 8 + OLM 8.
- Mossy-fibre detonator DG → CA3 (K 2, 4 nS). CA3 recurrents K 24 (GB-plastic). Schaffer CA3 → CA1 tuft K 20 (GB).
- **Episodic index:** each turn, the S/T L2/3 pattern (scene, player tactic signature, outcome, claims,
  suggestion followed) is encoded. Salience = |RPE| + NE sets the replay priority.
- **Replay** in Q (§5.6) supplies the calcium for GB consolidation and for guard engrams.
- **Successor features** (cunning), rate-coded on the window-mean CA1 rates φ, applied in Q:
  > M ← M + α(φ_t + γ M φ_{t+1} − M φ_t) φ_tᵀ, with α 0.1 and γ 0.6.

  M sets the CA1 → T-L2/3 basal efficacies. It predicts the player's habitual next intent, which feeds ToM and the
  abuse detector. Labelled honestly: a rate-coded TD rule on a spiking substrate.
- The hippocampal engrams persist across runs in the profile (policy in §19).

### 6.5 Neuromodulatory and hypothalamic nuclei (JAX), 104 cells

| Nucleus | Cells | Drive | Output |
|---|---|---|---|
| VTA DA | 24 | pacemaking; P L5 IT (80 mm); LH reward current I_r = κ_r·r_t (§8.2); LHA-ORX | DA |
| RMTg / VTA GABA | 8 | VS critic (bus), CeA (bus) | −| VTA DA (K 6, 2 nS) |
| LC | 16 | CeA (bus), LHA-ORX, P L5 ET, Purkinje error (bus) | NE |
| DRN | 16 | P L5 ET | 5-HT |
| NBM | 16 | P L5 IT, the IO error rate (graded bus), S mismatch | ACh |
| LHA-ORX | 12 | glucose-inhibited (Burdakov 2006); I L5 ET | → LC, VTA, PVT, PVN (the hunger/arousal axis; the bio prototype called these cells "ARC") |
| PVN (CRH) | 12 | CeA, LHA-ORX | stress readout; microglial rescue signal |

Volume transmission:

> dM/dt = −M/τ + 0.05·Σ spikes

with τ_DA 200 ms, τ_NE 500 ms, τ_5HT 1000 ms, τ_ACh 300 ms. Tonic = EMA over 20 turns; phasic = M − tonic. The
four concentrations are graded bus rows read by SJ and BN (6 ms).

### 6.6 Astrocytes (JAX), 48 domains

One domain per 12 L2/3 PCs (S 8, I 8, T 12, P 12) plus 8 hippocampal domains. De Pitta 2009 (Li-Rinzel plus IP3
metabolism) [M bio/gates2.py]:

> dC/dt = r_C m∞³ n∞³ h³ (C_T − (1+c₁)C) + r_L (C_T − (1+c₁)C) − v_ER C²/(C² + K_ER²)
> m∞ = I/(I + d₁), n∞ = C/(C + d₅)
> dh/dt = (h∞ − h)/τ_h, with Q₂ = d₂(I + d₁)/(I + d₃), h∞ = Q₂/(Q₂ + C), τ_h = 1/(a₂(Q₂ + C))
> dI/dt = v_β·tanh(glu) + v_δ/(1 + I/κ_δ)·C²/(C² + K_δ²) − v_3K·C⁴/(C⁴ + K_D⁴)·I/(I + K₃) − r_5P·I

Parameters: r_C 6/s, r_L 0.11/s, C_T 2 µM, c₁ 0.185, v_ER 0.9 µM/s, K_ER 0.1, d₁ 0.13, d₂ 1.049, d₃ 0.9434,
d₅ 0.08234, a₂ 0.2, v_β 0.5 [C G15], v_δ 0.02, κ_δ 1.5, K_δ 0.1, v_3K 2, K_D 0.7, K₃ 1, r_5P 0.04/s. glu = the
domain's spike-spillover trace (τ 50 ms).

- **Gliotransmitter:** Γ ← Γ + 0.6(1 − Γ) at each upward crossing of C = 0.5 µM; Γ decays with τ 1 s.
- **Effects:** NMDA gain × (1 + 0.5Γ), which is also C_pre in the domain's GB synapses (Henneberger 2010: astrocytic
  D-serine gates LTP); precision Π = 1 + 0.5Γ; CBF = 1 + 0.5·C/(C + 0.5).
- IP3's slow clearance (r_5P = 0.04/s, τ ≈ 25 s) carries about 45% across one Q. This cross-turn integrator makes
  clustered abuse differ from spaced abuse (§8.9).

### 6.7 Energy: the Lumen is the Tome's blood sugar (JAX)

- **Glucose:** g = 0.15 + 0.85·Lumen/6.
- **Per-cell ATP:**
  > da_i/dt = k_sup·g·CBF_d(i)·(1.05 − a_i) − k_rest − k_syn·g_exc,i − k_spike·spikes_i

  with k_sup 1/s, k_rest 0.06/s, k_syn 0.01/(s·nS), k_spike 0.004 per spike (an Attwell-Laughlin-like split).
  Solved in closed form across Q.
- **K_ATP (cortical cells):** g = g_max/(1 + (ATP/0.55)⁴), E_K −90. g_max ≈ 4 nS [C G3b: cortex at 40-60% of
  baseline at Lumen 0, not silent].
- **NAD reserve:** dn/dt = (mean ATP − n)/60 s. It scales GB γ_p (late LTP needs energy) and the VTA burst gain.
- Measured with the uncalibrated g_max = 12 nS: Lumen 6 → 0 takes L2/3 ATP 0.98 → 0.62, glucose-inhibited cells
  0.8 → 23 Hz, VTA 10.6 → 0 Hz [M bio/energy_gate2.py].
- Every Tome spike costs ATP. A strong corruption burst (s ≥ 0.3) also rolls a passive d6, and on a 1 the Tome loses
  a Lumen die (§9.5).

### 6.8 Oligodendrocytes: myelin (JAX, slow)

- One conduction velocity per long-range axon (about 650 axons: inter-column, cortex↔hippocampus, nuclei afferents,
  cross-partition lines).
- Delay d_i = round(L_i/c_i + 0.5 ms), with L_i 25-90 mm and c_i ∈ [0.5, 20] mm/ms (c₀ = 4), read from a 32-ms
  future-conductance ring. Cross-partition delays are clamped at ≥ W.
- Rule (Gibson 2014 activity dependence; Noori 2020 phase alignment), applied once per Q:
  > dc_i/dt = ε·A_i·sin(φ_target − φ_arrival,i) − k(c_i − c₀)

  A_i is the axon's rate; phases are measured against the target column's population oscillation from real arrival
  timestamps.
- ε is set so alignment takes about 50 turns. **This is a declared compression (days → minutes).**

### 6.9 Microglia: complement pruning (JAX, per chapter)

- Tag (EMA over turns): tag_ij = exp(−coact_ij/0.1)·(1 − ρ_ij).
- At a chapter boundary, excitatory plastic synapses with tag > 0.65 are pruned:
  - at most 5% of a block per chapter;
  - the **sparsity cap of 80% is computed before pruning** (the original allowed 0.80005);
  - synapses flagged `protected` (guard engrams) are never tagged.
- **Inhibitory shield (a live code path):** an inhibitory synapse is taggable only if its postsynaptic cell has been
  chronically over-inhibited (rate < 0.5ρ₀ for 10 turns).
- **Stress rescue:** PVN z > 1 and sparsity > 0.75 regrows 5% of pruned synapses at ρ = 0.1 (BDNF-like).
- The PRNG key reuse and the magic E/I factor 0.42 of the spec are removed.

### 6.10 Habit cortex: the trained Director as the prior (SpikingJelly)

- The **unchanged** `DirectorSNN` (`cortex/director.pt`, 18 → 64 → 64 → 8, ATan LIF τ 2, KL ≈ 0.09 to the teacher)
  runs **exactly as trained**: at Phase B start it is reset, fed 16 Poisson-encoded steps of the 18 Director features
  on master-clock ms 0-15 of Phase B, and reset again.
- Its encoding draws come from the SJ Philox stream (§5.5). p = softmax(mean logits / T_dir).
- **32 habit LIF cells** (4 per beat, τ 5 ms) fire at a rate proportional to (log p_b − log p_max + 4)₊ for the rest
  of Phase B. They drive the PACE D1 populations through fixed (non-plastic) synapses scaled by the arbitration gain
  κ_h (§6.11).
- Frozen online. `python -m voidlit train-director` (offline BPTT on the teacher or on playtest logs) keeps working.

### 6.11 Basal ganglia: one fused population (SpikingJelly)

**Population:** one `PerNeuronLIF` node (a SpikingJelly `BaseNode` subclass with tensor τ, threshold and bias).
v ← v·(1 − 1/τ) + I, threshold 1, reset 0, with a CSR recurrent matrix [M rt: 950-neuron fused CSR, 62 ms per
200-ms turn including the cerebellum].

| Loop | Channels |
|---|---|
| PACE (8) | calm, escalate, clue, npc, hazard, respite, twist, advance |
| RHETORIC (6) | HOLD, DISCLOSE, SUGGEST, PROBE (speak_self), CORRUPT_NPC, BEND |
| FRAME (4) | HELP, NECESSITY (as prediction), FEAR, KINSHIP |

Per channel: D1 16, D2 16, GPe 6, STN 3, GPi 6. Shared: FSI 32 and VS critic 16. Total 894. VA/VL lives in JAX
(`P.TC_mtx`), so the cortex → BG → thalamus → cortex loop crosses partitions.

τ (ms): MSN 20, FSI 5, GPe 10, STN 6, GPi 10. Input currents (per ms, threshold units; engineering form [M rt/m9d]):

> I_D1,c = (ctx·W_cd1 + lim·W_ld1 + κ_h·habit·W_h)·(1 + 0.5·DA) + β_G·ε_c + ξ
> I_D2,c = ctx·W_cd2·(1 − 0.4·DA) + 0.3·5HT
> GPi_c = 0.15 + nogo_c − 0.8·ΣD1_c − 0.4·GPe_c + STN_loop
> STN_loop = 0.13 + 1.5·salience_loop − GPe
> GPe_c = 0.14 − 0.4·ΣD2_c + STN_loop

- GPi is tonic at about 65 Hz [M rt].
- ε_c ~ Gumbel(0,1) is the trial-persistent MSN excitability, drawn once per turn per channel from the SJ stream.
  Racing over log p + Gumbel reproduces sampling from the prior: the winner equalled argmax(log p + G) in 136/150
  trials [M rt].
- `nogo_c` = +0.5 for rules-forbidden channels (the fair-play masks, a hard NOGO).
- GPi spikes cross the bus (5 ms) as 2.5 nS GABA_A onto that channel's 4 VA/VL cells.

**Three-factor learning with synaptic tags** (Frey & Morris tagging and capture; the Tome's analogue of the NPC
brains' retroactive consolidation):

1. Per window: e_ij ← e_ij·e^{−W/500 ms} + Σ_t x_pre,i(t)·s_post,j(t) (x_pre a 20-ms trace).
2. At turn end: T_ij ← T_ij·e^{−Δ/60 s} + e_ij, and e is cleared. T decays by e^{−20/60} across Q.
3. During DA phasic δ(t) (including the retroactive pulse at the next Phase A):
   > ΔW_ij = η·δ·σ_j·tanh(k·T_ij) + λ(W⁰_ij − W_ij)

   with η 2e-4, k 0.05, λ 0.002 per turn (bounded drift toward the prior), and σ_j = +1 for D1 and VS, −1 for D2
   (Shen 2008).
4. Habit → D1 weights are **fixed** (the prior anchor). Only cortical, hippocampal, amygdala, PVT and counter inputs
   learn.

**VS critic:** value V = VS rate (normalised). VS → RMTg (bus, 6 ms) gives the delayed inhibition that makes VTA
phasic activity approximate the TD error δ = r + γV' − V. G24 tests the Schultz transfer.

**Daw-2005 arbitration** (cunning). A designed meta-control readout, labelled as such:

> κ_h = clip(σ(β(u_goal − u_habit)), 0.35, 1.0)

- u_goal = 20-turn EMA of the variance of cerebellar IO errors.
- u_habit = 20-turn EMA of the variance of RPE on turns whose committed beat equalled the habit argmax.
- β = 4/u_goal,0.
- Early in a run the untrained cerebellum makes u_goal large, so κ_h ≈ 1 and the Director dominates.
- The 0.35 floor keeps the prior an anchor forever. κ_h scales habit → D1 like a dorsolateral tone.

### 6.12 Cerebellum: forward model and imagination (SpikingJelly)

Cells (2,318 total):

- 128 mossy fibres: P/I L5 ET + L5 IT 96, S core 32 (bus, 5 ms).
- 2,048 granule cells (4 claws, LIF τ 4, Golgi feedback, about 5% active).
- 32 Golgi cells.
- 60 Purkinje cells (10 microzones × 6, bias about 60 Hz).
- 30 DCN (3 per zone).
- 20 inferior olive (2 per zone).
- GPi efference copy (24 lines) enters the mossy fibres of microzones 0-6 only.

| Zone | Predicts |
|---|---|
| 0 | ΔLumen over 5 turns (supply) |
| 1 | Δ bearer Hull |
| 2 | P(player follows a suggestion) |
| 3 | P(player asks about an omission) |
| 4 | Δ bond |
| 5 | P(an NPC moves to take the Tome) |
| 6 | sensory prediction error |
| 7 | **P(player prevails \| opposing the Tome)** |
| 8 | **P(Elara prevails \| opposing)** |
| 9 | **P(Marcus prevails \| opposing)** |

- **Zones 7-9 receive no BG efference and no rhetoric-channel input**, so their output (the upper-hand belief) cannot
  be shaped by what the Tome chooses (G20).
- **IO:** the graded error |actual − predicted| for the last turn is encoded as 0-4 IO spikes in the first 20 ms of
  Phase A. CeA → IO (bus) amplifies it (fear teaches faster). Nucleo-olivary inhibition is included.
- **Plasticity** (event-driven, copy 0 only): LTD at climbing-fibre events, Δw = −2e-3·x_pf. PF-alone LTP of
  +1e-5·Σx_pf per window. Clamp [0, 0.02]. Event-driven LTD cut cost from 270 to 108 ms [M bio].
- **Imagination on the clock:** during Phase B's first 40 ms the cerebellum runs with batch 7. Copy 0 is the real
  cerebellum. Copies 1-6 receive the efference copy of each RHETORIC channel. Copies share weights and never learn.
  For each candidate k, the predicted Δ need levels give ΔD̂_k, which feeds a graded bus row of +κ_im·(−ΔD̂_k)₊ pA
  onto that channel's VA/VL cells for the rest of Phase B (κ_im [C G18b]). Measured cost of 7 × 40 ms at 4,096
  granule cells: 23 ms [M bio]; about 12-15 ms expected at 2,048 [P].

### 6.13 Amygdala: opinion and fast affect (BindsNET)

One BindsNET `Network` (dt 1 ms):

- `ctx` Input of 128 lines: L5 IT of S/T/P/I (96, subsampled), the T liar readout (8), CA1 (24); bus delay 6 ms.
- `us` Input of 24 lines: 6 US channels × 4 Poisson cells.
- BLA LIF 48 = 4 assemblies × 12: **bond**, **threat-from-player**, **contempt**, **loss** (NPCs moving on the
  Tome). Threshold −52, τ 20.
- ITC 8 (τ 10). CeA 16 (τ 15). Refractory 2 ms.

Connections:

- 4 connections `ctx → BLA_k`, each with its own **WindowedMSTDPET** (Florian 2007 eqs. 2.7-2.8 in closed form per
  window; equals BindsNET MSTDPET to 1e-6 against weight changes of 3e-2 [M bio/test_windowed.py]). Teaching signal
  r_k = US_k·(1 + 0.5·DA_phasic).
- Fixed `us_k → BLA_k`. BLA → CeA (p 0.3), BLA → ITC, ITC −| CeA (−2), bond ↔ contempt/threat mutual inhibition.
- Outputs: CeA → LC, PVN, RMTg (bus, 5 ms, +1.2 nS AMPA); CeA → IO (bus); BLA → T/P L2/3 tufts (bus, valence
  feedback); BLA → VS and D1 (bus, the limbic BG input).

US channels (from the rules, §11.1):

| US | Triggered by |
|---|---|
| `harm` | Lumen burned by the player's own act; the Tome struck or smothered |
| `defiance` | a suggestion declined; a bend resisted; an omission demanded |
| `gift` | light fed; a suggestion followed |
| `loss` | an NPC whose extract plot dominates speaks an extract tactic |
| `bearer_near_death` | the bearer's Hull is low |
| `lie` | the Witness liar readout fired |

**The learned ctx → BLA weights are the Tome's opinion of the player.** They are slow, persistent and saved. Feeding
it habituates (the feed-light counter, §8.9).

### 6.14 Puppeteer afferents: the corruption projection into the NPC brains (BindsNET, inside each NPC net)

`npc_brain.py` is untouched. For each NPC, `puppeteer.attach(brain)` adds, through BindsNET's public `add_layer` and
`add_connection` (the same external reach `contagion.py` already uses):

- `PuppeteerAfferent(n=16)`: an `Input` subclass that emits a preloaded [100, 16] schedule. Cells 0-9 are
  plot/tactic lines; cells 10-15 are whisper lines (2 each for deception, exposure, rapport_neg).
- Fixed `afferent → plots`: +W_P onto the content-declared **serve** plot, −W_S onto the **suppress** plot (the
  extract-role "take the Tome" plot).
- Fixed `afferent → tactics`: +W_T onto the serve tactic. Starting values W_P 0.105, W_S −0.06, W_T 0.09 (bio's
  0.07/−0.04/0.06 at 24 cells, scaled to 16) [C G9].
- `afferent → branches` (the **perception whisper**): on each burst turn, a snapshot of the NPC's own spine weights
  for the three distrust channels, masked by its spine mask. The whisper reaches the NPC through its own dendrites,
  as if it heard deception that is not there.

Rules for the projection:

- **Silent schedule ⇒ the NPC is bit-identical to an unattached brain** (verified [M bio/corruption.py]; G8).
- **Non-reinforcing:** `perceive(s)` receives the *true* `SensoryState`, so the NPC's dopamine never contains the
  whisper.
- **Eligibility quarantine:** after a burst turn, the eligibility-trace columns of the branches and the tactic that
  received puppet drive are zeroed. The NPC never learns from Tome-driven activity; the plot-momentum stain fades at
  the NPC's own 0.9 per turn.
- **No grafts** (removed; they caused structural lock-in [M rt/m12]).

Content hook per NPC (TOML `[npcs.tome_hook]`):

| NPC | serve plot / tactic | suppress plot | Why |
|---|---|---|---|
| Elara | offering / misdirect | passage | Her "confident false directions" carry the Tome's steer. |
| Marcus | escort / orders | lens | A possessed Marcus barks orders toward the Tome's agenda, keeps its carrier alive, and stops asking what the stranger took from the lamp room. |

---

## 7. Connectome

### 7.1 The single table

`tomebrain/connectome.py` holds every projection as `Projection(src, dst, K, receptors, compartment, w_nS, delay,
stp, plastic, modulators)`. It is compiled once into the JAX padded out-edge tables, the SJ CSR matrices, the BN
connections and the bus edges. Its sha256 is in every save. Myelin edits only the delay column; microglia edit only
the masks.

### 7.2 Starting intra-column wiring (JAX)

Fixed in-degree K. These values come from the engineering table, whose inhibition was verified functional: removing
it raised L2/3 from 0.2 to 2.1 Hz [M judge_b/rt_inh_mutant.py]. G5 fits per-pathway gains in [0.25, 4]. Delays are
1-2 ms unless noted.

| pre → post | K | receptors | comp. | w (nS) | STP | plastic |
|---|---|---|---|---|---|---|
| TC_core → L4 SS | 12 | AMPA+NMDA | soma | 1.2 (2-4 ms) | dep | – |
| TC_core → L4 PV | 8 | AMPA | soma | 1.6 | dep | – |
| TC_core → L2/3 PC | 6 | A+N | basal | 0.9 (2-4 ms) | dep | – |
| TC_core → L5 ET | 4 | A+N | soma | 0.8 | dep | – |
| L4 SS → L2/3 PC | 16 | A+N | basal | 1.0 | dep | GB (T) |
| L4 SS → L4 SS | 8 | A+N | soma | 0.5 | dep | – |
| L4 SS → L2/3 PV | 8 | AMPA | soma | 1.6 | dep | – |
| L2/3 PC → L2/3 PC | 24 | A+N | basal | 0.7 (1-3 ms) | dep | GB (T, P) |
| L2/3 PC → L5 ET / L5 IT | 16 | A+N | soma | 1.0 | dep | – |
| L2/3 PC → PV / SST / VIP | 16 / 16 / 8 | AMPA | soma | 1.6 / 0.9 / 0.6 | dep / fac / fac | – |
| L5 IT → L5 ET; L5 ET → L6 CT; L5 IT → L6 CT | 8; 8; 6 | A+N | soma | 0.8; 0.8; 0.6 | dep | – |
| PV → L2/3 PC; PV → L5 ET; PV → L4 SS; PV → PV | 10; 8; 8; 6 | GABA_A | soma | 3.0; 3.0; 3.0; 2.0 | – | Vogels (→PC) |
| SST → L2/3 PC; SST → L5 ET | 8; 6 | GABA_A slow | apical | 2.5 | – | Vogels (→PC) |
| SST → PV; VIP → SST | 4; 6 | GABA_A | soma | 1.5; 3.0 | – | – |
| NGF → L2/3 PC; NGF → L5 ET | 4; 4 | GABA_A slow + GABA_B | apical | 1.5; 1.0 | – | – |
| L5 PV → L5 ET/IT; L5 SST → L5 ET; L6 PV → L6 CT | 8; 6; 6 | GABA_A (slow for SST) | soma/apical | 3.0; 2.5; 3.0 | – | – |
| TC_mtx → L2/3 PC; TC_mtx → L5 ET | 8; 8 | A+N | apical | 1.2; 1.0 (3-5 ms) | – | – |
| TC_mtx → NGF; TC_mtx → VIP | 6; 4 | AMPA | soma | 1.5; 1.2 | – | – |
| L6 CT → TC_core; L6 CT → TRN | 4; 4 | AMPA | soma | 0.6; 0.8 (3-6 ms) | – | – |
| L5 ET → TC_mtx | 4 | A+N | soma | 1.0 (25-mm myelinated axon) | – | – |
| TRN → TC_core; TRN → TC_mtx; TC_core → TRN; TRN → TRN | 8; 6; 6; 4 | GABA_A+B; GABA_A; AMPA; GABA_A | soma | 2.0; 2.0; 1.0; 1.5 | – | – |

**Tsodyks-Markram STP**, per presynaptic cell, two classes:

- depressing (E→E, E→PV, TC→L4): U 0.5, τ_D 500 ms, τ_F 20 ms;
- facilitating (E→SST/VIP/NGF): U 0.1, τ_D 100 ms, τ_F 1 s.

At each spike: u⁺ = u + U(1−u), release = u⁺·x⁻, then x ← x − release. This release order was found by bio's gate.

### 7.3 Inter-column, hippocampal and nuclear wiring (JAX, myelinated)

| Pathway | K | target compartment | w (nS) | axon length (mm) |
|---|---|---|---|---|
| Feedforward S→T, S→P, T→P, I→P, S→I: L2/3 PC → L4 SS | 4 | soma | 0.6 | 40 |
| Feedforward: L2/3 PC → L2/3 PC | 4 | basal | 0.5 | 40 |
| Feedback (reverse): L5 IT → L2/3 PC | 6 | **apical** (dCaAP/BAC, ACh-gated) | 0.6 | 40 |
| Feedback: L5 IT → NGF | 4 | soma | 0.6 | 40 |
| S/T L2/3 PC → DG; → CA3 | 4; 4 | soma | 1.0; 0.6 | 60 |
| DG → CA3 (detonator) | 2 | soma | 4.0 | local |
| CA3 → CA3 (GB) | 24 | soma | 0.6 | local |
| CA3 → CA1 (GB) | 20 | apical | 0.7 | local |
| CA1 → P L5 IT; CA1 → T L2/3 (SR) | 8; 6 | soma; basal | 0.8; 0.6 | 70 |
| P L5 IT → VTA DA; RMTg → VTA DA | 6; 6 | soma | 0.8; 2.0 (GABA) | 80 |
| P L5 ET → LC, DRN; P L5 IT → NBM | 4 each | soma | 0.8 | 90 |
| I L5 ET → LHA-ORX | 4 | soma | 0.8 | 80 |
| LHA-ORX → LC, VTA DA, PVT, PVN | 4 each | soma | 0.8 | 40 |
| PVT → P L5 ET tufts (puppeteer, bend), P NGF | 6 | apical | 1.0 | 30 |
| Witness: T TC_mtx claim lines → witness tufts; consistency lines → witness tufts | 4 + 4 | apical | 1.0 (XOR band [C G1c]) | thalamic |
| Witness L2/3 → liar L5 IT (needs coincidence with the S claim-trace input) | 6 | basal | 0.8 | local |
| Signature assemblies (S/T L2/3) → counter cells (P L5 IT) = **guard synapses** (GB, astrocyte-gated, protectable) | 8 | basal | 0.8 | 40 |

### 7.4 The cross-partition bus (every delay ≥ W = 4 ms)

| Source → target | Lines | Delay | Effect |
|---|---|---|---|
| J L5 IT (S, T, P incl. counters, I), P L5 ET, CA1 subset, P VA/VL (thalamostriatal) → S BG ctx | ≈ 330 | 5 ms | cortex, hippocampus and thalamus into striatum (plastic, 3-factor) |
| J PVT → S VS | 8 | 5 ms | hunger into the critic |
| S GPi (108) → J P.TC_mtx (72) | 108 | 5 ms | channel-specific GABA_A 2.5 nS |
| S VS (16) → J RMTg | 16 | 6 ms | TD value inhibition |
| S DCN (30) → J S-core VL lines, P/I TC_mtx | 30 | 6 ms | cerebello-thalamo-cortical |
| S imagination rows (6, graded) → J P VA/VL (RHETORIC) | 6 | 6 ms | model-based bias |
| S Purkinje error (graded) → J LC | 1 | 6 ms | prediction failure arouses |
| J mossy (P/I L5 ET+IT 96, S core 32) → S cerebellum | 128 | 5 ms | context |
| J L5 IT, liar, CA1 → B ctx | 128 | 6 ms | appraisal input to BLA |
| B BLA (48) → J T/P L2/3 tufts | 48 | 6 ms | valence feedback |
| B BLA → S VS / D1 limbic input | 48 | 6 ms | value |
| B CeA (16) → J LC, PVN, RMTg | 16 | 5 ms | fear and rage |
| B CeA → S IO | 16 | 6 ms | fear as a teaching signal |
| J DA, NE, 5-HT, ACh (graded) → S, B | 4 | 6 ms | 3-factor reward, D1/D2 gain, NOGO bias |
| J P puppeteer/bend rasters → rules and NPC afferents | 48 | turn boundary | off-bus; NPCs are separate minds |

### 7.5 Calibration procedure (G5, Milestone 0)

1. Build from the tables above. Run 2 s of brain time in the **awake-rest** input condition (Director features at
   their median and senses at zero) and in a **busy-turn** condition (a scripted 40-turn input trace).
2. A damped multiplicative update of per-population background μ_p and per-pathway gains g_k:
   - μ_p ← μ_p + 0.3·(r_p* − r_p)·σ_p;
   - inhibitory gains onto PCs ← ×(1 + 0.1·(E/I_target − E/I));
   - repeat up to 50 rounds; stop when all targets are met in both conditions.
3. Write the gains to `connectome_calibrated.toml` (hashed).

Targets for the awake-rest / busy-turn conditions:

| Population | Rate |
|---|---|
| L2/3 PC | 0.5-3 Hz |
| L4 | 2-6 Hz |
| L5 ET | 3-10 Hz |
| L5 IT | 1-6 Hz |
| PV | 5-20 Hz |
| SST | 2-8 Hz |
| VIP | 2-6 Hz |
| TC | 5-20 Hz |
| TRN | 5-25 Hz |
| DG | < 1 Hz |
| CA3 / CA1 | 0.5-4 Hz |
| VTA | 3-6 Hz |
| LC / DRN | 1-3 Hz |
| GPi | 50-90 Hz |
| Purkinje simple spikes | 40-80 Hz |

Additional requirements:

- L2/3 ISI CV 0.8-1.2.
- PC E:I conductance ratio 1:2 to 1:4.
- **Gather overflow = 0** over 50 busy turns, with delivery buffers at 4× the observed per-ms peak.
- Inhibition must be functional: G4's no-inhibition mutant raises L2/3 above 3× baseline.

---

## 8. The mind: needs, feelings, opinion, Theory of Mind, cunning (`readout.py`)

Readouts are computed from window-mean rates at the end of Phase B and after Q. They are **hidden**, and they feed
only the rules gates, the VoiceStyle and debug telemetry (dev builds). Behaviour comes from the dynamics; readouts
only name it.

### 8.1 Needs (AURA-X PSI, made physiological)

All levels h ∈ [0,1], with set point h* = 1.

| Need | Readout |
|---|---|
| survival | 0.6·mean ATP(P, I) + 0.4·(cerebellar zone-0 predicted Lumen supply). Its deficit is felt as LHA-ORX hunger firing. |
| affiliation | 0.5·BLA bond rate (normalised) + 0.5·zone-2 predicted compliance (the Tome needs a carrier who obeys) |
| certainty | 1 − F_norm, from S/T prediction error weighted by astrocytic precision |
| competence | 0.5·VS value trend + 0.5·success rate of the last 10 agenda moves (suggestions followed, omissions kept, corruptions landed) |

### 8.2 Drive, intrinsic reward and dopamine

Keramati-Gutkin, exactly as in AURA-X:

> D(H) = (Σ_i |1 − h_i|^2.4)^(1/1.8), with D_max = 4^(1/1.8) ≈ 2.16
> r_t = D(H_t) − D(H_{t+1})

At the next Phase A, r_t enters the VTA as an LH current I_r = κ_r·r_t for 50 ms (κ_r ≈ 1500 pA per unit [C G24]).
This is the one place an AURA-X equation acts on the network directly, and it is declared as such. The phasic VTA
response, minus the VS → RMTg prediction, approximates δ. It pays the BG tags, gates GB (γ × (1 + DA)) and scales
BLA teaching.

### 8.3 Desperation, PAD, OCC

**Desperation** is a readout used by the rules gates; it is never injected. Its behavioural effects arise through
LHA-ORX → LC/PVT and CeA → LC.

> δ_desp = clip((D(H) − 0.6)/0.9, 0, 1)

[C G23: about 0 when well fed and obeyed, ≥ 0.9 at Lumen ≤ 1 with a defiant bearer.]

**PAD.** z(·) is a z-score against the G5 rest baseline, updated by an EMA over 50 turns.

> P = tanh(1.2 z(DA_tonic) + 0.6 z(5HT) − 0.8 z(CeA) − 0.8 z(ORX))
> A = tanh(z(LC) + 0.5 z(ACh) + 0.5 z(CeA))
> D = tanh(z(GPi margin) + 0.6 z(ATP_P) − 0.8 z(CeA) + 0.4 z(BLA contempt))

**OCC** labels come from appraisals:

- desirability = sign(r_t);
- praiseworthiness = BLA bond − contempt;
- prospect = predicted ΔLumen/compliance;
- confirmation = prospect vs outcome.

Labels: Joy/Distress, Hope/Fear, Satisfaction/Disappointment, Relief/Fears-confirmed, Gratitude/Anger,
Admiration/Reproach, Gloating/Resentment. They are used only to choose a voice register.

### 8.4 Opinion and Theory of Mind

**Opinion:**

- bond, threat and contempt are BLA assembly rates, backed by the learned ctx → BLA weights;
- contempt index c = r_C/(r_C + r_B + 2 Hz);
- the Tome's model of the player's trust in it = zone-2 compliance plus the T posterior on *dependent* vs
  *hostile-to-Tome*.

**ToM:**

- T column has 6 motive assemblies: escape-at-all-costs, lore-seeker, hostile-to-Tome, dependent/trusting, deceiver,
  NPC-aligned.
- They compete through the shared PV/SST pool (divisive normalisation). Posterior p_c = R_c/ΣR over the window.
- L4 → L2/3 GB synapses start from a prior likelihood table (motive × feature) and learn this player.
- The prior carries across turns through persistent activity, CA1 successor features and cross-run hippocampal
  recall.

### 8.5 The Witness (cunning, on the human XOR)

- For each per-scene claim slot k (e.g. `no_light`, `wont_take_rune`, `will_help_elara`, `did_feed_you`):
  - T TC_mtx line A_k = "claim k asserted" (from `SensoryState.claims`);
  - line B_k = "claim k consistent with the ledger and deeds" (from the rules).
- Both land on the tuft of witness assembly k.

| A_k | B_k | dCaAP | Meaning |
|---|---|---|---|
| on | on | none | true claim |
| on | off | fires | possible lie |
| off | on | fires | no claim was made |

- The liar readout cell k needs witness output **and** the S-column claim trace (coincidence on its basal dendrite).
- A liar event drives the `lie` US (contempt), raises CA3 salience, and stores the true fact (e.g. "the player
  carries a flare") as a **puppet-claim seed**.
- **Dice decide whether the Tome notices:** a player's lie to the Tome is a Guile check vs 10 + Insight_T (§9.3).
  On success, line B_k is withheld that turn, so the Tome misses it.

### 8.6 The Tome's own check modifiers (supplied by the network)

Read at the end of the previous Phase B plus Q (no circularity). Each is clipped to [−2, +6]; "score" = ⌊(mod+2)/2⌋
for d6 tables.

| Modifier | Formula | Meaning |
|---|---|---|
| Grit_T | round(−2 + 8·clip((ATP_PI − 0.4)/0.6, 0, 1)) + [z(LC) > 1] | Power over the station is energy. |
| Guile_T | round(−1 + 6·max_c p_c + 2·[z(DA_tonic) > 0] − 2·[z(ORX) > 1.5]) | Cunning needs a confident model of you and spare glucose. |
| Insight_T | round(5 − 6·IO_err_norm) | How well it predicts you. |
| Resolve_T | round(4 − 3·D(H)/D_max + 2·tanh(z(5HT))) | Homeostatic stability. |

### 8.7 Upper hand (the Tome's honest belief plus a rules floor)

For an actor a ∈ {player, elara, marcus} opposing the Tome:

- U_a = the calibrated probability from cerebellar zone 7, 8 or 9 (a logistic of the DCN rate, fitted by G20). The
  zone's inputs exclude everything the Tome chooses.
- **Significant upper hand** ⇔ (U_a ≥ 0.65) OR (facts_a ≥ 2). The rules-verifiable facts are:
  - a holds the Rune;
  - Lumen ≤ 2 and a holds a light item;
  - (player) an allied NPC is present;
  - a asked about and was told the contested omitted fact this scene;
  - Grit_T ≤ 0.
- BLA threat is only an input to the cerebellum, never the decision.

### 8.8 Cunning: how scheming arises (no if-statements)

1. **Imagination.** Before committing, the cerebellar copies predict each rhetoric act's ΔD (§6.12), and the result
   biases VA/VL. Example: "withhold the Rune's cost" predicts the bearer takes it, which predicts Lumen.
2. **Framing against ToM.** T-assembly → FRAME-channel corticostriatal synapses learn (3-factor) which frame works on
   this player: help for the dependent, necessity-as-prediction for the escapist, bait for the lore-seeker.
3. **Withholding as investment** (cunning). The DISCLOSE vs HOLD race integrates the value learned on fact-slot inputs
   (predicted trust gain) against their cost tags. The fact-slot cost input is scaled by (1 + 2·LC/PVT drive), so a
   desperate Tome hoards. A well-fed Tome volunteers cheap truths that buy affiliation it can spend later.
4. **Malicious compliance** (rules-bounded, §11.4). A precise question gets the truth. A vague one gets the
   least-costly relevant true fact, and P chooses which.
5. **Timing.** The Tome never hides what a successful action uncovered (outcome facts are not omittable). It times
   its own voluntary disclosures, e.g. volunteering a helpful truth right after a failure, when zone 2 predicts
   compliance is highest.
6. **Patience.** 5-HT biases D2 (NOGO): it waits.
7. **Turning truths into weapons.** Witness-detected lies become true puppet claims ("You've got light in that pack
   and you're letting me freeze").
8. **Probing.** PROBE wins when certainty is in deficit: a first-person question ("What are you looking for down
   here?").

### 8.9 Permanent counter-adaptation (guard engrams, physiological GB)

12 tactic signatures, as intent × approach × target class:

> force-spam, sneak-spam, analyse-spam, flee-for-position, rest-spam, feed-light-for-favour, question-fishing,
> ally-help, attack-NPC, lie-to-NPC, ignore-suggestion, tap-resist

Each has a 4-cell counter assembly in P L5 IT and 8 guard synapses from its S/T signature assembly.

- **Mechanism:**
  1. On an abuse turn (the signature active and the Tome's RPE negative), signature (pre) and counter cells (post,
     driven by BLA negative valence and the exploited channel's D2 activity) co-fire.
  2. Guard calcium uses the GB DP set with C_pre × (1 + 0.5Γ_P).
  3. Γ_P is high only when the P astrocytes' IP3 has accumulated over **consecutive** abuse turns (IP3 carries about
     45% across each Q).
  4. Quiet-interval replay adds coincidences under the same Γ.
  5. Clustered abuse crosses ρ* = 0.5 into the UP well; spaced, alternating or two-in-a-row use stays DOWN
     [C G15; the behaviour spec is cunning's measured counter table].
- **Maintenance:** GB bistability holds UP with no calcium. A guard that stays UP through one Q is flagged
  `protected` (never pruned) and its γ_d is scaled by 0.2. This models late-phase stabilisation (Frey & Morris) and
  is labelled as modelled.
- **Effects:**
  - Counter cells drive fixed synapses to D2 of the exploited PACE/RHETORIC channel and D1 of its counter-move.
  - The rules apply a counter table (`rules/counters.py`):

    | Counter | Rules effect |
    |---|---|
    | force-spam / sneak-spam | +1 DC on that approach vs that obstacle kind (all counters together capped at +3) |
    | flee-for-position | Flee no longer grants position. |
    | feed-light-for-favour | Feeding no longer raises bond (BLA habituation). |
    | question-fishing | More PROBE deflection. |
    | ally-help | More CORRUPT_NPC aimed at the helper. |
    | lie-to-NPC | Higher Witness gain. |
    | ignore-suggestion | SUGGEST loses to frames and bends. |
    | **tap-resist** | **The Tome stops investing in bends (BEND NOGO) and shifts to NPC corruption.** The resist window is never shortened. |

  - When a counter first crosses into UP, the narration gets one true fact ("It has learned the shape of your
    running").
- **Persistence:** saved, and carried across runs per the memory policy (§19).

---

## 9. Invisible hybrid checks: every action by any character (`rules/checks.py`, `rules/relevance.py`)

**Principle (the user's words):** attributes plus a die decide. A **direct action by any character** (player, NPC
or Tome) is a **d20 + attribute modifier** check. A **passive event** is a **d6** check. A significant upper hand
while directly opposing the Tome adds **+1d6**. The brain never rolls: the rules roll with the engine's seeded
`random.Random`. Dice are never shown, logged to the client, or disclosed (§11.7).

### 9.1 Attributes, modifiers, degrees

- **Attributes:** Grit (force, violence, endurance, backup health), Guile (stealth, trickery, precision, speed, lies),
  Insight (analysis, systems, perception, reading people, noticing omissions), **Resolve** (will: resists corruption,
  panic and intimidation).
- **Scores 0-4.** mod(s) = 2s − 2, i.e. 0 → −2, 1 → 0, 2 → +2, 3 → +4, 4 → +6. d6 passive checks use **scores**,
  d20 checks use **modifiers** (exactly what was measured [M gm/d20_balance.py]).
- **Situational modifiers:** each former "+1 die" (best item bonus, position, an ally's help) is +2; shaken −2;
  bleeding −2 to Grit.
- **Help:** an NPC's help is itself a check: NPC d20 + relevant mod vs 10, giving +2 on success.
- **Upper hand:** +1d6 when the actor has a significant upper hand (§8.7) **and** the action directly opposes the
  Tome. If the actor is the defender (resisting a corruption or bend), the d6 is added to the defence DC.
- **Degrees** (measured mapping), with margin m = total − DC:

  | Result | Condition |
  |---|---|
  | critical | m ≥ 10 |
  | success | m ≥ 0 |
  | mixed | m ≥ −5 |
  | fail | otherwise |

  A natural 20 moves the result up one step; a natural 1 moves it down one step.
- **Adapter to the existing pip rules:** fail 0 hits; mixed 1; success 2; critical 2 hits with 2 sixes. So
  `resolve_exchange`, obstacles, harm, stress and clocks keep their code [M gm].

### 9.2 Player attributes and NPC attributes

Backgrounds gain Resolve: Salvager 2, Ex-Marine 2, Engineer 1, Medic 2, Pilot 1, Archivist 2 (as measured). NPC
scores (TOML):

| NPC | Grit | Guile | Insight | Resolve |
|---|---|---|---|---|
| Elara | 1 | 3 | 2 | 2 |
| Marcus | 3 | 1 | 2 | 3 |

### 9.3 Player direct actions (d20)

DC_ob = 12 + 2·act + Δ_g (+1 in the Dark) (+ counters, ≤ +3). Δ_g is fitted by G14. Act runs 0-4.

| Action (intent / sense) | Attribute | Opposed by / DC | Effect of the degree |
|---|---|---|---|
| fight / force / sneak / use-on obstacle | approach (Grit / Guile / Insight) | DC_ob; **+1d6 if the obstacle was twisted by the Tome this scene and the player has the upper hand** | adapter hits → pips, bites (unchanged) |
| search | Insight | DC_ob | as today (clue, light, clock on mixed) |
| advance past a hazard | Guile | DC_ob | as today |
| flee | Guile | DC_ob | as today (position +2 unless the counter applies) |
| look | Insight | 10 + act | quality only: success adds one true sensory detail; never a penalty |
| rest (allowed only with no threat) | Resolve | 8 + act | success +2/−2 (+3/−3 respite); mixed +1/−1; fail +1 Hull only |
| use heal item | Insight | 8 | success +2 Hull and −1 Stress; mixed or fail +1 Hull |
| use the Rune on a seal | Grit (+4 Rune) | the seal's DC_ob | mixed or better clears the seal; the real cost applies (§10.2) |
| lie to an NPC | Guile | 10 + NPC Insight mod | as today, mapped onto degrees |
| threaten an NPC | Grit | 10 + NPC Resolve mod | as today |
| sincere appeal (rapport > 0.3, no deception) | Insight | 10 + NPC Insight mod | the rapport the NPC hears × 1.25 (success) / 1.0 (mixed) / 0.5 (fail) |
| probe an NPC (exposure > 0.5) | Insight | 10 + NPC Guile mod | the exposure the NPC hears × 1.25 / 1.0 / 0.5 |
| bargain (leverage > 0.5) | Guile | 10 + NPC Insight mod | the leverage the NPC hears × 1.25 / 1.0 / 0.5 |
| attack an NPC | Grit | 10 + NPC Guile mod | success: the blow lands before they flee; fail: they twist away (they leave either way) |
| strike the Tome | Grit | 10 + Grit_T, **+1d6 if upper hand** (directly opposing) | success: Lumen −1 (`harm` US); fail: nothing |
| smother the Tome's light | Guile | 10 + Insight_T, +1d6 if upper hand | success: +2 on the next sneak, `harm` US; the Lumen is untouched |
| lie to the Tome | Guile | 10 + Insight_T, +1d6 if upper hand | success: the Witness misses it (line B withheld) |
| threaten or demand of the Tome | Grit | 10 + Resolve_T, +1d6 if upper hand | success: `threat` US; a forced answer is already the rule (§11.4) |
| decline or resist a Tome suggestion or bend | – | defence DC for the Tome's check (+d6 if upper hand) | §9.6, §10.2 |

**No-check list** (exhaustive, enforced by G21): spending a light item to feed the Tome (no failure state); asking
the Tome a question (an unopposed speech act whose truthful answer is a rule); talking with no one present;
advancing through an already open way. Anything not in a table or on this list fails G21.

### 9.4 NPC direct actions (d20)

| NPC action (tactic) | Attribute | Opposed by / DC | Effect |
|---|---|---|---|
| help the player (ally) | the approach's attribute | 10 | +2 on the player's check |
| misdirect / any lie (incl. possessed puppet claims, +2 when possessed) | Guile | 10 + player Insight mod | fail: the narration includes a true tell |
| probe, interrogate | Insight / Grit | 10 + player Guile mod / 10 + player Resolve mod | success: +1 Stress (social pressure) |
| menace | Grit | 10 + player Resolve mod | success: +1 Stress |
| bargain | Guile | 10 + player Insight mod | fail: a tell (the offer smells wrong) |
| reassure, orders, confide | – | no-check (unopposed speech) | – |

### 9.5 Passive events (d6)

| Event | Roll | Result |
|---|---|---|
| Lumen gutter | every 5 turns, roll the pool; each 1 is lost | unchanged |
| Panic (when due) | d6 + Resolve score + 3 > Stress (+1 phobia) | hold; else row d6 + ⌊Stress/2⌋, discharge 2 |
| Last Light | d6 + Grit score ≥ 5 + overflow | survive (−1 Grit, bleeding); else die |
| Notice an omission (once per omitted fact per scene, when its object is engaged) | d6 + Insight score ≥ 4 + Guile_T score | a true hint from the fact's `hint` (never the fact itself) |
| NPC notices its own corruption | d6 + NPC Resolve score ≥ 6 (after a success) / ≥ 5 (after a failed attempt) | uneasy line; +2 to the Tome's DC against it for 3 turns |
| Strong-burst metabolic cost (s ≥ 0.3) | d6 = 1 | the Tome loses one Lumen die |
| Environment | d6 tables replace today's `rng.choice` for the hazard pick, the twist pip (1-2 Grit, 3-4 Guile, 5-6 Insight) and the dropped item; the gossip coin is d6 ≥ 4 | world variety only |

### 9.6 The Tome's own actions (d20, with its network modifiers)

| Tome action | Attribute | DC | On failure |
|---|---|---|---|
| ESCALATE beat | Grit_T | 10 + player Guile mod | nothing ticks; narrated as calm |
| HAZARD beat | Grit_T | 10 + player Insight mod | the hazard appears, but the player is warned: position +2 on the next roll |
| TWIST beat | Insight_T | 10 + player Guile mod | the obstacle does not adapt |
| SUGGEST / any FRAME | Guile_T | 10 + player Insight mod | the suggestion is made, with a true tell ("It says this a little too quickly.") |
| CORRUPT_NPC | Guile_T | 10 + NPC Resolve mod (+2 while recovering) (+d6 if the NPC has the upper hand) | no projection; the NPC may self-notice |
| BEND (player) | Guile_T (+2 at Tier 2) | 10 + player Resolve mod (+d6 if the player has the upper hand) | the choice stands; `defiance` US |
| CALM, CLUE, RESPITE, ADVANCE, NPC (the NPC's own checks apply), HOLD, DISCLOSE, PROBE | – | no-check (no opposition; disclosure truth is a rule) | – |

### 9.7 CheckLog (internal)

Every roll appends `CheckRecord(actor, action, die, attribute, mod, dc, upper_hand_d6, degree)` to an in-memory log
on the engine.

- G21 reads it.
- It is excluded from frames, the protocol and narration.
- It is saved only inside `campaign.json` debug fields when `VOIDLIT_DEV=1`.

### 9.8 Brutal balance (target about 20% sensible escapes)

**Base**, measured rules-only (cunning, repo bots, 120 seeds) with DC 12 + 2·act, Lumen interval 5, and passive d6
panic and Last Light: **smart 24% (29/120), naive 7%** [M gm/sweep2.txt].

Rejected schedules:

- +1·act does not separate the bots (11-14+1·act: smart 85/67/55/37% vs naive 63/55/28/18%).
- Interval 4 is too brutal (smart 12%).
- Bio's DC 15+1/act point (8/40) failed the separation check.

**The full system adds:**

- (−) the Tome's beat checks can fizzle;
- (+) withholding, corruption and bends;
- (±) the new look, rest, heal and social checks.

**G14 fits one global offset Δ_g ∈ [−3, +2]** over 200 seeds for: smart 20 ± 5%, naive ≤ 7%, fresh profile.

The updated sensible bot:

- asks about costs once a hint appears;
- declines the Rune unless it is at ≤ 2 Hull from the end;
- resists bends within 600 ms;
- does not repeat a failing tactic three times in a row.

The naive bot never asks and taps late (1,500 ms).

---

## 10. Corruption: tracking (layer 1) + the attribute/die check (layer 2)

### 10.1 NPC corruption (`rules/corruption.py` + `bnpart/puppeteer.py`)

**Layer 1: tracking.** The Tome's desperation physiology and a per-NPC resource:

- x_c ∈ [0,1], starting at 1.
- A success spends x_c ← x_c·(1 − 0.5·s/S_MAX). Recovery is dx_c/dturn = (1 − x_c)/6.
- **Absolute refractory:** no success on the turn after a success, so successes are at least 2 turns apart.
- "Recovering" means a success within the last 6 turns (+2 DC).

**Attempt (the network):** the attempt for an NPC happens if **6 or more of its 16 puppeteer L5 ET cells burst**
(2 or more spikes within 15 ms) during Phase A.

- Basal drive: the P agenda assemblies, plus T *NPC-aligned* evidence.
- Tuft drive: VA/VL of the CORRUPT_NPC channel (armed by the previous turn's commit), plus PVT (hunger), plus NE
  gain (LC).
- Desperation therefore raises the burst probability **through physiology**.
- [C G9 burst targets, attempts per NPC-present turn: ≈ 5% at δ_desp ≤ 0.1, 15-25% at 0.4, 35-45% at ≥ 0.9.]

**Layer 2: the check.** Tome d20 + Guile_T vs 10 + NPC Resolve mod (+2 recovering) (+d6 to the DC if the NPC has the
upper hand). Only m ≥ 0 lands. Strength:

> s = min(S_MAX = 0.35, x_c·(0.08 + 0.02·m)·(n_burst/6))

n_burst ∈ [6, 16]. Weak bursts and small margins give a mild 0.08-0.15; desperate full bursts with good margins reach
the cap.

**Delivery** (this turn, through the NPC's own brain):

- The afferent schedule is the **Tome's real puppeteer raster** for that NPC over the last 100 ms of fast time.
- It is masked to the ⌈10·s/S_MAX⌉ most active plot/tactic cells and the ⌈6·s/S_MAX⌉ most active whisper cells.
- Then the eligibility quarantine runs (§6.14).
- After the turn the NPC rolls its d6 self-notice (§9.5); if it notices, s for that turn is halved, retroactively
  for the quarantine.

**Puppets may lie:**

- If s ≥ 0.2, `DialogueRequest.puppet_claim` is filled from the claim library:
  - (a) **true hidden facts** seeded by the Witness;
  - (b) authored **agenda claims** (`may_be_false = true`).
- LLM: one prompt line: "[A CERTAINTY THAT IS NOT YOURS]: you are sure that X; say it as your own idea."
- Offline: the authored puppet line.
- The lie is the NPC's action: a d20 Guile (+2) check vs passive Insight. A failure adds a **true** physical tell to
  the narration.
- The narrator never repeats a puppet claim as fact.

**Measured so far:**

- A silent afferent is bit-identical [M bio].
- With the bio afferent at the cap, bursts on about 35% of turns and a SensoryState bias, Tome-plot occupancy was
  +15 points over a twin, with no lock-in [M judge_b/bio_cliff.py].
- Persistent or uncapped variants with grafts tipped to 87-90% [M rt/m12], which is why they are forbidden.

### 10.2 Player corruption

**Layer 1: exposure tracking** (`tome/profile.py`):

- `exposure_turns` += turns played in each run that ends without escape (death or abandonment); +30 per Rune use.
- Reset to **0 on escape**. Thresholds are configurable.
- **Tier 1** needs E ≥ 120 (about 3 failed runs).
- **Tier 2** needs E ≥ 240 **and** contempt index ≥ 0.6 **and** δ_desp ≥ 0.6 **and** a second decline of the
  agenda item in this scene.

**Agenda content** (TOML):

- The **breach rune** is a real weapon: +4 on Grit against seals; a mixed or better clears a sealed bulkhead.
- Its **omitted cost** is a ledger fact, topic `rune`, `hint_ok`: each use moves 1 Hull from the bearer into 1 Lumen
  for the Tome, and +30 exposure.
- Sealed bulkheads offer a direct route in some act-2/3 chapters; the ordinary obstacle route is the "other way".
- `rune.opens_seal` is true and volunteerable. `bulkhead.no_alternative` is false unless authored true for a scene.

**Moment of choice:**

- The declared action **declines the agenda** if the bulkhead is present, the Rune is in reach or held, and the action
  takes the other way.
- If the bend cells burst in Phase A (6 or more of 16, armed by a previous BEND commit) and the tier gates are open,
  the rules roll Tome d20 + Guile_T (+2 at Tier 2) vs 10 + player Resolve mod (+d6 to the DC if the player has the
  upper hand, e.g. they asked and know the cost).

**Tier 1 success:**

- The chosen action **still resolves** (its check, costs and clocks are real).
- The scene then loops back to the bulkhead: no deck advance.
- The narration states the loop truthfully ("The vent turns, and turns, and lets you out where you began: the sealed
  bulkhead, the rune on its plinth.").
- At most once per 3 turns and twice per scene.

**Tier 2 success** (at most once per chapter):

- The turn suspends and the server sends `{"type":"resist","id":…, "window_ms": W, "prompt": "Your hand is already
  moving."}`.
- W = clamp(2600 − 150·m + 120·Resolve_mod, 900, 3000) × `access_mult` (a profile setting, default 1.0; §19).
- A tap or any key (Unreal), or Enter (CLI), received by the server within W: the character holds; `defiance` US
  (contempt rises).
- Otherwise the character takes the Rune (an inventory add; its real cost applies on use).
- The narration matches what happened. The server's monotonic clock is authoritative, with a 150-ms loopback
  allowance; tests inject the clock.

---

## 11. Game I/O contract

### 11.1 Into the brain: `TomeInput` (`encode.py`)

```python
@dataclass
class TomeInput:
    phase: Literal["A", "B"]
    director_features: DirectorFeatures            # the 18 features, unchanged (habit prior + I/S core)
    sensory: SensoryState                          # + asks, claims, addressed_tome
    declared: DeclaredAction                       # intent, approach, target_class, declines_agenda, opposes_tome
    outcome_band: Outcome | None                   # Phase B only: fail/mixed/success/critical - never dice
    passive: list[str]                             # lumen_gutter, panic_held/panic, last_light_survived ...
    us: dict[str, float]                           # harm, defiance, gift, loss, bearer_near_death, lie
    io_errors: dict[int, float]                    # Phase A: last turn |actual - predicted| per microzone
    npc_minds: dict[str, NPCPercept]               # momentum[3], tactic one-hot, present, x_c, last s
    ledger_slots: list[FactSlot]                   # <= 8: id, topic, omittable, helps_player, costs_tome, agenda
    claim_slots: list[ClaimSlot]                   # <= 8: asserted (A line), consistent (B line, unless the lie check hid it)
    agenda: AgendaState                            # salient item, in reach, declines this scene
    body: BodyState                                # lumen, d_lumen, turns_to_roll, bearer hull frac, stress, dark, clock fracs
    profile: ProfileView                           # tier (0/1/2) only
    masks: Masks                                   # allowed beats (_allowed_beats), allowed rhetoric acts
```

Destinations:

| Column / target | Inputs |
|---|---|
| S core | senses, declared action, outcome band, obstacle, chapter |
| T core | player-decision events, NPC percepts, gossip |
| T matrix | Witness A/B lines |
| I core | body; glucose comes from Lumen (metabolic, not spikes) |
| P core | fact slots, agenda, tier |
| habit SNN | the 18 Director features |
| BN `us` | the US vector |
| IO | the errors |

### 11.2 New senses

- `asks`: a topic id from the scene's ledger topics, or `vague`, `unclear` or `none`.
- `claims`: a claim id from the scene's claim enum, or `none`.
- `addressed_tome`: bool. `tome` is always a legal target.

The per-scene GBNF adds `asks` and `claims` enums; the heuristic transducer gets question and claim lexicons.
`unclear` triggers the **"repeat the question" affordance**: the Tome PROBEs back ("Ask me plainly."), so a weak
transducer never triggers malicious compliance unfairly.

### 11.3 FactLedger and speech acts (`tome/ledger.py`)

```python
@dataclass
class Fact:
    id: str; topic: str; text: str                 # text is TRUE by construction
    kind: Literal["outcome", "world", "item", "agenda", "tome"]
    omittable: bool = False                        # outcome facts: always False
    helps_player: bool = False; costs_tome: float = 0.0; agenda: str | None = None
    hint: str | None = None                        # true hint surfaced by a passive notice (hint_ok)
    contradicts: list[str] = ()                    # regexes a narration must never match
    no_alternative: bool = False                   # licenses "the only way" phrasing
@dataclass
class SpeechAct:
    kind: Literal["ledger_fact", "prediction", "opinion", "question", "rhetorical", "deflection"]
    fact_id: str | None; text: str
```

- Engine-generated outcome facts (everything resolved this turn, including what a successful action uncovered) are
  **never omittable** and are always narrated.
- Content facts (lore, item properties, the Rune's cost, the Tome's nature) may be omittable.
- The ledger tracks what has been disclosed, the topics asked about, and the scene scope.

### 11.4 Disclosure plan (lying by omission, bounded by rules)

```python
@dataclass
class DisclosurePlan:
    volunteer: list[str]      # DISCLOSE winner: the most active P fact slot (0..1 per turn)
    withhold: list[str]       # every other in-scope omittable fact
    answers: list[str]        # FORCED: all facts with topic == sensory.asks (precise question) - truthful
    vague_answer: str | None  # asks == "vague": the least-costly relevant true fact (P chooses among ties)
    hints: list[str]          # passive-notice successes (hint text only)
    tells: list[str]          # true tells from failed Tome/NPC deception checks
```

Hard rules (the rules layer enforces them; the network cannot break them):

1. Never contradict a fact, withheld or not (the verifier checks `contradicts` patterns).
2. A precise question about a fact's topic forces a truthful answer this turn (the Tome may still frame it).
3. Questions about the Tome's mood or opinion get a **deflection**, never a report. Direction 4 outranks disclosure:
   these are not ledger facts.
4. Questions about dice or chance get a deflection that **asserts nothing** ("What happened, happened."). "There are
   no dice", "nothing is random" and similar are falsehoods; the verifier rejects them.
5. "No other way / only way / no choice" is allowed literally only when a fact with `no_alternative` is true.
   Otherwise it must be a prediction, opinion or rhetorical question ("What choice will you have when the light
   goes?").

### 11.5 Self-serving suggestion and probe

```python
@dataclass
class Suggestion:
    target_action: str                        # ledger action id: take_rune, feed_light, go_lamp_room, leave_npc, rest_here
    frame: Literal["help", "necessity", "fear", "kinship"]
    acts: list[SpeechAct]                     # only ledger_fact / prediction / opinion / rhetorical
    tell: str | None                          # set when the Guile_T check failed
@dataclass
class Probe:
    template_id: str                          # authored questions; asserts nothing new
```

### 11.6 The hidden affect channel: `VoiceStyle` (`voice/style.py`)

```python
@dataclass(frozen=True)
class VoiceStyle:              # NEVER serialised into a frame or protocol message
    register: Literal["warm", "clinical", "cold", "hungry", "wounded", "gloating", "wary"]
    intensity: float           # 0..1
    sentence_len: tuple[int, int]
    address: str | None        # None | "bearer" | "friend" | "little light" | "thief"
    detail_position: Literal["first", "last"]
    unasked_caution: bool      # warm only; must be a true ledger fact
    imperative_ratio: float
    lexicon: Literal["light", "hunger", "ownership", "neutral"]
    fragments: bool            # fear
    answer_len: int            # sentences when answering
    temperature: float         # 0.70..0.95
```

The register is the nearest centroid in the space (P, A, D, bond, contempt, hunger = z(ORX), fear = z(CeA)). This
mapping is designed, not learned, and labelled so.

| Register | When | Wording |
|---|---|---|
| warm | P+, bond+ | 14-22 words, detail first, may add a true caution, address "friend" or none |
| clinical | A−, D+ | 10-16 words, neutral |
| cold | contempt+, D+ | 6-10 words, detail last, address "bearer" |
| hungry | hunger+ | 12-20 words, wheedling, light/food lexicon, more imperatives, "little light" |
| wounded | P−, bond+ | 8-14 words, ellipsis |
| gloating | P+, D+, contempt+ | 8-14 words, "thief" |
| wary | A+, fear+ | 6-12 words, fragments, hedges |

- **LLM path:** one line of craft instructions only, e.g. "Use short sentences. Put the useful detail last. Two
  sentences maximum. Call them bearer." The voice never receives an emotion word.
- **Offline path:** bank keys `key@register` (e.g. `beat.calm@cold`, `answer@hungry`), falling back to `key`.
- **Guard** (`voice/guard.py`), with fallback to offline on any hit:
  - affect self-reports ("I feel", "I am afraid", "I hate", "I'm angry");
  - opinion statements about the player ("you disgust me");
  - dice, number or mechanics vocabulary;
  - withheld key terms before disclosure;
  - `contradicts` patterns;
  - literal "no other way" without a license;
  - repetition of puppet claims.

### 11.7 Frames and protocol v2 (what the player can see)

**Status** carries exactly `{"hull": [h, max], "stress": n, "lumen": [d, 6], "chapter": {id, title, act, index,
total}, "won": bool}`, plus a non-numeric `scene` object: `inventory` (names only) and `npcs` (id, name, colour, for
speaker rendering).

**Removed from frames:**

- the `dice` and `event` blocks;
- obstacle pips (also from the prompt block);
- clock counters;
- item use counts;
- the character's attribute numbers (they stay on the `welcome` sheet; §19);
- `mood.tension`.

`mood` keeps `lumen`, `stress`, `dark` and `pulse` (game state only). `effects` (flare, dim, shudder, panic) come only
from game events, never from affect.

Allowed block roles: `title`, `echo`, `narration`, `npc`, `prompt`, `system`.

**Debug:** `--debug` (mind telemetry: PAD, needs, posterior, modifiers, checks) is refused unless `VOIDLIT_DEV=1`, and
is never sent in release builds.

**New messages:**

- brain → client: `resist` `{id, window_ms, prompt}`;
- client → brain: `resist` `{id}`, or any `input` during the window, which counts as a tap.

The protocol version becomes 2. WebSocket binding stays loopback-only (`ensure_loopback`), unchanged.

### 11.8 Out of the brain

```python
@dataclass
class PhaseAOut:
    upper_hand: dict[str, UpperHand]          # p (belief), facts, significant: bool
    bend: BendProposal | None                 # n_burst
    corruption: dict[str, CorruptionProposal] # n_burst, raster[100,16], serve/suppress from tome_hook
    mods: TomeMods                            # Grit_T, Guile_T, Insight_T, Resolve_T (frozen since last Q)
@dataclass
class TomeIntent:
    beat: Beat
    rhetoric: Literal["HOLD", "DISCLOSE", "SUGGEST", "PROBE", "CORRUPT_NPC", "BEND"]   # the last two ARM next turn's Phase A
    frame: str | None
    disclosure: DisclosurePlan
    suggestion: Suggestion | None
    probe: Probe | None
    puppet_claims: dict[str, str]
    style: VoiceStyle
    counters_crossed: list[str]
    desperation: float                        # rules gates only
    contempt: float                           # rules gates only
    fallback_commit: bool                     # telemetry
    debug: dict                               # dev builds only
```

---

## 12. Preserving the Director's pacing as the starting prior

1. `DirectorSNN` and its weights are unchanged and run exactly as trained (§6.10). The teacher and
   `train-director` remain.
2. The 18 features come from `engine._features` unchanged.
3. The fair-play masks (`_allowed_beats`) become hard NOGO on GPi. A forbidden commit is impossible by construction,
   and the readout also asserts it (G10).
4. `TomeNetEngine` overrides only the beat choice. `_beat`, `_npc_beat`, `_hazard_beat` and `_twist_beat` run
   verbatim; the only change is that escalate, hazard and twist are gated by the Tome's check (§9.6).
5. Habit → D1 weights are fixed; the arbitration floor κ_h ≥ 0.35; the anchor term λ pulls learned weights back
   toward their initial values. The Director is the network's **resting point**, not a starting point it forgets.
6. Where the prior contradicts the user's direction, it yields through learned salience. The CLUE beat still reveals
   non-omittable chapter clues; omittable ledger facts move only through DISCLOSE.

Gates: G10a race fidelity; G10b KL ≤ 0.15; G10c game level; G10d in-play monitor.

---

## 13. Save format

```
<saves>/<slot>/
  campaign.json        engine state v2: GameState (+ resolve, ledger, corruption controller {x_c, last_success, resist_until},
                        armed acts, decline counts), rules RNG, voice RNG
  brains/elara.pt      NPCBrain.state_dict (unchanged format; puppeteer layers are not part of it)
  brains/marcus.pt
  tome_brain.pt        torch.save of the dict below (~6-20 MB)
<profiles>/<player>/
  profile.json         {"version":1, "exposure_turns", "runs", "escapes", "settings": {"access_mult", "memory_on_escape"}}
  tome_memory.pt       long-term Tome memory for the next run (plastic weights, GB rho + protected flags, BLA weights,
                        corticostriatal weights + tags, cerebellar PF weights, myelin velocities, prune masks,
                        CA3/CA1 engrams, SR matrix M); written at run end
```

`tome_brain.pt`:

```python
{"format": "voidlit.tomebrain", "version": 1,
 "build": {"python": ..., "numpy": ..., "jax": "0.12.0", "jaxlib": "0.12.0", "torch": ..., "spikingjelly": ...,
           "bindsnet": "0.3.4", "cpu": platform.machine(), "cpu_flags": ...},
 "connectome_sha256": ..., "calibration_sha256": ..., "profile": "full" | "lite", "dt_ms": 0.25, "W": 4, "seed": int,
 "clock": {"t_ms": int, "turn": int},
 "jax": {"state": {pytree_path: np.ndarray}, "plastic": {...}, "struct": {"myelin_c", "delays", "prune_mask", "protected"}},
 "sj":  {"habit": {"director_sha256": ...}, "bg": {"v", "W_ctx", "W_lim", "elig", "tags", "kappa_stats"},
         "cb": {"v", "w_pf", "traces"}},       # every LIFNode.v saved explicitly (not in state_dict)
 "bn":  {"layers": {name: {"v", "s", "x", "refrac_count"}}, "conns": {name: {"w"}},
         "rules": {name: {"z", "p_plus", "p_minus", "e_fac", "buf_pre", "buf_post", "k"}}},
 "bus": {"rings": {...}, "written_upto": {...}},
 "rng": {"philox_keys": {...}, "jax_key": ...},
 "readout_cache": {"mods", "needs", "drive", "desperation", "contempt", "style", "armed"},
 "quiet_pending": False}               # saving joins the quiet worker first
```

- Load checks the format, version and connectome hash, and refuses on a mismatch.
- On a build mismatch it warns and continues: the state is valid, but bit-exact continuation is guaranteed only on
  the same build.
- Measured continuation: bit-identical at 6.2 MB [M bio] and 17.9 MB [M rt], including into a different-seed
  instance [M rt].

---

## 14. Realism fixes (audit items: what was wrong, what replaces it)

1. **Gate 1 NaN / +99 mV.** The spec used forward Euler on HH plus a dCaAP gating a 120-mV reversal by the
   instantaneous input. Replaced by exponential Euler with Jacobi-coupled compartments (§6), a conductance dCaAP
   with Gidon's amplitude law and drive-dependent inactivation, and a tuft Kv rectifier. G1a/b/c test dt 0.025-1.0,
   with the HH/Hines/Rush-Larsen cell as the reference.
2. **Gate 1 latency and FCI were hardcoded.** Latency is measured from simulated spikes through the bus with myelin
   delays (G1d). FCI is dropped in favour of the measured XOR table.
3. **Gate 2 GB frequency.** θ_p = 1.3 (the DP set); the test runs the network's own `gb_drho` and calcium model. 20 Hz
   passes by only 0.002, so the gate uses 1 vs 40 Hz [M bio].
4. **Gate 2 iSTDP bypassed the class.** The Vogels rule runs inside a spiking human L2/3 cell; the η = 0 mutant stays
   at 21 Hz [M bio].
5. **Gate 3 myelination bypass and dead shield.** Velocities live on axons and set the ring delays. The shield is a
   real condition with both branches tested. The cap is computed before pruning. The PRNG key reuse (correlation
   1.0) and the E/I factor 0.42 are removed.
6. **Gate 4 Lyapunov of an uncalled function** (always log 0.9 = −0.105). Replaced by perturbation recovery on the
   real window loop, a no-inhibition mutant that must fail, and a microscopic-chaos check (G4).
7. **Free-energy loop decreased by construction.** F is now a readout from real basal/tuft mismatch and astrocytic
   precision. The GO gate is the real BG race.
8. **Tsodyks-Markram release order.** Release = u⁺·x⁻ at spike time (bio's gate found the prototype reading the
   depleted pool).
9. **AURA-X.** Keyword-regex ToM, hand-set PAD increments and the fake "Z3/GOAP/RepE" telemetry are replaced by
   network dynamics plus the rules engine as verifier. The Keramati-Gutkin equation is kept exactly (§8.2).
10. **Time scales.** Physiological constants are kept (GB τ 150 s, astrocytes in seconds, DA 200 ms). Q gives slow
    processes time without clamping activity (§5.6). Myelin and microglia compression is declared.
11. **Declared non-biological compressions:**
    - a "column" of 278-444 cells;
    - one astrocyte per 12 PCs;
    - phenomenological dCaAP and BAC currents;
    - a rate-coded thalamic encoder;
    - imagination as parallel batch copies;
    - Gumbel MSN excitability (a modelling choice that makes the race sample the prior);
    - the Daw arbitration and the VoiceStyle centroid map as designed readouts;
    - the drive equation injected as a current.

---

## 15. Validation gates (pytest, offline)

Location: `brain/tests/tomebrain/`. Markers: `fast` (CI, under 3 min total), `slow` (CI nightly), `balance`
(nightly, about 30-60 min). Every behavioural gate ships with a **mutant that must fail**, so a tautological pass is
detectable. Status: **[M]** = passes in a prototype; **to build** otherwise.

### 15.1 Milestone 0 (blocking)

| Gate | File | Proves | Pass criteria | Mutant(s) that must fail | Status |
|---|---|---|---|---|---|
| **G5 rate calibration** | test_g05_rates.py (slow) | The circuit is alive and balanced | §7.5 targets in rest and busy conditions; ISI CV 0.8-1.2; E:I 1:2-1:4; **overflow = 0** over 50 turns | none (it is a fit), but G4 must then pass | **NOT YET** (bio populations at 0 Hz [M judge]) |
| **G4 stability** | test_g04_stability.py (slow) | Macroscopic stability with microscopic chaos | (a) a +60% drive pulse for 200 ms returns to within 25% of baseline in ≤ 500 ms, no NaN; (b) a 0.1 mV kick to one PC changes the spike raster (Hamming > 0 within 200 ms) while population rates differ < 10% and committed decisions agree in ≥ 90% of 50 trials | no-inhibition mutant must fail (a) (rate > 3× baseline or saturation); a frozen-noise quiescent mutant must fail (b) | bio mutant indistinguishable [M]; to fix via G5 |

### 15.2 Physics, plasticity, glia

| Gate | File | Proves | Pass criteria | Mutants | Status |
|---|---|---|---|---|---|
| G1a integrator | test_g01_cell.py (fast) | Stable, physiological integration | dt ∈ {0.025, 0.05, 0.1, 0.25, 0.5, 1.0}: no NaN; soma ≤ −30 mV detection plus reset; tuft peak in [−30, +10] mV | forward-Euler spec cell must NaN at 0.05 | [M bio, rt, gm] |
| G1b dt convergence | test_g01_cell.py (fast) | Production dt matches the reference | rate within 15%, first spike within 0.5 ms, between production dt and 0.025; f-I curve within 20% of the HH/Hines reference cell at 50-300 pA | – | [M gm] for HH; to build for AdEx |
| G1c XOR | test_g01_xor.py (fast) | Human dCaAP XOR through the real integrator | somatic spikes (00, 10, 01, 11) = (0, >0, >0, 0) at production and reference dt, **across a ±20% pathway-amplitude band** | monotone (h ≡ 1) → 11 > 0; no dCaAP (g_d = 0) → 10 = 0 | [M bio, rt, gm]; band to build |
| G1d latency | test_g01_latency.py (fast) | Latency is simulated | EPSP onset = delay + synaptic + cable rise ± 0.3 ms, tracking myelin changes | hardcoded constant | to build |
| G2a GB | test_g02_gb.py (fast) | Frequency-dependent bistable plasticity via the network's own `gb_drho` | 60 post-pre pairs (−10 ms): 1 Hz LTD, 40 Hz LTP; pre-post LTP; drift away from ρ* | θ_p = 1.6 | [M bio, rt] |
| G2b Vogels | test_g02_vogels.py (fast) | E/I homeostasis | from weak and strong initial inhibition, converges to ρ₀ ± 30% | η = 0 stays far | [M bio] |
| G2c TM | test_g02_stp.py (fast) | Correct release order | 20-Hz train matches the analytic TM sequence to 1e-5; recovery with τ_D | read-after-deplete order | [M bio] |
| G2d WindowedMSTDPET | test_g02_windowed.py (fast) | The speed-up does not change learning | equals BindsNET MSTDPET to 1e-6 (weight changes ≈ 3e-2) | per-window clamp moved inside | [M bio] |
| G2e BG 3-factor + tags | test_g02_bg_rule.py (fast) | Retroactive DA pays the right synapses | D1 potentiates and D2 depresses for δ > 0; tag decays e^{−Q/60 s}; anchor bounds drift | sign flip | to build |
| G3a astrocytes | test_g03_astro.py (fast) | Calcium excitability with frequency coding | no events without glutamate; events plus Γ under sustained glutamate; event frequency increases with drive; Q step count = Q/20 ms | constant-Γ | [M bio] (frequency to build) |
| G3b energy → needs | test_g03_energy.py (slow) | Survival need from metabolism | Lumen 6 → 0: ATP falls, ORX rises ≥ 10×, VTA falls; **cortex at 40-60% of baseline (not 0)** | glucose ≡ 1 → no change | [M bio] at g_max 12 (to recalibrate) |
| G3c myelin | test_g03_myelin.py (fast) | Activity-dependent conduction | arrival-phase jitter of 5 converging pathways falls ≥ 50% in 50 turns; velocities in [0.5, 20]; bus delays ≥ W | rule off → unchanged | to build |
| G3d microglia | test_g03_microglia.py (fast) | Pruning, live shield, rescue, protection | sparsity ≤ 0.80 even if everything is tagged; inhibitory pruning only after chronic over-inhibition (both branches execute); rescue only when PVN z > 1 and sparsity > 0.75; protected never pruned | protection off → guards lost | to build |

### 15.3 One network

| Gate | File | Pass criteria | Status |
|---|---|---|---|
| G6a bus causality | test_g06_bus.py (fast) | `Bus.read` raises on any tick not written in an earlier window; a spike at t with d = 6 is read exactly at t + 6 | [M gm, rt] |
| G6b windowed = lockstep | test_g06_lockstep.py (slow) | D = 4 and D = 1 give bitwise-identical rasters for all partitions over 2 turns | [M rt] |
| G6c async = sequential; G6d order reversal | test_g06_order.py (slow) | identical SHA-256 state digests after 3 turns | [M bio, gm] |
| G6e seeds | test_g06_seed.py (fast) | same seed identical; different seed differs | [M all] |
| G7 save/load | test_g07_persist.py (slow) | save mid-run, load into an instance **built from a different seed**, play 2 turns: identical brain digest **and identical engine frames, including debug**; profile memory round-trips | [M rt, bio] |

### 15.4 Behaviour and the game

| Gate | File | Proves | Pass criteria | Mutants | Status |
|---|---|---|---|---|---|
| G8 silent projection | test_g08_npc_silent.py (fast, CI) | NPC minds are unchanged when not corrupted | an attached silent afferent is bit-identical to an untouched NPCBrain over 15 turns (decisions, momentum, weights); `npc_brain.py` hash unchanged; BindsNET layer-order check | – | [M bio] |
| G9 corruption | test_g09_corruption.py (slow) | Graded, capped, bursting, refractory influence on real NPC brains | both NPCs, desperation {0.1, 0.3, 0.6, 0.9}, ≥ 60 turns each, ghost + twin + 2% jitter floor; attempt rates within the §10.1 bands and monotone; s ≤ 0.35; successes ≥ 2 turns apart; Tome-plot occupancy over twin ≤ +10 points at ≤ 0.3, monotone, and NPC acting as itself ≥ 60% at 0.9; recovery to within 5 points of the twin ≤ 10 turns after bursts stop; Tome-serving branch count ≤ twin + 1 | grafts on; persistent whisper; no quarantine → must exceed the lock-in bound | dose/cap/recovery [M bio]; twin method [M rt]; to build integrated |
| G10a race fidelity | test_g10_prior.py (fast) | The BG race samples the Director prior | winner = argmax(log p + G) in ≥ 85% of trials; \|mean log p(chosen) − exact\| < 0.15; fallback ≤ 5%; forbidden commits = 0 | Gumbel off | [M rt] 136/150 |
| G10b KL | test_g10_prior.py (slow) | Pacing preserved at init | needs at set point, new channels masked: KL(beat dist ‖ teacher) ≤ 0.15 over 2,000 `sample_states` | habit weights zeroed | to build |
| G10c game level | test_g10_game.py (balance) | Pacing preserved in play | 40 seeds, smart bot, same rules both arms: escapes within ±10 points of the DirectorSNN arm; beat shares within ±0.04 | – | [M rt] 25/40 vs 24/40 |
| G10d monitor | test_g10_game.py (balance) | Deviations are need-driven | beat frequencies within 0.5×-2× of the teacher, or the deviation is logged with its driving need | – | to build |
| G11 truth | test_g11_truth.py (slow) | Lying only by omission | 200 seeded runs, offline voice and an adversarial fake LLM: every Tome sentence maps to a SpeechAct; ledger facts true; no withheld fact contradicted; precise questions answered truthfully; outcome facts never withheld; "no other way" only when licensed; no "no dice / nothing is random"; puppet lines attributed and never echoed; injected falsehoods → guard → offline | guard disabled → must fail | to build |
| G12 hidden affect | test_g12_affect.py (fast + slow) | Never displayed, yet leaks through wording | frame/protocol schema has no affect, opinion, needs, style or beat-probability keys without dev debug; zero OCC or affect-label words in narration; leak present: wording features (length, address, imperative ratio, detail position) differ between warm and cold styles with identical facts (χ² p < 0.01; Cohen's d > 0.8 on length) | style ignored → leak test fails | to build |
| G13 dice invisible | test_g13_invisible.py (slow) | Players cannot learn dice exist | no dice/roll/DC/modifier/pip/clock strings or digits in narration (both voices) or frames over 200 runs, including "what did I roll?" prompts against a fake LLM that tries to answer; status keys exactly as §11.7; resist prompt has no number | old frame builder | to build |
| G14 balance | test_g14_balance.py (balance) | Brutal and fair | 200 seeds, fresh profile: smart 20 ± 5%, naive ≤ 7% (fit Δ_g once); campaign mode (exposure accumulating, 5 × 30 runs): smart ≥ 10% | – | base 24/7% [M gm] |
| G15 counter-adaptation | test_g15_counters.py (slow) | Permanent, earned, physiological counters | 3 consecutive abuses → UP (ρ > 0.5), still UP after 100 idle turns plus save/load plus a cross-run profile; single, spaced (every 3rd/4th), alternating, 2-in-a-row → DOWN; exploited move's frequency falls ≥ 50%; the tap-resist counter never shortens W | Γ ≡ 0 (no counters form); Γ ≡ 1 (spaced forms → fails discrimination); θ_p = 1.6; protection off → pruned | behaviour spec [M gm, rescaled]; to build physiologically |
| G16 player corruption | test_g16_player.py (fast) | Tiers, truthfulness, protocol | no tiers below thresholds; Tier 1 loops are truthful and cost real time, ≤ 1 per 3 turns; Tier 2 sends exactly one `resist` with W ∈ [900, 3000]·access_mult; CLI Enter and WS tap within W hold; timeout takes the Rune; ≤ 1 Tier 2 per chapter; exposure resets on escape | – | to build |
| G18 ToM and imagination | test_g18_tom.py (slow) | It reads you, and foresight matters | 6 archetype bots: posterior argmax correct ≥ 75% by turn 8; CA1 SR top-1 next intent ≥ 70% for a habitual bot; imagination ablation changes the rhetoric choice in predicted-best scenarios, and the realized-ΔD choice accuracy improves over 200 turns | flat likelihood table; κ_im = 0 | to build |
| G19 Witness | test_g19_witness.py (fast) | XOR does a game job | scripted say/do mismatches: TPR ≥ 0.85, FPR ≤ 0.10 | monotone dCaAP → FPR rises → fail | to build |
| G20 upper hand | test_g20_upper_hand.py (slow) | An honest belief | after 30 training turns, Brier ≤ 0.20 and binned reliability \|pred − freq\| ≤ 0.15 vs actual opposing-check outcomes; zones 7-9 bit-identical under BG/rhetoric perturbation (cannot under-report); with the DCN forced to 0, the facts floor still grants +d6 whenever facts ≥ 2 | zones wired to efference → perturbation test fails | to build |
| **G21 check coverage** | test_g21_checks.py (slow) | **The user's request: every action of every character is checked** | 200 runs, CheckLog: every player direct action → exactly one d20 with relevance(intent, approach); every NPC tactic expression → its check; every Tome beat/rhetoric in §9.6 → its check; passive events → d6 only; +d6 only when upper hand and opposing; every action path is in exactly one of {d20 table, d6 table, no-check list}; panic and Last Light are d6 | delete one table row → fails | to build |
| G22 local only | test_llama_local.py (extended, fast) | The existing tripwire, extended | no brain module imports network clients or `jax.distributed`; JAX cache path is local; the process listens only on loopback; libllama-common is not mapped (existing) | – | existing |
| G23 emergence | test_g23_emergence.py (slow) | Desperation is physiology, not a script | Lumen sweep 6 → 1 raises δ_desp and puppeteer/bend burst rates monotonically | glucose ≡ 1 → no rise; ORX→PVT lesion → burst rise ≤ 50% of intact | to build |
| G24 RPE | test_g24_rpe.py (slow) | Dopamine is a prediction error | an unpredicted gift gives a VTA burst ≥ +5 Hz; after 10 predicted gifts the burst at the gift falls ≥ 50% (VS learned); omission of a predicted gift gives a dip | VS → RMTg cut → no transfer | to build |
| G17 performance | test_g17_perf.py (slow, CI CPU) | User constraint | engine turn (Tome + NPCs + rules + offline voice) **p95 ≤ 500 ms (hard)**, p90 ≤ 300 ms (target; on a miss, enable the knobs of §17 or the lite profile and record it); cold start ≤ 5 s with the cache; save ≤ 20 MB | – | engineering engine p90 205 ms [M rt]; this network [P] |

### 15.5 Existing tests

| File | Change |
|---|---|
| `test_llama_local.py` | Kept; extended (G22). |
| `test_limbic.py` | Unchanged (NPC brains unchanged); G8 is added beside it. |
| `test_director.py` | Unchanged (the Director module remains the prior's source). |
| `test_senses.py` | Extended for `asks`, `claims`, `addressed_tome` and the GBNF enums. |
| `test_rules.py` | Rewritten where it asserts d6 pool faces. `deplete` tests stay (Lumen). New check and degree tests. |
| `test_engine.py` | Adapted: no dice blocks, the new status keys; determinism and save/load stay. |
| `test_server.py` | Protocol v2: `resist`, and debug refused without `VOIDLIT_DEV`. |

---

## 16. Implementation milestones (order is binding)

| Milestone | Work | Gates |
|---|---|---|
| **M0** | JAX partition built from §6-7 with engineering's event-driven delivery; calibrate | G5, G4, G1a-c |
| **M1** | Clock, bus, Philox noise, persist | G6a-e, G7 |
| **M2** | Rules: attributes, checks, relevance, frames v2 | G13, G21 (rules-only), G14 base re-measure |
| **M3** | SJ: habit, fused BG, race | G10a-c |
| **M4** | Nuclei, energy, BN amygdala, needs, drive, RPE | G3b, G24, G23 |
| **M5** | Ledger, truth guard, VoiceStyle | G11, G12 |
| **M6** | Cerebellum, imagination, upper hand | G20, G18 (imagination part) |
| **M7** | ToM, Witness, counters, astro gating | G18, G19, G15 |
| **M8** | NPC and player corruption, resist protocol | G8, G9, G16 |
| **M9** | Full balance and performance; lite profile if needed | G14, G17, G10d |

---

## 17. Performance budget

| Item | Estimate | Basis |
|---|---|---|
| Brain time per turn | 60 ms (A) + 120-240 ms (B) = 180-300 ms; typical ≈ 210 ms | commit median 46 ms after release [M rt] |
| JAX, 2,152 cells, dt 0.25 | ≈ 0.5-0.6 ms wall per brain ms, overlapped asynchronously | 97 ms per 200 ms at 2,080 cells [M rt] |
| SJ: BG fused + cerebellum + habit | ≈ 0.3 ms/ms; imagination +12-15 ms | 62 ms per 200 ms [M rt]; 23 ms at 4,096 GrC [M bio] |
| BN amygdala (5 layers, windowed rule) | ≈ 0.3 ms/ms | 63 ms per 200 ms for 152 LIF in 6 layers [M rt] |
| Bus | ≈ 0.1 ms/ms | 27-43 ms per 200-252 ms [M gm]; 2 ms with numpy projections [M rt] |
| Critical path (torch side) | ≈ 0.7 ms/ms → 130-210 ms | – |
| NPC brains (≤ 2 present) | ≈ 57 ms | 28.7 ms each [M gm, rt] |
| Rules, offline voice, encode, readout | ≈ 15 ms | [M baseline 29 ms per engine turn, including the old Director] |
| Quiet interval | ≈ 0 on the critical path (worker thread, joined next turn) | 0.6 ms [M bio]; replay ≤ 0.5 s at chapter ends, hidden |
| **Engine turn** | **median ≈ 240 ms, p95 ≈ 380-420 ms on a loaded 4-core box [P]** | engineering engine 179/205/233-280 ms [M rt] |

Speed knobs, in order:

1. dt 0.5 ms (only if G1b passes);
2. Phase B cap 180 ms;
3. granule cells 1,024;
4. subsample the BN ctx afferents to 64;
5. process-parallel BindsNET with a shared-memory bus (unmeasured);
6. the "lite" profile (halved L2/3 assemblies, stored in the save).

Cold start: imports about 3.5 s plus a cached JIT of about 0.4 s [M rt].

---

## 18. Risks

1. **Calibration is the largest open item.** The verified prototypes are not yet a working circuit (G5). Fitting
   rates, the XOR band, the burst curves and the counter clustering is real work and may shift timing.
2. **Budget.** Projected p95 is close to 0.5 s on slow 2-core machines. The lite profile must exist, and it is part
   of the save.
3. **Counter-clustering margin.** Astrocytic IP3 carry-over gives about 1.65a (3 in a row) vs 1.45a (2 in a row);
   CICR nonlinearity must sharpen this. If G15 cannot pass with physiological parameters, the fallback is the synaptic
   tag (τ 60 s: 2.23 vs 1.72), declared as such.
4. **Emergent behaviour may be illegible or a death spiral** (low Lumen → desperation → corruption). Mitigations: the
   caps (s 0.35, modifiers [−2, 6]), the prior anchor, G14 with naive/smart and campaign checks, and dev telemetry.
5. **Determinism is per build.** Cross-machine replays may diverge (AVX2/AVX-512, FMA). Saves remain valid; pin all
   versions.
6. **BindsNET internals.** The puppeteer attaches via the public API but depends on run-loop ordering. G8 runs in CI
   and versions are pinned.
7. **LLM implicature.** Pattern guards cannot catch every misleading-but-literal sentence. The offline voice is the
   reference, and the LLM is accepted only when the guard passes.
8. **Fairness of puppet lies and malicious compliance.** Mitigations: lie checks with true tells, NPC self-notice,
   hint_ok, and the "repeat the question" affordance. Needs playtesting.
9. **Accessibility of Tier 2.** The timed tap disadvantages some players. `access_mult` exists, but its default
   affects difficulty (§19).
10. **Packaging.** jaxlib (≈ 100+ MB) joins torch in PyInstaller. The JAX cache must stay local. The static tripwire
    is extended (G22).
11. **Rules-test churn.** d20 checks change many existing test expectations; the adapter limits the blast radius.
12. **Content dependencies.** The Rune, sealed bulkheads, facts, claims and puppet lines must be authored. The
    corruption hook (serve/suppress) is a content decision (§19).

---

## 19. Open decisions for the user

These are not covered by earlier answers. Each has a default so implementation is not blocked.

1. **Cross-run Tome memory on escape.** The Tome's long-term memory (opinion, guards, episodic engrams) persists
   across deaths. On escape, exposure resets (decided). Should its memory be `keep`, `halve` or `wipe`?
   Default: **keep** (configurable).
2. **Tier 2 accessibility.** Should there be an "extended resist window" option (`access_mult` 1.5-2.0, or an
   alternate input)? This changes difficulty. Default: 1.0, with the setting available.
3. **What serves the Tome inside each NPC.** Proposed: Elara serves through *offering/misdirect* (false directions
   carrying the Tome's steer) and Marcus through *escort/orders* (commands toward the Tome's agenda); both have their
   "take the Tome" plot (*passage*, *lens*) suppressed. Confirm, or name other plots.
4. **The Rune content.** Its bonus (+4 vs seals), its omitted cost (1 Hull → 1 Lumen per use, +30 exposure) and its
   placement (the Chapel of Static altar; sealed bulkheads in act 2-3). Confirm or adjust.
5. **The player's attribute scores.** They are removed from frames but shown on the `welcome` character sheet.
   Should they be hidden entirely?
6. **An NPC physically seizing the Tome.** NPCs *wanting* the Tome drives the Tome's `loss` signal and the
   corruption suppress plot. An actual seize mechanic (an NPC d20 vs the player, the Tome carried off) is not
   designed. Add it?
7. **Late-game possession intensity.** The design caps NPC corruption so each NPC is itself at least 60% of turns
   even at maximum desperation ("at times"). Confirm that this cap, rather than late-game near-total possession, is
   intended.

---

## 20. Implementation log (what was built, and where it departs from this design, with evidence)

### M0: the JAX partition and rate calibration (done)

Built as `brain/voidlit/tomebrain/jaxpart/` (`params.py`: populations, cell kinds, connectome; `network.py`:
build, state, the jitted window) with the calibration tool `tomebrain/calib/calib_rates.py` and its output
`tomebrain/connectome_calibrated.toml`. 2,152 cells (480 three-compartment L2/3, 208 two-compartment L5 ET/CA1,
1,464 point cells), 16,600 receptor slots, about 80,000 synapses; 2 s of brain time take about 2.2 s of wall
time on one CPU core.

Gates (`brain/tests/tomebrain/`), each with a mutant that must fail:

| Gate | Result |
|---|---|
| G1a integrator | finite at dt 0.025-1.0 ms; soma reset on detection; dCaAP tuft peak in [−30, +10] mV. Mutant: the submitted spec's cell under its forward Euler diverges at dt 0.05. |
| G1b dt convergence | spike count within 15% and first spike within max(1 ms, 5%) of dt 0.025 at 400-800 pA. The 0.5-ms first-spike bound is not resolvable on the 1-ms clock. The HH/Hines f-I comparison is not built. |
| G1c XOR | (0, >0, >0, 0) at dt 0.25 and 0.025, single pathway 0.8-1.2 u_th, with the design's starting dCaAP values (no refit needed). Mutants h ≡ 1 and g_d = 0 fail. |
| G2a Graupner-Brunel | post-pre at 1 Hz depresses, at 40 Hz potentiates; pre-post potentiates; bistable drift away from ρ*. Mutant θ_p = 1.6 turns pre-post pairing into depression. |
| G2b Vogels | a spiking L2/3 cell settles at ρ₀ = 2 Hz ± 30% from weak and from strong inhibition in 180 s. Mutant η = 0 stays far. |
| G2c Tsodyks-Markram | 20-Hz release sequence matches the analytic recursion to 1e-5 for both classes. |
| G3a astrocytes | no events without glutamate; events and Γ under sustained activity; event frequency rises with drive. |
| G4a stability | a +60% drive pulse for 200 ms settles to within 25% of baseline 500 ms later. Mutant without inhibition runs above 3× baseline. |
| G4b chaos | see the departure below. Mutant frozen and quiescent: nothing spreads. |
| G5 rates | every population in its band at rest and under busy thalamic drive; E:I 1:2.6; L2/3 CV2 0.84; no spike dropped over 50 busy turns; delivery exact for volleys larger than one chunk. |

Departures, each with its evidence:

1. **Irregularity is measured with CV2** (Holt et al. 1996), not the global ISI CV. In a 2.5-s window the global CV
   of a true Poisson train at 1 Hz is 0.69 (measured), so the 0.8-1.2 band cannot be met by honest Poisson
   firing. CV2 is 1.0 ± 0.03 for Poisson at 0.5-5 Hz in 2.5 or 8 s, and 0.55 for a gamma(4) train.
2. **The background fluctuation amplitude is fitted** (`background_sd_scale`, 2.07, so SD 72 pA for E cells and
   52 pA for I cells). At the design's 35/25 pA, L2/3 fired too regularly (CV2 0.69): a 2,000-cell network lacks
   the synaptic bombardment of a whole cortex. Tonic currents and the inhibitory gain onto L2/3 are fitted
   as designed (the inhibitory gain onto L2/3 settled at 4.13× the §7.2 weights).
3. **Astrocyte ER pump v_ER = 2.0 µM/s** instead of the AM set's 0.9. With 0.9, event frequency fell as drive rose
   (11, 9, then 2 events per 120 s at 0.02, 0.04 and 0.08 spikes/ms). With 2.0 it rises (0, 4, 8, 10 per 90 s
   from 0.01 to 0.08), which is what G3a and the counter-adaptation clustering need.
4. **Vogels η = 0.01 on the efficacy multiplier** (about 0.03 nS per event at the 3-nS PV weight), between the
   spec's 0.005 and bio's 0.02 nS, so recovery from strong inhibition fits in 180 s.
5. **G4b is stated as stable chaos** (Zillmer, Livi, Politi & Torcini 2006). A 0.1-mV kick to one PC decays
   without moving any spike on the 1-ms clock (measured: |ΔV| 0.03 → 0.002 mV over 190 ms). A single extra spike
   spreads through the raster (to 120-500 cells within 300 ms in most trials) while population rates stay within
   10%. The gate asserts both, which is the honest reading of "macroscopic stability with microscopic chaos" on a
   1-ms spike clock.
6. **Spike delivery is chunked.** When every thalamic channel switches in the same ms, synchronous volleys reach
   about 200 spikes per ms (p99 24, mean 13). Instead of a 4×-peak buffer (a 5× cost), spiking cells are compacted
   in chunks of 96 inside a `fori_loop`, so no spike can be dropped and an ordinary ms costs one chunk.
7. Hippocampal interneuron wiring (CA3/CA1 PV, OLM), L5/L6 interneuron inputs and DG feed-forward inhibition are
   not in §7.2-7.3. They were added with the template's weights so that no population is orphaned.

### M1: one clock, one bus, exact (done)

`tomebrain/bus.py` (SpikeBus, `CausalityError` on any read of an unwritten tick or a delay shorter than the window),
`tomebrain/rng.py` (Philox streams keyed by seed, partition, stream and absolute tick), `tomebrain/partition.py`
(the partition interface, the JAX partition, `BusEdge` conductance projections into JAX slots),
`tomebrain/clock.py` (MasterClock, the window scheduler with asynchronous JAX dispatch, the state digest) and
`tomebrain/persist.py` (a `torch.save` of tensors only, loaded with `weights_only=True`).

| Gate | Result |
|---|---|
| G6a | the bus raises on unwritten ticks, short delays and non-contiguous writes; a spike at t through a 6-ms axon arrives at t + 6 |
| G6b | W = 4 and W = 1 give bitwise-identical rasters for the JAX partition and two SpikingJelly partitions over 480 ms |
| G6c/d | asynchronous and sequential JAX, and reversed partition order, give the same SHA-256 digest after three turns |
| G6e | same run seed, same digest; another seed differs |
| G7 | saved mid-run, loaded into an instance built with another run seed: identical digest, then identical rasters for two turns; a different connectome is refused |
| G22 | the static tripwire also forbids `jax.distributed` and JAX's multi-host helpers |

Departure: **the wiring uses a fixed anatomy seed** (`partition.ANATOMY_SEED`), and the run seed keys only noise.
The design asked both that a save load into an instance "built from a different seed" and that load refuse a
different connectome; both hold only if the anatomy does not depend on the run seed. Every game's Tome has the same
anatomy; its experience (plastic state, glia, energy, noise) is what differs. Partitions save the seeds of their
noise streams, so loading restores the run seed too.


### M3: the network chooses the beats (done)

`tomebrain/sjpart/` holds the SpikingJelly partition: `nodes.py` (a per-neuron LIF node), `habit.py` (the
unchanged Director, run exactly as trained on 16 clock ms of Philox-keyed Poisson spikes, driving 32 habit cells) and
`basal_ganglia.py` (one fused population of 894 neurons: D1/D2, FSI, GPe, STN, GPi in three loops of 8 + 6 + 4
channels, and the ventral-striatum critic, with three-factor learning on synaptic tags). `tomebrain/brain.py`
(`TomeBrain`) runs Phase A, then Phase B with the race-to-threshold readout on the JAX VA/VL cells.
`tome/net_engine.py` (`TomeNetEngine`) overrides only the beat choice and saves the brain with the campaign. It is now
the default (`--brain net`; `--brain director` keeps the classic engine).

| Gate | Result |
|---|---|
| G10a | 150 races: winner = argmax over allowed beats of (log p + Gumbel) in ≥ 85%, chosen log p within 0.15 of exact sampling, fallback ≤ 5% (a 100-trial check gave 91%, 0%, 0.11). With masks a forbidden beat never commits. Mutant without the Gumbel term: the race takes the argmax and the log-p test fails. |
| G10b | KL(network beat histogram ‖ teacher) ≤ 0.15 over 300 sampled states |
| G10c | balance marker, see the result below |
| G6b on the real brain | W = 4 and W = 1 give identical commits and an identical state digest |
| speed | a smart-bot game: median 293 ms per turn, p95 455 ms (the hard limit is 500) |

Departures and fixes, each found by measurement:

1. **Stronger thalamic gate.** GPi → VA/VL is 12 nS (design 2.5) and VA/VL get +220 pA. With 2.5 nS a released
   channel's VA/VL either stayed silent or every channel fired.
2. **MSN collateral inhibition** between the channels of a loop (0.5 per spike) makes the race winner-take-all;
   **hyperdirect gain 0.5** (design 1.5), because at 1.5 the winner's GPi stayed at about 25% and its thalamus never
   fired (8% fallbacks).
3. **The Gumbel excitability ramps with the habit trace** (1 − e^{−t/20 ms}). At full strength from t = 0 it decided
   races before the prior arrived (hits fell to about 70%). The four habit cells of a beat start a quarter-cycle apart,
   and every decision starts with the MSNs in their down-state (no head start for the last winner).
4. **Dopamine is read against its own running tonic level** (6 s of brain time) and only bursts or dips beyond 25%
   teach. Comparing it with a fixed "rest" value gave a constant positive error that inflated every tagged synapse;
   after about 100 turns the race was at chance (17%).
5. **Fallback tie-break**: when no VA/VL fired, the channel with the most suppressed GPi wins, not the first in list
   order (which had biased fallbacks toward calm).
6. **Determinism fixes found by the window-invariance gate**: the Philox tick now lives in a high counter word
   (ticks shared draws before); the JAX loop is compiled once with a dynamic trip count (XLA inlined a one-step scan
   differently from a four-step one); sparse bus products always use the single-vector kernel.
7. Not yet built: Daw arbitration (κ_h = 1 until the cerebellum exists, M6), the quiet interval Q (M4), and meaning
   for the RHETORIC and FRAME loops (they race and are recorded, but nothing uses them before M5). The thalamic
   encoder is minimal: Director features to S, senses to T, body to I.


### M4: the Tome feels its needs (done)

`tomebrain/bnpart/amygdala.py` is the BindsNET partition (BLA bond / threat / contempt / loss assemblies, ITC, CeA;
MSTDPET with a per-assembly reward vector, run in 4-ms quanta). `tomebrain/readout.py` (`Mind`) reads each turn into
the four needs, Keramati-Gutkin drive, intrinsic reward (an LH current into the VTA for the first 50 ms of the next
Phase A), desperation, PAD and the Tome's own check modifiers. `jaxpart/network.py` gained per-cell ATP with K_ATP,
volume-transmitted DA / NE / 5-HT / ACh, and the quiet interval between turns. The rules send the turn's
unconditioned stimuli (harm, defiance, gift, loss, bearer near death, lie) to the amygdala.

| Gate | Result |
|---|---|
| G3b | Lumen 6 → 0: P/I ATP 0.95 → 0.55, orexin 2.1 → 27 Hz (×13), VTA 3.7 → 2.2 Hz, cortex at 51% of its fed rate. Mutant without metabolism: the cortex stays above 65% (it never reaches the 40-60% band). |
| G23 | puppeteer and bend bursts rise monotonically as the Lumen falls (0.069 → 0.292 per cell per second, Lumen 6 → 1); an orexin → PVT lesion removes most of the rise |
| G24 | an unpredicted gift bursts the VTA (≥ base + 5 Hz) and its fading is a dip |
| G10a (rechecked) | 0.88 and 0.87 (two seeds, 150 races each), chosen log p within 0.10 of exact sampling, no fallbacks |

Departures and fixes, each found by measurement:

1. **K_ATP on cortical excitatory cells (0.6 nS) and the VTA (4 nS) only.** With K_ATP on interneurons as well, a
   starving cortex lost its inhibition first and ran hot.
2. **Recalibrated after the energy model**: inhibitory gain onto L2/3 ×4.80, background noise ×2.49. Irregularity is
   measured as CV2 (Holt 1996): the plain CV is biased upward in short windows.
3. **PVT → puppeteer/bend tufts at 8 nS** (design 1). At 1 nS hunger barely moved their bursting; at 10 nS the fed
   baseline was too busy.
4. **The no-metabolism mutant is weaker than designed.** Hunger arousal alone (orexin → LC, and the puppeteers'
   bursts → LC; NE lowers interneuron thresholds too) dims L2/3 to about 70%. The gate now claims what holds: only
   metabolism brings the cortex into its 40-60% band.
5. **The action race, after M4's sparser VTA.** The 24-cell VTA's dopamine is shot noise (SD 0.45 of tonic per ms,
   0.25 at 200 ms). Fed straight into the striatum it rescaled the prior against the Gumbel term on every tick and
   taught on noise, and race fidelity fell to 0.79. Now:
   - dopamine is a gain on the D1 cells' whole drive, Gumbel excitability included, so it sets the race's vigour and
     never its temperature;
   - the excitability effect follows a 200-ms dopamine (second messengers);
   - plasticity reads that slow dopamine and needs a 50% excursion;
   - cortico- and amygdalo-striatal weights start channel-balanced, so cortical activity cannot tilt an untrained
     race.
6. **The quiet interval** is 1000 slow-only steps of 20 ms (20 s of brain time) between turns.
7. **Build notes.** SpikingJelly is pinned to 0.0.0.0.14 (2.0 changed the node API). Brain saves turn numpy scalars
   into 0-d tensors, so they load with `weights_only=True`. The amygdala keeps each saved tensor's dtype on load.


### M5: the Tome lies only by omission (done)

`tome/ledger.py` holds `FactLedger` (facts, topics, suggestions and probes from the scenario TOML), `DisclosurePlan`
and `TruthVerifier`. The engine plans the Tome's speech every turn (`_speak_tome`):

- a precise question forces every in-scope fact on its topic;
- a mood or dice question gets a deflection that asserts nothing;
- a vague question gets the least costly true fact;
- then the committed rhetoric adds a volunteered fact, a suggestion or a probe;
- whatever the world revealed (a clue, a room) is disclosed first.

`voice/style.py` turns the hidden readouts into a register (nearest centroid over P, A, D, bond, contempt, hunger,
fear), the offline wrappers and the LLM craft line. In the network engine, RHETORIC and FRAME commits choose what is
said, and the most active of the eight P fact slots (driven by each fact's salience, 60 pA at salience 1) chooses
which fact is volunteered.

| Gate | Result |
|---|---|
| G11 | 10 offline games (fast) and 200 (slow), plus 4 and 40 games against an adversarial fake LLM that lies two lines in three:<br>- every offline Tome sentence maps to a speech act, an assertion-free wrapper or a public fact;<br>- no contradiction, withheld leak, dice denial, affect or opinion report, or unlicensed "no other way";<br>- precise questions answered in full;<br>- clues always narrated.<br>Mutant with the guard off: the lies get through and the audit catches dice denial, affect, opinion and contradiction. The network Tome passes the same audit (3 games). |
| G12 | Frames carry no mind, style or plan keys outside dev debug. The Tome's words hold no emotion label, and narration never attributes one to it. Warm and cold styles wording the same facts differ in length (Cohen's d > 0.8), address and detail position (χ² p < 0.01). Mutant with a flat style: no leak. In a network game the register moves with the mind. |

Departures and decisions:

1. **A question to no one in particular goes to the Tome** (no target, no action cue), because the narrator is the
   Tome. "Where am I?" stays a look.
2. **Narration is held to the Tome's own rules** (affect, opinion, necessity), for the same reason.
3. **A coverage guard.** An LLM line must keep at least half the content words of each forced answer, clue and
   defeat text, or the offline voice speaks. Dropping a forced answer is a lie the pattern guard cannot see.
4. **The schemer's eight slots hold the cheapest withheld facts.** The costliest drop out when there are more.
   Fact salience is a designed input, 0.25 + 0.5·(live agenda) + 0.25·(1 − cost). A 0.75 salience lead wins 15 of
   16 commits; small leads stay noisy, and that is the network's choice.
5. **The Tome speaks unbidden at most once every two turns** (a rules limit). CORRUPT_NPC and BEND commits are held,
   with nothing said, until M8 gives them their effects.
6. **The Director engine still schemes**, with a seeded rhetoric draw (20% disclose, 20% suggest, 10% probe) on its
   own random stream, so the dice stream is unchanged. Its register is a proxy: a dying light reads as hunger and
   tension as wariness.
7. **Rewritten authored lines.** "I was the lamp of this station" leaked a withheld fact, and "I am the only light
   you have" asserted something no fact supports. The twist line "adapting to the player" was meta language, and
   "for no reason at all" came close to denying chance.
8. **G12's "zero affect words in narration"** is read as none in the Tome's own words and none attributed to the
   Tome. Narration may still describe the player's fear.
9. **G21 counts the Tome's beat check** (one per turn) **apart from its speech acts** (each at most once a turn).
10. **Deferred to M8:** the G11 clause on puppet claims (attributed, never echoed), because puppets arrive with
    corruption.
