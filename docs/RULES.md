# The Voidlit Tome: rules

The app plays **TOME: Core Rules (Draft 0.2)**, the project's rules document
(<https://claude.ai/code/artifact/09d100ff-71c2-4696-8d3b-da93391a2dbe>). That document is the source of truth:
when this file and the Core Rules disagree, the Core Rules win, and the engine in `brain/voidlit/rules/` should be
fixed to match. Names, classes, Openings, tables and prose come from the TOME Forge's content files, shipped as
a snapshot in `brain/voidlit/content/tome/` (refresh it with `python tools/sync_content.py <folder or bundle>`).

This page only lists what the engine implements, what the app adds on top, and what it leaves out.

## What the engine implements

* **Casting**: a d6 pool of the stat, plus Talent and gear dice, capped at 6; every 5–6 is a Hit. A zero pool rolls
  one die that hits only on 6. Hits at or above the chapter's Difficulty is a success; some Hits is short; none is
  a failure and the chapter Bites. **Push** once: reroll the misses, and every 1 is +1 Stress.
* **The Bearer**: Might, Grace, Wits, Nerve; Grit 5; Stress 0–10 with a class minimum; Wounds (2d6 + open Wounds);
  three open Wounds or a Last Rites result offers a **Bargain** (a stat −1, Hunger +1, a Brand). With two Brands the
  TOME takes you.
* **Stress and Panic**: from Stress 5, each new point checks 2d6 ≤ Stress; at 10 it breaks straight into Panic.
* **The deck**: 54 hidden Leaves (36 chapters, 12 Echoes, 4 Breathers, 2 Interventions, a third at Hunger 4+).
  Chapters are Hostile, Hazard, Relic or Witness, at Depth 2/4/6 for shallow/deep/abyssal. Clear 3 (short) or 5
  (long) and the **Crucible** rises (Depth 10, Difficulty 2), demanding a different kind after every cast.
* **The Lumen**: 6 − Hunger dice (minimum 2), rolled after each chapter; 1s go dark. When it dies the Crucible rises
  at once and every Bite is +1.
* **Classes and Openings**: the six classes and their abilities, Kits and Bonds, the Opening roll (Steven's
  2026-10-14 rule: not yet through the playtest simulator), Origins and Debts.
* **Progression**: Insight, the Haven, raising stats, Talents, Pages (12), Hunger, feeding Relics, Fragments,
  Naming the Agenda, the Last Threshold and Legacy.
* **The Oracle**: ask the TOME a yes/no question (`? is the air breathable`).
* **Voice by Stress**: plain, uneasy, eldritch, and the TOME breaking through on a Panic.

## What the app adds

These are the app's own, offered to the project for the Core Rules and the Forge:

* **Place packs** (`content/places/*.toml`): a place in the Unknown with its own Leaves and people.
  *The Hollow Lantern* is the first. Its NPCs have BindsNET brains and appear on Witness Leaves.
* **Allies**: a person whose brain currently favours you lends **+1 die** to your casts.
* **Gear help**: Kit items add a die to the kinds they suit (a scanner on Relics, lockpicks on Hazards, and so on).
* **The Director** (a SpikingJelly network) chooses only how a draw is shown (a dot, shapes or the dark) and the
  narrative beat (calm, foreshadow, a clue, a person acting). It never changes a roll or the deck.

## What is not in the app yet

Group play, Vigil mode, recovering a Debt, and Insight for acting on a Bond or a Debt.

## One reading to settle

Panic 7 ("Run") during the Crucible: the app reads it as a lost action. The playtest simulator ends the trial as
survived. The difference is about 5 points of short-trial survival; the rules thread has been asked to settle it.

## Credits and inspirations

* **Voidlight** (Fari RPGs): d6 pools where 5–6 hit, a shrinking light pool as the trial's clock.
* **Mothership** (Tuesday Knight Games): Stress and Panic, the workaday crew, rules-light play.
* **Escape the Dark Sector** (Themeborne): a hidden chapter deck that ends in a boss.
* The FigCat list of rules-light and OSR sci-fi RPGs, for the off-screen turn that became NPC gossip.
