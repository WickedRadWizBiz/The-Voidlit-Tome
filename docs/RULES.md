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
  (long) and the **Crucible** rises (Depth 8, Difficulty 2, Damage 2). It opens on Hostile and demands a
  different kind after every cast, shown before you choose. There is no running from it.
* **The Lumen**: 6 − Hunger dice (minimum 2), rolled after each chapter; 1s go dark. When it dies the Crucible rises
  at once and every Bite is +1.
* **Classes and Openings**: the six classes and their abilities, Kits and Bonds, the Opening roll (final in
  Core Rules; it simulated neutral), Origins and Debts.
* **Progression**: Insight, the Haven, raising stats, Talents, Pages (12), Hunger, feeding Relics, Fragments,
  Naming the Agenda, the Last Threshold and Legacy.
* **The Oracle**: ask the TOME a yes/no question (`? is the air breathable`).
* **Voice by Stress**: plain, uneasy, eldritch, and the TOME breaking through on a Panic.

## Rules that started in the app

The rules thread adopted these into the Core Rules on 2026-10-14 (pending Steven's confirmation there):

* **Allies**: one person met on a Witness Leaf who currently trusts you (their BindsNET brain favours alliance)
  lends **+1 die**. Attack or abandon them and they leave.
* **Gear helps kinds**: a scanner on Relics and Hazards, a battery or lockpicks on Relics, a toolbelt on Hazards and
  Relics, prayer cords on Witnesses: +1 die and a Use. Rifles and pistols are ranged; torches, tool-arms and blades
  are melee.
* **The Director** (a SpikingJelly network) chooses only how a draw is shown (a dot, shapes or the dark) and the
  narrative beat (calm, foreshadow, a clue, a person acting). It never changes a roll, a number or the deck.
* **Panic 7 ("Run")**: take 1 Harm and abandon the Leaf (it does not count). In the Crucible it is a lost action.
  The Vessel, in the Crucible or once the Lumen is out, loses its next action to a panic instead of a Lumen die.
* **Places**: a place in the Unknown with its own Leaves and people, authored in the Forge (`places`, `people`,
  and Leaves that name a `place`). *The Hollow Lantern* is the first. Its people appear on its Witness Leaves;
  its Leaves are dealt only there. A local TOML pack in `content/places/` can add a place the Forge lacks.

## What is not in the app yet

Group play, Vigil mode, recovering a Debt, and Insight for acting on a Bond or a Debt.

## Credits and inspirations

* **Voidlight** (Fari RPGs): d6 pools where 5–6 hit, a shrinking light pool as the trial's clock.
* **Mothership** (Tuesday Knight Games): Stress and Panic, the workaday crew, rules-light play.
* **Escape the Dark Sector** (Themeborne): a hidden chapter deck that ends in a boss.
* The FigCat list of rules-light and OSR sci-fi RPGs, for the off-screen turn that became NPC gossip.
