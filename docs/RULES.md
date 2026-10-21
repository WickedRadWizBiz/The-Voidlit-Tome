# The Voidlit Rules

Original rules for the Tome, written for this project. They borrow the *spirit* of several rules-light sci-fi
games; no text or tables are copied from them (see Credits).

> **The player never sees any of this.** Dice, checks, modifiers, pips and clocks are invisible: no frame, status
> bar, log or narration names them, even when asked (`tests/test_invisible.py`, gate G13). Outcomes reach the
> player only through the narration and the status bar (Hull, Stress, Lumen, chapter). This page is for developers.

## The checks

Every action by any character (the player, an NPC, the Tome) gets an attribute + die check relevant to it. The
full table is `brain/voidlit/rules/relevance.py`; an action missing from it cannot be rolled (gate G21).

* **Direct action: d20 + modifier vs DC.** The margin sets the degree:

  | Margin | Degree |
  |---|---|
  | ≥ 10 | **Critical** |
  | ≥ 0 | **Success** |
  | ≥ −5 | **Mixed** (it works, at a cost) |
  | otherwise | **Fail** |

  A natural 20 moves the result up one step and a natural 1 moves it down one.
* **Passive event: d6** (+ a score) against a target: the Lumen guttering, panic, Last Light, the world's tables.
* **Upper hand: +1d6.** When the actor has a significant upper hand on the Tome *and* directly opposes it, add a
  d6 to the total (a defender with the upper hand adds it to the DC instead). Until the Tome's network is built,
  "significant" means two or more rules-verifiable facts: Lumen at 2 or less while holding a light; an ally
  present; the Tome's own Grit at 0 or less.

**Situational modifiers**: the best kit bonus gives +2 per step; a better position +2; an ally's help +2 (the ally
rolls d20 + its modifier vs 10 to give it); shaken −2; bleeding −2 to Grit.

## Your character

Four attributes, each scored 0–4. The d20 modifier is `2 × score − 2` (0 → −2, 2 → +2, 4 → +6):

* **Grit**: force, violence, endurance. Also your backup health.
* **Guile**: stealth, trickery, precision, speed, lies.
* **Insight**: analysis, systems, perception, reading people, noticing what goes unsaid.
* **Resolve**: will. It holds against panic, intimidation and corruption. It is never an approach.

**Hull** is your health (5–7). **Stress** runs from 0 to 10. Kit gives bonuses to some approaches or intents, or
does something special: *light* items feed the Lumen, *heal* items restore Hull. The welcome sheet shows the four
scores; frames never do.

Six backgrounds: Salvager, Ex-Marine, Engineer, Medic, Pilot, Archivist.

NPCs have the same four attributes (Elara 1/3/2/2, Marcus 3/1/2/3), and their tactics are checked too:
misdirection and offers (Guile vs 10 + your Insight mod; a failure leaves a true tell in the narration), probing,
interrogation and menace (success costs you 1 Stress). Reassurance, orders and confiding are unopposed.

## Obstacles

Every enemy, hazard, lock and horror is a hidden row of **pips** (Grit, Guile, Insight). Say what you do; the Tome
reads your approach and rolls d20 + modifier against the obstacle DC, **12 + 2 per act** (+1 in the dark, plus the
scenario's `dc_offset`). The degree maps onto hits: fail 0, mixed 1, success 2, critical 2 plus an extra pip.

* Each hit from a matching approach strips one matching pip.
* Hits with no matching pip are not wasted: **every two strip one pip of any kind** (brute force works, slowly).
* Strip every pip and the obstacle is overcome (and you shed 1 Stress).
* On a **fail**, an uncleared obstacle bites for its full harm and stress. Locks cost time instead: a clock ticks.
* A good **look** (Insight vs 10 + act) tells you, in words, what an obstacle would give to. Looking never hurts.

## Harm, Last Light and Stress

Harm comes off Hull. Harm past zero Hull forces a **Last Light** save, a passive d6 + Grit against 5 + the
overflow. Fail and you die; succeed and you stay up, lose 1 Grit and start bleeding.

Gaining Stress at 4 or more triggers a **Panic check**, a passive d6 + Resolve + 3 that must beat your Stress (+1
with a phobia). Otherwise roll on the Panic table (d6 + Stress/2). Panicking discharges 2 Stress first. Results run
from an adrenal spike through freezing, fumbling, gaining a phobia and smashing a lamp, to the Dark answering you.

## The Lumen

The Tome's own light is a pool of up to six dice. Every 5 turns, roll it; every 1 gutters out. Twists, panics and
some clocks burn it too; light sources restore it. At zero **the Dark has the sector**: every bite does +1, and
the Tome pushes escalation. The renderer dims the voxels with it, so you watch your light die.

## The Tome acts too

The Tome's moves are checks with its own modifiers (Grit, Guile, Insight, Resolve, each −2 to +6). Until its network
supplies them, its Grit follows its light (`−2 + 8 × Lumen fraction`) and the rest sit at Guile +2, Insight +2,
Resolve +3.

| Move | Check | On a fail |
|---|---|---|
| Escalate | Grit vs 10 + your Guile mod | nothing ticks; a quiet moment instead |
| Hazard | Grit vs 10 + your Insight mod | the hazard comes, but you saw it coming (+2 on your next roll) |
| Twist | Insight vs 10 + your Guile mod | nothing adapts |

You can turn on the Tome as well: strike it (Grit vs 10 + its Grit; a hit costs it one light), smother its glow
(Guile vs 10 + its Insight; +2 on your next Guile roll), lie to it, or threaten it. Each opposes the Tome directly,
so the upper-hand d6 applies. Asking it a question is never rolled; it answers in the first person.

## Clocks, rest and allies

* **Clocks** (*The Choir Wakes* 8, *Reactor Drift* 6) tick when the Tome escalates and when locks fail loudly. A
  full clock fires its consequence.
* **Rest** once per chapter when nothing is threatening you, a Resolve check vs 8 + act: success +2 Hull and
  −2 Stress (+3/−3 in a respite room), mixed +1/−1, fail +1 Hull only. Crossing into a new act gives +1 Hull and
  −1 Stress.
* **Allies**: an NPC whose brain currently favours alliance tries to help (+2 when its own check succeeds). Their
  minds change with how you treat them, and with what other NPCs say about you behind your back.
* **Talk is checked when it bites**: lies (Guile vs 10 + their Insight), threats (Grit vs 10 + their Resolve),
  offers, probing questions and sincere appeals are rolled, and the result changes what their brain *hears*.

## The sector deck

A run is a start chapter, two chapters from each of acts I–III (drawn from four each), and an escape chapter.
Get through the escape alive to win.

## Credits and inspirations

* **Voidlight** (Fari RPGs, Boundless system, ORC License): d6 pools where 5–6 hit, a shrinking light pool as the
  run's clock, attributes as backup health.
* **Mothership** (Tuesday Knight Games): Stress and Panic, the horror of the workaday crew, rules-light play.
* **Escape the Dark Sector** (Themeborne): a chapter deck escalating by act toward a final escape, enemies as rows
  of symbols your approaches must match.
* **Blades in the Dark / Scum and Villainy**: progress clocks; position via the Flee and help bonuses.
* The FigCat list of rules-light and OSR sci-fi RPGs (Traveller, Stars Without Number, Death in Space and others),
  for the off-screen faction turn that became NPC gossip.
