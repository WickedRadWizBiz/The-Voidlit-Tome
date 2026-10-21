# Tome protocol v2 (brain ⇄ body)

JSON text messages over a WebSocket. The brain listens on `ws://127.0.0.1:8765` (`python -m voidlit serve`), and
only ever on a loopback address: the Tome is a local game and is never reachable from another machine.

## Client → brain

```json
{"type": "hello", "protocol": 2, "options": {"seed": 7, "background": "medic", "name": "Ash"}}
{"type": "input", "text": "I pry the iris open with the cutter"}
{"type": "command", "name": "save", "args": {"slot": "quicksave"}}
{"type": "command", "name": "load", "args": {"slot": "quicksave"}}
{"type": "command", "name": "new",  "args": {"seed": 8}}
{"type": "ping"}
```

## Brain → client

| type | when | payload |
|---|---|---|
| `welcome` | after `hello` or `new` | `protocol` (2), `scenario`, `backgrounds`, `character` (the sheet: `name`, `background`, `grit`, `guile`, `insight`, `resolve`, `hull_max`, `inventory` names), `llm` (bool: a local llama.cpp model is loaded) |
| `thinking` | as soon as `input` arrives | — (ripple the surface while the mind works) |
| `frame` | the result of a turn, or the opening | see below |
| `saved` / `loaded` | after commands | `slot` |
| `pong` | after `ping` | — |
| `error` | malformed or out-of-order messages | `message` |

## Frame

```json
{
  "type": "frame", "protocol": 2, "seq": 12, "over": false,
  "blocks": [
    {"role": "echo",      "spans": [{"text": "> I pry the iris open", "color": "#7f8796", "glow": 0.0, "effect": "none"}]},
    {"role": "narration", "spans": [{"text": "The iris grinds open...", "color": "#c9d1e0", "glow": 0.0, "effect": "rise"}]},
    {"role": "npc",       "speaker": "Elara Voss", "data": {"npc": "elara"}, "spans": [...]},
    {"role": "npc",       "speaker": "The Tome", "data": {"npc": "tome"}, "spans": [...]},
    {"role": "narration", "spans": [{"text": "The Tome's light gutters and shrinks.", "glow": 0.5, ...}]},
    {"role": "prompt",    "spans": [...]}
  ],
  "status": {
    "hull": [4, 5], "stress": 3, "lumen": [5, 6],
    "chapter": {"id": "ration_vault", "title": "The Ration Vault", "act": 1, "index": 1, "total": 8},
    "won": false,
    "scene": {"inventory": ["Medkit", "Scalpel"], "npcs": [{"id": "elara", "name": "Elara Voss", "color": "#6fe3cf"}]}
  },
  "mood": {"lumen": 0.833, "stress": 0.3, "dark": false, "pulse": "breath"},
  "effects": [{"kind": "shudder", "strength": 0.6}]
}
```

* **Dice are invisible.** No frame carries dice, rolls, checks, DCs, modifiers, obstacle pips, clock counters,
  item use counts or attribute scores, and no text block contains a digit or names a mechanic (gate G13,
  `brain/tests/test_invisible.py`). Outcomes reach the player only through narration and `status`.
* **No mind on display.** Nothing in a frame reveals the Tome's mood, opinion, needs or plans; they leak only through
  its wording. `mood` is game state only (the Lumen and the player's Stress), and `effects` come only from game
  events.
* **roles**: `title`, `echo`, `narration`, `npc`, `prompt`, `system`. Render top to bottom. An `npc` block whose
  `data.npc` is `"tome"` is the Tome speaking in its own voice. Extra `narration` blocks after the NPC lines are
  things the Tome tells outright (light found, panic, a tell in someone's voice).
* **span.color** is sRGB hex. **glow** (0–1) drives emissive intensity. **effect** is a hint:
  `rise` (default: voxels lift in reading order), `pulse`, `flicker`, `ripple`, `none`.
* **mood.lumen** scales the brightness of the whole surface. `dark` means only red and dread colours remain
  readable. `pulse` sets the surface's idle breathing rhythm.
* **effects**: `shudder` (camera and surface jolt), `dim`, `flare`, `panic` (glyphs scramble briefly), `ripple`.
* **Debug** is a developer feature. `--debug` is refused unless the environment sets `VOIDLIT_DEV=1`; then each
  frame gains a `debug` object (the SensoryState, every NPC brain's decision, the beat and its probabilities, every
  check this turn, the Tome's modifiers, internal notes and gossip). Release builds never send it.

## Changes from v1

* Removed the `dice` and `event` block roles; outcomes are narrated.
* `status` lost `character`, `obstacles`, `clocks` and item counts; it gained `scene` (inventory names and the NPCs
  present). Attribute scores moved to the `welcome` sheet, which also gained `resolve`.
* `mood.tension` removed.
* The prompt block names the obstacles present, without pips.
