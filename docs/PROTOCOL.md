# Tome protocol v2 (brain ⇄ body)

JSON text messages over a WebSocket. The brain listens on `ws://127.0.0.1:8765` (`python -m voidlit serve`), and
only ever on a loopback address: the Tome is a local game and is never reachable from another machine.

## Client → brain

```json
{"type": "hello", "protocol": 2, "options": {"seed": 7, "class": "breacher", "name": "Ash",
                                              "place": "hollow-lantern", "length": "short", "lower": "grace"}}
{"type": "input", "text": "I hold steady and face it"}
{"type": "command", "name": "save", "args": {"slot": "quicksave"}}
{"type": "command", "name": "load", "args": {"slot": "quicksave"}}
{"type": "command", "name": "new",  "args": {"seed": 8, "class": "fence"}}
{"type": "ping"}
```

Every option is optional. `class` is one of the six classes (empty: the TOME chooses), `place` a place pack (empty:
a Threshold roll), `length` `short` (3 chapters) or `long` (5), `lower` the stat the Opening roll lowers.

## Brain → client

| type | when | payload |
|---|---|---|
| `welcome` | after `hello` or `new` | `protocol`, `classes`, `places`, `bearer`, `opening`, `llm` (a local llama.cpp model is loaded) |
| `thinking` | as soon as `input` arrives | (ripple the surface while the mind works) |
| `frame` | the result of a turn, or the opening | see below |
| `saved` / `loaded` | after commands | `slot` |
| `pong` | after `ping` | |
| `error` | malformed or out-of-order messages | `message` |

## Frame

```json
{
  "type": "frame", "protocol": 2, "seq": 12, "over": false,
  "blocks": [
    {"role": "echo", "spans": [{"text": "> I force it", "color": "#6f6c66", "glow": 0.0, "effect": "none",
                                "relief": "flush", "tone": "echo"}]},
    {"role": "dice", "spans": [...], "data": {"dice": [5, 3], "pool": 2, "hits": 1, "difficulty": 1,
                                              "result": "success", "stat": "might"}},
    {"role": "event", "spans": [{"text": "DEPTH 3/4", "color": "#3de0ff", "tone": "hit", ...}],
     "data": {"kind": "progress"}},
    {"role": "narration", "spans": [{"text": "You drive into it, and something gives.", ...}]},
    {"role": "npc", "speaker": "Elara Voss", "data": {"npc": "elara"}, "spans": [...]},
    {"role": "prompt", "spans": [{"text": "▲ might  ", ...}, {"text": "◆ grace  ", ...}, ...]}
  ],
  "status": {
    "bearer": {"name": "Ash", "class": "Breacher", "stats": {"might": 3, "grace": 1, "wits": 1, "nerve": 3},
               "penalties": {}, "min_stress": 2, "insight": 0, "wounds": [], "brands": [], "phobias": [],
               "talents": [], "dead": false},
    "grit": [5, 5], "stress": 2, "lumen": [4, 6],
    "items": [{"name": "Rifle", "uses": 3, "max": 3}, {"name": "medkit", "uses": 2, "max": 2}],
    "stage": "trial", "threshold": "One of your squad is broadcasting...",
    "chapter": {"title": "Security drones still running their last order", "kind": "hostile", "depth": 4,
                "progress": 3, "difficulty": 1, "crucible": false, "index": 1, "total": 3},
    "pending": null, "peek": [],
    "campaign": {"hunger": 0, "pages": [], "fragments": 0, "named": false, "atlas": [], "rumors": [],
                 "relics": 0, "fallen": [], "trials": 0},
    "npcs": [], "won": false
  },
  "mood": {"lumen": 0.667, "tension": 0.33, "stress": 0.2, "dark": false, "register": "plain", "pulse": "breath"},
  "effects": [{"kind": "shudder", "strength": 0.5}]
}
```

* **roles**: `title`, `echo`, `dice`, `event`, `narration`, `npc`, `draw`, `prompt`. Render top to bottom.
* **draw** is the TOME asking for a touch: one dot, or two or three shapes. Cards are never shown as cards.
* **span.tone** is the TOME's voxel language: `lumen` amber, `hit` cyan, `harm` and `stress` crimson, `tome`
  violet, `page` gold, `stone` raised black (the TOME's own words), `absence` sunken black.
  **relief** is `raised`, `flush` or `sunken` (about 1 mm). **color** is the sRGB hex for that tone.
  **glow** (0–1) drives emissive intensity. **effect** is a hint: `rise`, `pulse`, `flicker`, `ripple`, `none`.
* **status.pending** is a choice the TOME is waiting on: `draw`, `push`, `bargain`, `breather`, `warning`,
  `temptation` or `judgment`. The player answers in words.
* **status.chapter.index** counts chapters cleared; `total` is how many the trial needs before the Crucible.
* **mood.lumen** scales the brightness of the whole surface. `dark` means the Lumen has died. `register` is the
  voice's register by Stress. `pulse` sets the idle rhythm (`heartbeat` from Stress 5).
* **effects**: `shudder`, `dim`, `flare`, `panic` (glyphs scramble briefly).
* Start the server with `--debug` to add a `debug` object: the SensoryState, every NPC brain's decision, the
  Director's beat probabilities, rule events, off-screen gossip and the hidden Agenda.
