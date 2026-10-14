# Tome protocol v1 (brain ⇄ body)

JSON text messages over a WebSocket. The brain listens on `ws://127.0.0.1:8765` (`python -m voidlit serve`).

## Client → brain

```json
{"type": "hello", "protocol": 1, "options": {"seed": 7, "background": "medic", "name": "Ash"}}
{"type": "input", "text": "I pry the iris open with the cutter"}
{"type": "command", "name": "save", "args": {"slot": "quicksave"}}
{"type": "command", "name": "load", "args": {"slot": "quicksave"}}
{"type": "command", "name": "new",  "args": {"seed": 8}}
{"type": "ping"}
```

## Brain → client

| type | when | payload |
|---|---|---|
| `welcome` | after `hello` or `new` | `protocol`, `scenario`, `backgrounds`, `character`, `llm` (bool) |
| `thinking` | as soon as `input` arrives | — (ripple the surface while the mind works) |
| `frame` | the result of a turn, or the opening | see below |
| `saved` / `loaded` | after commands | `slot` |
| `pong` | after `ping` | — |
| `error` | malformed or out-of-order messages | `message` |

## Frame

```json
{
  "type": "frame", "protocol": 1, "seq": 12, "over": false,
  "blocks": [
    {"role": "echo",      "spans": [{"text": "> I pry the iris open", "color": "#7f8796", "glow": 0.0, "effect": "none"}]},
    {"role": "dice",      "spans": [...], "data": {"dice": [5, 2, 6], "pool": 3, "hits": 2, "outcome": "success", "approach": "grit"}},
    {"role": "narration", "spans": [{"text": "The iris grinds open...", "color": "#c9d1e0", "glow": 0.0, "effect": "rise"}]},
    {"role": "npc",       "speaker": "Elara Voss", "data": {"npc": "elara"}, "spans": [...]},
    {"role": "event",     "spans": [{"text": "-1 HULL", "color": "#ff3d5a", "glow": 0.8, "effect": "pulse"}]},
    {"role": "prompt",    "spans": [...]}
  ],
  "status": {
    "character": {"name": "Ash", "background": "Medic", "grit": 1, "guile": 3, "insight": 2, "conditions": []},
    "hull": [4, 5], "stress": 3, "lumen": [5, 6],
    "chapter": {"id": "ration_vault", "title": "The Ration Vault", "act": 1, "index": 1, "total": 8},
    "obstacles": [{"id": "scavenger_drone", "name": "a scavenger drone", "kind": "enemy", "pips": ["guile", "grit"]}],
    "clocks": [{"id": "choir", "name": "The Choir Wakes", "filled": 2, "segments": 8}],
    "inventory": [{"name": "Medkit", "uses": 2}], "npcs": [{"id": "elara", "name": "Elara Voss", "color": "#6fe3cf"}],
    "won": false
  },
  "mood": {"lumen": 0.833, "tension": 0.41, "stress": 0.3, "dark": false, "pulse": "breath"},
  "effects": [{"kind": "shudder", "strength": 0.6}]
}
```

* **roles**: `title`, `echo`, `dice`, `narration`, `npc`, `event`, `prompt`. Render top to bottom.
* **span.color** is sRGB hex. **glow** (0–1) drives emissive intensity. **effect** is a hint:
  `rise` (default: voxels lift in reading order), `pulse`, `flicker`, `ripple`, `none`.
* **mood.lumen** scales the brightness of the whole surface. `dark` means only red and dread colours remain
  readable. `pulse` sets the surface's idle breathing rhythm.
* **effects**: `shudder` (camera and surface jolt), `dim`, `flare`, `panic` (glyphs scramble briefly), `ripple`.
* Start the server with `--debug` to add a `debug` object: the SensoryState, every NPC brain's decision, the
  Director's beat probabilities, and off-screen gossip. Useful for an in-editor overlay.
