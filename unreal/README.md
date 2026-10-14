# VoidlitTome: the Unreal Engine 5.8 body

The plugin renders the Tome: a 1:4:9 matte-black slab whose face is a grid of machined voxels. By default that
is 320 × 720 voxels at 1.25 mm, filling a 40 × 90 cm face. Flush voxels vanish into the black; raised ones (about
1 mm) glow and spell out the game. Everything on screen is voxels: your typing, the narration, NPC voices, dice,
obstacle pips, the status bar, and the Lumen dimming as your light runs out.

> **Status:** written against the UE 5.8 C++ API but not yet compiled. This repo's CI has no Unreal toolchain.
> Expect small fixes on the first build; the Python brain it talks to is fully tested.

## Setup

1. Copy `unreal/VoidlitTome/` into `<YourProject>/Plugins/VoidlitTome/` and regenerate project files.
   (C++ project required. The plugin uses the engine's `WebSockets`, `Json` and Slate modules.)
2. Build and open the editor. Enable **Python Editor Script Plugin** and **Editor Scripting Utilities** if they are
   off.
3. **Tools → Execute Python Script…** → `Plugins/VoidlitTome/Content/Python/create_tome_material.py`.
   This creates `/VoidlitTome/M_TomeVoxel` (Custom nodes that include
   `/Plugin/VoidlitTome/TomeVoxel.ush`) and `/VoidlitTome/M_TomeSlab`.
4. Start the mind (from the repo):
   ```bash
   cd brain && python -m voidlit serve --llama http://127.0.0.1:8080   # or omit --llama to play offline
   ```
   Or tick **Spawn Brain** on the actor and set *Python Executable* and *Brain Working Directory* (the repo's
   `brain/` folder).
5. Drag a **TomeMonolithActor** into a dark level, aim a camera at its face (+X), and press Play. Type; press Enter.
   `:save`, `:load` and `:new` work as commands.

## How it works

| Piece | File | Role |
|---|---|---|
| `ATomeMonolithActor` | `TomeMonolithActor.*` | Slab, voxel face, brain link, invisible Slate text box capturing the keyboard, optional sidecar process |
| `UTomeVoxelSurfaceComponent` | `TomeVoxelSurfaceComponent.*` | ISM of cubes (one per voxel, column/row in PerInstanceCustomData) plus two state textures; diffs each new page and starts only the transitions that changed |
| `FTomeTypesetter` | `TomeTypesetter.*` | Word-wraps frame blocks into a scrolling log; composes the page (status bar, log tail, prompt, input line) |
| `FTomeGlyphFont` | `TomeGlyphData.inl` | 8×16 voxel glyphs from DejaVu Sans Mono Bold (`Tools/gen_tome_glyphs.py`), including ▲ ◆ ● for approach pips |
| `UTomeBrainClient` | `TomeBrainClient.*` | WebSocket client for [docs/PROTOCOL.md](../docs/PROTOCOL.md), with reconnect backoff and Blueprint events |
| `TomeVoxel.ush` | `Shaders/` | All animation on the GPU: smoothstep rises with typewriter delays, pulse and flicker, page ripples, the "thinking" breath, panic jolts, the heartbeat, Lumen dimming and the dread-red Dark |

### Performance notes

* 230k instances × 12 triangles ≈ 2.8M triangles, with no per-frame CPU work. Textures are only rewritten when a
  frame arrives or the player types (≈4.6 MB upload). Lower `Columns`/`Rows` and raise `VoxelPitchCm` for
  weaker GPUs.
* Shadows are off by default (`CastShadow`). Turn them on for close-up shots: millimetre voxel shadows sell the
  machined look.
* For a camera far from the slab, a future LOD could swap the ISM for a single plane sampling the same textures.
