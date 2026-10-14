"""Refresh the app's content snapshot from the TOME Forge.

    python tools/sync_content.py /path/to/tome/content         # a folder of Forge collection files
    python tools/sync_content.py tome-content-export.json      # a Forge "Export JSON" bundle

The game files are owned by the Forge; this only copies them into voidlit/content/tome and checks they load.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voidlit.tome.content import CODEX_DIR, Codex  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("source", help="a Forge content folder or an export bundle")
    p.add_argument("--out", default=str(CODEX_DIR))
    args = p.parse_args()
    src, out = Path(args.source), Path(args.out)
    codex = Codex.load(src)
    codex.validate()
    out.mkdir(parents=True, exist_ok=True)
    for name, entries in codex.c.items():
        blob = {"collection": name, "schemaVersion": 1, "entries": entries}
        if src.is_dir() and (src / f"{name}.json").exists():
            blob = json.loads((src / f"{name}.json").read_text())
        (out / f"{name}.json").write_text(json.dumps(blob, indent=2, ensure_ascii=False) + "\n")
    print(f"synced {len(codex.c)} collections, {sum(len(v) for v in codex.c.values())} entries -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
