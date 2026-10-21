# Tome brain prototypes (evidence, not product code)

These are the throwaway scripts behind the **[M]** (measured) tags in
[docs/TOME_BRAIN_DESIGN.md](../../docs/TOME_BRAIN_DESIGN.md). Three design teams wrote them while competing on the
Tome's network: `bio/` (the biophysical design that won), `rt/` (engineering/runtime) and `gm/` (the cunning
game-master design). Files at the top level are earlier single-cell, column and timing experiments.

They are kept so every measured claim stays reproducible. They are unpolished: paths are hard-coded to the
original session (`/home/user/The-Voidlit-Tome/brain` on `sys.path`, a JAX venv), and several monkeypatch the engine
as it was before the invisible-check rules landed. The product code lives in `brain/voidlit/`; nothing imports
from here.
