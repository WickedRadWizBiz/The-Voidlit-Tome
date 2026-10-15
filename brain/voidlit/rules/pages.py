"""The TOME's twelve Pages: powers it lends, each raising its Hunger once opened (Core Rules 0.2, "Pages").

A Page is opened at a Haven for 4 Insight, or for free at a Temptation Echo. Using one costs what it lists.
"""

from __future__ import annotations

import random

from .bearer import Kind
from .events import Event
from .trial import PAGE_IDS, Trial

OPEN_COST = 4
USE = {
    "ward": "Ignore all Harm from one source (1 Stress)",
    "unlock": "Clear a Relic chapter instantly (1 Lumen die)",
    "read": "See the shapes of the next 3 Leaves (once per trial)",
    "mend": "Heal one Wound fully (2 Stress)",
    "unmake": "Deal 4 progress to any enemy (1 Lumen die)",
    "quiet": "Remove 3 Stress (1 Harm)",
    "kindle": "Add 1 Lumen die (2 Stress)",
    "command": "A living Witness obeys one order (1 Stress)",
    "step": "Skip a chapter you just drew; draw again (1 Lumen die)",
    "borrow": "Reroll any roll, keeping the better (1 Stress)",
    "speak": "The TOME answers any one question truthfully and fully (2 Stress)",
    "become": "For one chapter, every roll gains 2 dice (a Panic roll afterward)",
}


def find_page(text: str) -> str | None:
    low = text.lower()
    return next((p for p in PAGE_IDS if p in low), None)


def use_page(t: Trial, page: str, rng: random.Random) -> tuple[bool, str]:
    """Spend a Page. Returns (used, why-not). ``borrow`` is spent through ``Trial.decide_push``;
    ``speak`` needs the engine to supply the truth, so it only pays the cost here."""
    if page not in t.pages_open:
        return False, "That Page is not open."
    ch = t.chapter
    b = t.bearer
    kind = ch.active_kind if ch else None
    mark = len(t.log)
    if page == "ward":
        t.ward = True
        t.add_stress(1, rng)
    elif page == "unlock":
        if kind is not Kind.RELIC or ch.crucible:
            return False, "Unlock opens a Relic chapter."
        t.dim_by(1)
        ch.progress = ch.depth
        t._settle(rng)
    elif page == "read":
        if "read" in t.used:
            return False, "Read works once per trial."
        t.used.append("read")
        t.peek = [leaf.shape for leaf in reversed(t.deck[-3:])]
    elif page == "mend":
        hurt = list(b.wounds)
        if not hurt:
            return False, "There is no Wound to mend."
        b.wounds.remove(max(hurt, key=lambda w: w.roll))
        t.add_stress(2, rng)
    elif page == "unmake":
        if kind is not Kind.HOSTILE:
            return False, "Unmake strikes an enemy."
        t.dim_by(1)
        ch.progress = min(ch.depth, ch.progress + 4)
        t._settle(rng)
    elif page == "quiet":
        b.stress = max(b.min_stress, b.stress - 3)
        t.harm(1, rng, guardable=False, source="quiet")
    elif page == "kindle":
        t.kindle(1)
        t.add_stress(2, rng)
    elif page == "command":
        if kind is not Kind.WITNESS or ch.crucible:
            return False, "Command needs a living Witness."
        ch.progress = ch.depth
        t.add_stress(1, rng)
        t._settle(rng)
    elif page == "step":
        if ch is None or ch.crucible or ch.casts:
            return False, "Step skips a chapter only before its first cast."
        t.dim_by(1)
        t.chapter = None
        t.phase = "draw"
        t.chapters_seen -= 1
    elif page == "speak":
        t.add_stress(2, rng)
    elif page == "become":
        if ch is None:
            return False, "Become lasts one chapter; draw one first."
        t.become = True
    elif page == "borrow":
        return False, "Borrow is used on a roll that missed."
    t.log.insert(mark, Event("page", f"PAGE: {page.upper()}", "page", f"You use the {page.title()} Page.",
                             {"page": page}))
    return True, ""
