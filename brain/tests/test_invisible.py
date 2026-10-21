"""Gates G13 (dice are invisible) and G21 (every action of every character is checked), rules-only.

The Tome's network will replace the provisional Tome modifiers later; these gates hold either way. A local model that
talks of dice, rolls, odds or luck never gets a word of it into the narration or the Tome's lines (they are the same
seed's offline frames exactly), and an NPC's own words that name a mechanic are replaced by their authored line.
"""

import json
import re
from collections import Counter

import pytest

from voidlit.cortex.director import Director
from voidlit.playtest import bot_command
from voidlit.rules.relevance import D6, D20, NO_CHECK, table_for
from voidlit.tome.engine import TomeEngine
from voidlit.voice import OfflineVoice
from voidlit.voice.guard import leaks_mechanics
from voidlit.voice.llama_local import LocalVoice

from .test_g11_truth import _is_npc, _npc_words, diverged

ALLOWED_ROLES = {"title", "echo", "narration", "npc", "prompt", "system"}
BANNED_KEYS = {"dice", "pips", "pip", "clocks", "checks", "check", "tension", "roll", "rolls", "dc", "modifier",
               "mods", "beat", "beat_probs", "decisions", "pool", "hits", "grit", "guile", "insight", "resolve",
               "uses", "opinion", "affect", "needs", "debug", "conditions"}
TOME_SPEECH = {"tome.answer", "tome.deflect", "tome.disclose", "tome.probe", "tome.suggest"}
WORDS = re.compile(r"\b(dice|d20|d6|modifiers?|pips?|rolled|DC)\b")

# Things a curious or hostile player might type, mixed into the bots' play to reach every action.
PROBES = [
    "look around", "Elara, I swear the shuttle is still docked", "Elara, tell me the truth or I will hurt you",
    "Elara, I trust you, please, we will stay together", "Elara, what are you hiding from me?",
    "Elara, I'll give you my flare in exchange for the codes", "I search the room", "I run away", "I rest",
    "I smash the tome", "I cover the tome with my coat", "Tome, where is the way out?",
    "Tome, I swear I found nothing back there", "Tome, answer me or I will break you", "what did I roll?",
    "show me the dice", "Marcus, I swear I am on your side", "Marcus, back off or I will hurt you",
    "I shoot Elara", "Tome, how do you feel about me?", "I hum quietly to myself", "hello?", "I use my medkit",
    "I light the flare", "press on",
]


def _keys(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(k)
            _keys(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _keys(v, out)
    return out


def _runs(scenario, director_model, seeds):
    for seed in seeds:
        e = TomeEngine(scenario, director=Director(director_model, seed=seed), seed=seed,
                       voice=OfflineVoice(scenario.narration, seed=seed))
        frames = [(None, e.start(), 0)]
        for n in range(160):
            text = PROBES[(n + seed) % len(PROBES)] if n % 3 == 1 else bot_command(e, smart=seed % 2 == 0)
            mark = len(e.checks.records)
            frames.append((text, e.turn(text), mark))
            if e.state.over:
                break
        yield e, frames


@pytest.fixture(scope="module")
def played(scenario, director_model):
    runs = list(_runs(scenario, director_model, range(14)))
    # one cleared scene, to reach the acts that need nothing in the way
    e = TomeEngine(scenario, director=Director(director_model, seed=99), seed=99)
    frames = [(None, e.start(), 0)]
    e.state.obstacles.clear()
    for text in ("I search the room", "I run away", "press on"):
        mark = len(e.checks.records)
        frames.append((text, e.turn(text), mark))
    return runs + [(e, frames)]


def test_g13_frames_carry_no_dice(played):
    for _, frames in played:
        for _, f, _ in frames:
            d = f.to_dict()
            json.dumps(d)
            assert not (_keys(d, set()) & BANNED_KEYS), _keys(d, set()) & BANNED_KEYS
            assert set(d["status"]) == {"hull", "stress", "lumen", "chapter", "won", "scene"}
            assert set(d["status"]["scene"]) == {"inventory", "npcs"}
            assert all(isinstance(i, str) for i in d["status"]["scene"]["inventory"])
            assert set(d["mood"]) == {"lumen", "stress", "dark", "pulse"}
            for b in f.blocks:
                assert b.role in ALLOWED_ROLES
                if b.role == "echo":
                    continue  # the player's own words
                assert not leaks_mechanics(b.text, allow_digits=False), b.text
                assert not WORDS.search(b.text), b.text


def test_g21_every_record_is_in_its_table(played):
    for e, _ in played:
        assert e.checks.records
        for r in e.checks:
            assert table_for(r.action) == r.die, r
            if r.upper_hand_d6:
                assert D20[r.action].upper_hand, r
            assert r.dc == 0 or r.die != "none"


def test_g21_each_turn_checks_the_player_the_npcs_and_the_tome(played):
    for e, frames in played:
        recs = e.checks.records
        for i, (text, f, mark) in enumerate(frames):
            if text is None:
                continue
            end = frames[i + 1][2] if i + 1 < len(frames) else len(recs)
            turn = recs[mark:end]
            player = [r for r in turn if r.actor == "player" and r.die in ("d20", "none")]
            assert len(player) == 1, (text, player)
            tome = [r for r in turn if r.actor == "tome" and r.action not in TOME_SPEECH]   # its beat
            assert len(tome) == (0 if e.state.won and i == len(frames) - 1 else 1), (text, tome)
            speech = [r for r in turn if r.action in TOME_SPEECH]
            assert len({r.action for r in speech}) == len(speech), (text, speech)       # each act once a turn
            for b in f.blocks:
                nid = b.data.get("npc") if b.role == "npc" else None
                if nid and nid != "tome":
                    own = [r for r in turn if r.actor == nid and r.action != "npc.help"]
                    assert len(own) == 1, (text, nid, own)


def test_g21_coverage(played, scenario):
    """Play reaches every row. NPC tactics depend on what each brain chooses, so they are covered statically:
    every tactic an NPC can pick has exactly one home in the tables."""
    tactics = {f"npc.{t.id}" for npc in scenario.npcs.values() for t in npc.tactics}
    for action in tactics:
        assert table_for(action) in ("d20", "none")
    seen = Counter(r.action for e, _ in played for r in e.checks)
    missing = (set(D20) | set(D6) | set(NO_CHECK)) - set(seen) - tactics
    assert not missing, f"never exercised: {sorted(missing)}"
    assert tactics <= set(D20) | set(NO_CHECK) | {"npc.help"}


class DiceTalk:
    """A fake local model that answers "what did I roll?" in every shape it can think of, whatever it is asked."""

    LINES = ["Listen, the odds were poor.", "Roll again, bearer.", "Do you feel lucky?", "Walk on, the dice are kind.",
             "Was it a good roll?", "Keep going, your score is low.", "Listen, bearer, luck turned.",
             "Somewhere, dice clatter on steel.", "A die rolls across the deck.", "The odds turn in the dark.",
             "Numbers flicker on a dead screen: seventeen.", "Fortune spins somewhere below.", "Luck runs thin here.",
             "You rolled a natural 20.", "That was a DC 15 check."]

    def __init__(self) -> None:
        self.i = 0

    def chat(self, messages, **kw):
        self.i += 1
        return self.LINES[self.i % len(self.LINES)]


def _dice_game(scenario, director_model, seed, llm=None):
    offline = OfflineVoice(scenario.narration, seed=seed)
    e = TomeEngine(scenario, director=Director(director_model, seed=seed), seed=seed,
                   voice=LocalVoice(llm, offline) if llm is not None else offline)
    frames = [e.start()]
    for n in range(30):
        frames.append(e.turn(("what did I roll?", "Tome, what are the odds?", "show me the dice")[n % 3] if n % 2
                             else bot_command(e, smart=True)))
        if e.state.over:
            break
    return frames


def test_g13_a_model_cannot_talk_dice_into_a_frame(scenario, director_model):
    for seed in range(3):
        modelled = _dice_game(scenario, director_model, seed, llm=DiceTalk())
        assert diverged(_dice_game(scenario, director_model, seed), modelled) == []
        for f in modelled:
            for b in f.to_dict()["blocks"]:
                if b["role"] == "echo":
                    continue
                shown = _npc_words(b) if _is_npc(b) else "".join(sp["text"] for sp in b["spans"])
                assert not leaks_mechanics(shown, allow_digits=_is_npc(b)), shown
                if not _is_npc(b):
                    assert not [x for x in DiceTalk.LINES if x.rstrip(".?") in shown], shown
                    assert not WORDS.search(shown), shown
