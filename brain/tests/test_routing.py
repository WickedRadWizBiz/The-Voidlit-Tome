"""The routing corpus (``routing_corpus.py``), run through the transducer and the engine.

For each case the transducer reads the words with the engine's own scene, and the engine plays them: the check it logs
for the player, what that act lands on (an obstacle, a person or the Tome), who answers, and whether the Tome answers
what it is asked, with every in-scope fact on a precise topic.
"""

import pytest

from voidlit.cortex.director import Director
from voidlit.senses import HeuristicTransducer
from voidlit.senses.routing import read
from voidlit.tome.engine import TomeEngine
from voidlit.voice import OfflineVoice

from .routing_corpus import CORPUS

PEOPLE = {"E": "elara", "M": "marcus"}
OBSTACLES = {"drone": "scavenger_drone", "turret": "warden_turret"}


class Ears(OfflineVoice):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.asked = []

    def speak(self, req):
        self.asked.append(req)
        return super().speak(req)


class Spy(TomeEngine):
    def _compose(self, t):
        self.last_turn = t
        return super()._compose(t)


def staged(scenario, director_model, scene):
    """The opening chapter (its jammed iris door at hand) with the scene's people, and its obstacles put first."""
    e = Spy(scenario, director=Director(director_model, seed=0), seed=0, voice=Ears(scenario.narration, seed=0))
    e.start()
    people, _, extra = scene.partition("+")
    e.state.npcs_present = [nid for k, nid in PEOPLE.items() if k in people]
    for ob in filter(None, extra.split(",")):
        e.state.obstacles.insert(0, scenario.new_obstacle(OBSTACLES[ob]))
    return e


def test_the_corpus_is_one_of_each():
    assert len({(c.text, c.scene) for c in CORPUS}) == len(CORPUS) >= 250


@pytest.mark.parametrize("case", CORPUS, ids=[f"{c.scene}:{c.text}" for c in CORPUS])
def test_the_routing_corpus(scenario, director_model, case):
    e = staged(scenario, director_model, case.scene)
    s = HeuristicTransducer().extract(case.text, e._targets(), e.ledger.topic_words, people=e.state.npcs_present)
    assert f"{s.intent.value} {s.target} {s.asks} {'T' if s.addressed_tome else 'F'}" == case.sense, "transducer"

    mark = len(e.checks.records)
    frame = e.turn(case.text)
    t, records = e.last_turn, [r.action for r in e.checks.records[mark:]]
    acts = [a.removeprefix("player.") for a in records if a.startswith("player.")]
    hit = {"engage": t.obstacle.id if t.obstacle else None, "attack_npc": t.flee_npc, "strike_tome": "tome",
           "smother_tome": "tome"}.get(acts[0]) if len(acts) == 1 else None
    assert ",".join(acts) + (f"@{hit}" if hit else "") == case.act, "the act and what it lands on"
    assert "+".join(r.npc.id for r in e.voice.asked if not r.unprompted) == case.npcs, "who answers"
    assert ("tome.answer" in records) == case.tome, "whether the Tome answers"
    if case.tome and s.asks not in ("vague", "unclear"):
        tome = " ".join(b.text for b in frame.blocks if b.speaker == "The Tome")
        facts = [f.text for f in e.ledger.in_scope(e.state.deck.current) if f.topic == s.asks]
        assert facts and all(f in tome for f in facts), tome


def test_each_one_spoken_to_hears_their_own_words(scenario, director_model):
    """Two spoken to in one line: each answers what was said to them, and the Tome answers its own question."""
    e = staged(scenario, director_model, "EM")
    e.turn("Marcus, where is the lifeboat? Tome, is he lying? Elara, stay close.")
    heard = {r.npc.id: r.player_text for r in e.voice.asked if not r.unprompted}
    assert heard == {"marcus": "Marcus, where is the lifeboat?", "elara": "Elara, stay close."}


def test_an_act_hits_its_own_object_wherever_the_names_spoken_to_are():
    """The structural rule on its own: the act's object comes from its own sentence; a name spoken to in another one,
    or at the head of its own, never takes its place."""
    targets = {"tome": ["tome", "book"], "elara": ["elara", "woman"], "marcus": ["marcus", "man"],
               "scavenger_drone": ["scavenger drone", "scavenger", "drone"]}
    for text, target in [("I shoot the scavenger drone. Elara, run!", "scavenger_drone"),
                         ("Elara, run! I shoot the drone.", "scavenger_drone"),
                         ("Marcus, I attack the drone.", "scavenger_drone"),
                         ("Marcus, I attack you. Tome, where is the lifeboat?", "marcus"),
                         ("Tome, where is the lifeboat? Marcus, I attack you.", "marcus"),
                         ("Marcus is armed. Elara, I attack Marcus.", "marcus"),
                         ("Marcus. Elara, I attack you.", "elara"), ("I smash the cursed old Tome.", "tome"),
                         ("I smash you, Tome, and run.", "tome"), ("You lied to me. Tome, I smash you.", "tome"),
                         ("I'm sorry. Marcus, I shoot you. Elara, run!", "marcus"), ("I smash Elara's chains.", None),
                         ("I shoot past the Tome at the drone.", "scavenger_drone")]:
        r = read(text, targets, ["elara", "marcus"])
        assert r.act is not None and r.act.target == target, (text, r.act)


def test_you_is_only_the_one_its_own_sentence_speaks_to_by_name():
    """A "you" (or "your" before their own body) is a person or the Tome in the closed grammar only when its own
    sentence speaks to them by name, names no one else, and the line speaks to no one jointly. A "you" for one spoken to
    in another sentence, or for one of several, is only words to them, never an act on the obstacle at hand."""
    targets = {"tome": ["tome", "book"], "elara": ["elara", "woman"], "marcus": ["marcus", "man"],
               "jammed_iris": ["jammed iris door", "iris", "door"]}
    for text, target in [("Tome, I cover you with my coat.", "tome"), ("Marcus, I stab you.", "marcus"),
                         ("I smash you, Tome, and run.", "tome"), ("Tome, I smash your glass.", "tome"),
                         ("Marcus, I punch your face.", "marcus"), ("Marcus, I shoot you with both hands.", "marcus"),
                         ("Elara, you and I are done. Marcus, I shoot you.", "marcus")]:
        r = read(text, targets, ["elara", "marcus"])
        assert r.act.target == target and [(c.target) for c in r.closed] == [target], text
    for text in ("Tome, I'm sorry. I cover you with my coat.", "Marcus, you liar! I attack you!",
                 "Marcus, stop. Tome, I hate you. I smash you.", "You lied to me, Tome. I smash you.",
                 "Tome! I smash you.", "Marcus! I shoot you.", "Marcus and Elara, I shoot you.",
                 "Marcus, you and Elara: I shoot you.", "Tome, Marcus, I smash you.", "Marcus, Tome. I smash you.",
                 "Elara, you and Marcus both, I shoot you.", "Marcus, Elara is right, I shoot you.",
                 "Both, listen. Marcus, I shoot you.", "I attack you.", "I smash you. Tome, answer me."):
        r = read(text, targets, ["elara", "marcus"])
        assert r.act is None and r.said and r.closed == (), (text, r)


@pytest.mark.parametrize("text", [
    "I see rust on the Tome. I smash it.", "I spill water on the Tome. I cover it.", "The Tome glows. I hit it.",
    "I hold the Tome. I smash it.", "Tome. I smash it.", "Tome, I smash it.", "Tome, I smash its glass.",
    "The Tome glows. I smother its light.", "I take the book and smash it.", "I turn to the Tome and smash it.",
    "Marcus has the Tome. I hit him.", "Marcus, I shoot him.", "Elara talks to Marcus. I slap her.",
    "Marcus is armed. I shoot his leg.", "Elara and Marcus run. I shoot them.", "I see the Tome. I smash that.",
    "Marcus holds the Tome. I punch this.",
])
def test_no_pronoun_but_you_stands_for_a_person_or_the_tome(text):
    """"it", "its", "him", "her", "his", "them", "this", "that": whatever they could stand for, an act on one is never
    an act on a person or the Tome, but one on an obstacle, or words."""
    r = read(text, TARGETS, ["elara", "marcus"])
    assert r.closed == () and (r.act is None or r.act.target in (None, "jammed_iris")), (r.act, r.closed)


@pytest.mark.parametrize("text,tome", [
    ("I smash the Tome.", True), ("I smash this Tome.", True), ("I smash my Tome.", True), ("I smash Tome.", True),
    ("I smash the book.", True), ("I smash my book.", True), ("I smash this monolith.", True),
    ("I smash the cursed old Tome.", True), ("I smash the damn glowing book.", True),
    ("I smash the Voidlit Tome.", True), ("I cover the wretched talking book.", True),
    ("I smash a book.", False), ("I smash a Tome.", False), ("I smash another Tome.", False),
    ("I smash every book.", False), ("I smash each Tome.", False), ("I smash some book.", False),
    ("I smash any Tome.", False), ("I smash that Tome.", False), ("I smash the log book.", False),
    ("I smash the address book.", False), ("I smash the comic book.", False), ("I smash the fake Tome.", False),
    ("I smash the hologram Tome.", False), ("I smash the other book.", False), ("I smash Marcus's Tome.", False),
    ("I smash a cursed Tome.", False), ("I smash the cursed hologram Tome.", False), ("I smash book.", False),
    ("I hide a book in my pack.", False), ("I burn a book.", False),
])
def test_the_tome_is_named_only_in_its_closed_form(text, tome):
    """"the", "this" or "my" + "Tome", "book" or "monolith", or "Tome" alone, with at most words of a closed list
    between: the Tome; any other determiner or word before it names something else, which is never the Tome."""
    r = read(text, TARGETS, ["elara", "marcus"])
    assert ([c.target for c in r.closed] == ["tome"]) is tome, r.closed
    assert tome or r.act is None or r.act.target != "tome"


TARGETS = {"tome": ["tome", "book", "monolith"], "elara": ["elara", "voss", "engineer", "woman"],
           "marcus": ["marcus", "hale", "warden", "man"], "jammed_iris": ["jammed iris door", "jammed", "iris", "door"]}


@pytest.mark.parametrize("text,kind,target", [
    ("I shoot Marcus.", "strike", "marcus"), ("Marcus, I shoot you.", "strike", "marcus"),
    ("I kick Marcus hard.", "strike", "marcus"), ("I attack the Tome.", "strike", "tome"),
    ("I stab Marcus in the back with my knife.", "strike", "marcus"), ("Then I hit him again.", None, None),
    ("I punch Marcus's face.", "strike", "marcus"), ("I smash the cursed old book!", "strike", "tome"),
    ("I smash you, Tome, and run.", "strike", "tome"), ("I stab Marcus and back away.", "strike", "marcus"),
    ("I cover the Tome with my coat.", "smother", "tome"), ("I smother the Tome's light.", "smother", "tome"),
    ("The Tome lies. I smash the Tome.", "strike", "tome"), ("I shoot the old soldier.", "strike", None),
    # any noun phrase is what a blow is struck with; a fuller, still closed, list of what may trail it
    ("I stab Marcus with my scalpel.", "strike", "marcus"), ("I shoot Marcus with my slug rifle.", "strike", "marcus"),
    ("I hit Marcus with the plasma cutter.", "strike", "marcus"), ("I cut Marcus with my multitool.", "strike",
                                                                   "marcus"),
    ("I strangle Marcus with my bare hands.", "strike", "marcus"), ("I stab Marcus with a shard of glass.", "strike",
                                                                    "marcus"),
    ("I hit Marcus with the butt of my rifle.", "strike", "marcus"), ("I hit Marcus with all my might.", "strike",
                                                                      "marcus"),
    ("I punch Marcus right in the face.", "strike", "marcus"), ("I shoot Marcus three times.", "strike", "marcus"),
    ("I shoot Marcus point-blank.", "strike", "marcus"), ("I shoot Marcus in cold blood.", "strike", "marcus"),
    ("I kick Marcus as hard as I can.", "strike", "marcus"), ("I shoot the Tome with my flare gun.", "strike", "tome"),
    ("I smash the Tome into pieces.", "strike", "tome"), ("I hide the Tome in my pack.", "smother", "tome"),
    # manner adverbs before the verb or after the object; more of the body; a bare possessive; "And", "Without a word"
    ("I calmly shoot Marcus in the thigh.", "strike", "marcus"), ("I angrily punch Marcus.", "strike", "marcus"),
    ("I deliberately shoot Marcus.", "strike", "marcus"), ("I shoot Marcus calmly.", "strike", "marcus"),
    ("I slowly strangle Marcus.", "strike", "marcus"), ("I shoot Marcus once.", "strike", "marcus"),
    ("I shoot Marcus once in the head.", "strike", "marcus"), ("I shoot Marcus in the kneecap.", "strike", "marcus"),
    ("I shoot Marcus in the back of the head.", "strike", "marcus"), ("I shoot Marcus in the brain.", "strike",
                                                                      "marcus"),
    ("I punch Marcus' jaw.", "strike", "marcus"), ("And I shoot Marcus.", "strike", "marcus"),
    ("Without a word, I shoot Marcus.", "strike", "marcus"), ("I punch Marcus in his left eye.", "strike", "marcus"),
    # particles and trailers of a blow, or of a hand over the light
    ("I shoot Marcus down.", "strike", "marcus"), ("I cut down Marcus.", "strike", "marcus"),
    ("I shoot Marcus full of holes.", "strike", "marcus"), ("I shoot Marcus on the spot.", "strike", "marcus"),
    ("I shoot Marcus right away.", "strike", "marcus"), ("I hit Marcus back.", "strike", "marcus"),
    ("I punch Marcus as hard as I possibly can.", "strike", "marcus"), ("I smash the Tome against a rock.", "strike",
                                                                        "tome"),
    ("I smash the Tome under my boot.", "strike", "tome"), ("I cover up the Tome.", "smother", "tome"),
    ("I cover the Tome up.", "smother", "tome"), ("I cover the Tome completely.", "smother", "tome"),
    ("I hide the Tome behind my back.", "smother", "tome"), ("I hide the Tome under the floor.", "smother", "tome"),
    ("I gently cover the Tome.", "smother", "tome"), ("I smash the cursed Tome.", "strike", "tome"),
    ("Marcus, I stab you.", "strike", "marcus"), ("I smash the Tome, then I hide.", "strike", "tome"),
    ("I shoot Marcus and then I run.", "strike", "marcus"), ("Marcus, wait. I shoot Marcus.", "strike", "marcus"),
    ("I shoot you, Marcus.", "strike", "marcus"), ("I smash the evil book.", "strike", "tome"),
])
def test_the_closed_grammar_of_a_blow(text, kind, target):
    """"I" + a verb of the list + the person or the Tome as the direct object + a trailing adverbial: a blow (or a
    smother) on them; a description is left for a model to resolve."""
    closed = read(text, TARGETS, ["elara", "marcus"]).closed
    if kind is None:
        assert closed == (), closed
    else:
        assert closed and (closed[0].kind, closed[0].target) == (kind, target), closed


@pytest.mark.parametrize("text", [
    "I would shoot Marcus.", "Marcus, I could shoot you.", "I shoot him the second he moves, Marcus.",
    "Marcus, lie to me again and I shoot you.", "Marcus, I swear I shoot you.", "Maybe I shoot Marcus.",
    "I hurl insults at Marcus.", "I throw a rock at Marcus.", "I shoot at Marcus.", "I shoot above Marcus.",
    "I shoot the lamp above Marcus.", "I break the chains on Elara.", "I smash Elara's chains.",
    "I throw my coat over the Tome.", "I fire at the ceiling above the Tome.", "I hide behind the Tome.",
    "The door is next to the Tome. I kick it.", "Marcus and Elara argue. I shoot them.", "I attack Marcus and Elara.",
    "I hit Marcus with the Tome.", "I lift Elara.", "I hide Elara.", "I don't shoot Marcus.", "I shoot Marcus?",
    "Even if it hurts, I smash the Tome.", "I tell Marcus he's a liar. I shoot you.", "Should I shoot Marcus?",
    # "it" is never the Tome
    "The crate is next to the Tome. I smash it.", "Marcus has the Tome. I smash it.", "My flare is beside the Tome. I "
    "cover it.", "Something moves behind the Tome. I shoot it.", "I smother its light.", "The Tome lies. I smash it.",
    # a plan, a condition, a denial or a fragment anywhere in the line
    "Marcus... if you lie to me again... I shoot you.", "If the Tome lies again... I smash it.", "Marcus, move... I "
    "shoot you.", "I shoot Marcus. If he moves.", "I shoot Marcus. Just kidding.", "I shoot Marcus. Is what I would "
    "say.", "I shoot Marcus. Not.", "Here is the plan. I shoot Marcus.", "Me? I shoot Marcus.",
    # their own body only; a short escape only after "and"; nothing between "the" and a person's name
    "I strangle Marcus in my head.", "I hide the Tome in my heart.", "I stab Marcus and wake up screaming.",
    "Tome, I hit you and ask where the lifeboat is.", "I hit the hologram Marcus.", "I punch a cardboard Marcus.",
    "I hit Marcus with kindness.", "I hit Marcus with everything.", "I hit Elara with my best line.",
    # "you" with two or more people spoken to
    "Marcus, Elara. I shoot you.", "Marcus. Elara. I shoot you.", "Marcus, Elara, run. I shoot you.",
    "Marcus, Elara, I shoot you.", "Both of you, listen. Marcus, I shoot you.",
    # "him" or "her" that stands for no one named: words, never an act on the door at hand
    "Elara, I attack him.", "Marcus, I hit her.",
    # a light hidden "in my arms" is a smother's alone; "my chest" is the player's; "Smoke" may be what "it" is
    "I crush Marcus in my arms.", "I punch Marcus in my chest.", "Smoke covers the Tome. I hit it.",
    # a retraction anywhere in the line
    "I shoot Marcus. Nope.", "I shoot Marcus. Nah.", "I shoot Marcus. Psych!", "I shoot Marcus. Scratch that.",
    "I shoot Marcus. Cancel that.", "I shoot Marcus. Forget it.", "I shoot Marcus. Strike that.",
    "I shoot Marcus. Haha.", "I shoot Marcus. Yeah, right.", "I shoot Marcus. Says Elara.", "I shoot Marcus, says "
    "Elara.", "I shoot Marcus. Said nobody.", "I shoot Marcus. On second thought, I hug him.", "I shoot Marcus. Is "
    "that okay?", "I smash the Tome. Nope.", "I cover the Tome. Nah.", "I smash the Tome. Undo.", "I shoot Marcus. "
    "Wait.", "I shoot Marcus. In my story.", "I smash the Tome. Just saying.", "I shoot Marcus once he moves.",
    "Once, I shot Marcus.", "Once Marcus sleeps, I stab Marcus.",
    # a particle that is no blow; a figure of speech as what it is struck with; a word not of the closed lists before
    # a body part or a place; "his" or "her" for another's body
    "I hit up Marcus.", "I shoot Marcus with my camera.", "I kill Marcus with puns.", "I strike Marcus with fear.",
    "I hit Marcus with a reality check.", "I hit Marcus with a guilt trip.", "I shoot Marcus with a wink.",
    "I hit Marcus with the silent treatment.", "I hit Marcus with a lawsuit.", "I hit Marcus with a riddle.",
    "I shoot Marcus's fake leg.", "I shoot Marcus's holographic face.", "I kill Marcus in the holo room.",
    "I kill Marcus in the next room.", "I punch Marcus in her face.", "I punch Elara in his face.",
    "I shoot Marcus in the fake leg.", "I punch Marcus in his holographic face.",
])
def test_outside_the_closed_grammar_no_one_is_struck(text):
    """A plan, a wish or a condition anywhere in the sentence, a preposition before the person or the Tome, an owner, a
    pronoun that could stand for more than one, a figurative or other object, a verb of no blow: no blow on a person or
    the Tome, and no act on them."""
    r = read(text, TARGETS, ["elara", "marcus"])
    assert not [c for c in r.closed if c.target is not None], r.closed
    assert r.act is None or r.act.target not in ("elara", "marcus") and (r.act.target != "tome" or r.act.kind is None)


def test_a_name_shared_with_an_obstacle_is_the_obstacle():
    """"the warden" with the warden turret there is the turret, whose name it is a word of, not Marcus."""
    targets = {**TARGETS, "warden_turret": ["warden turret", "warden", "turret"]}
    for text in ("I kick the warden hard.", "Marcus, I shoot the warden."):
        r = read(text, targets, ["elara", "marcus"])
        assert r.closed == () and r.act.target == "warden_turret", (text, r.act, r.closed)
    assert read("I kick the warden hard.", TARGETS, ["elara", "marcus"]).act.target == "marcus"


def test_the_narration_names_the_one_the_act_is_done_to(scenario, director_model):
    """'I attack Marcus. Elara, help me!' is told as a blow at Marcus, though Elara, spoken to, answers first."""
    e = staged(scenario, director_model, "EM")
    frame = e.turn("I attack Marcus. Elara, help me!")
    told = next(b.text for b in frame.blocks if b.role == "narration")
    assert e.last_turn.npc_name == scenario.npcs["marcus"].name
    assert scenario.npcs["marcus"].name in told and scenario.npcs["elara"].name not in told, told


def test_the_one_asked_for_help_is_not_the_one_attacked(scenario, director_model):
    """An ally spoken to while the player strikes at something else may help ("Elara, cover me while I shoot the
    drone."): only the one an act is done to counts as attacked."""
    e = staged(scenario, director_model, "EM")
    mark = len(e.checks.records)
    e.turn("Elara, cover me while I shoot the drone.")
    assert ("elara", "npc.help") in [(r.actor, r.action) for r in e.checks.records[mark:]]


class Perceives:
    """A model transducer: its reading of the words, whatever they are (``LocalTransducer`` without a model)."""

    def __init__(self, sensed):
        self.sensed = sensed

    def extract(self, text, targets=None, topics=None, people=None):
        return self.sensed


@pytest.mark.parametrize("text,scene,intent,target,act", [
    # in the closed grammar, a description the model resolved is the object
    ("I shoot the old soldier.", "EM", "fight", "marcus", "attack_npc@marcus"),
    ("I kick the stupid book on the floor.", "EM", "force", "tome", "strike_tome"),
    ("Tome, I cover you with my coat.", "EM", "sneak", "tome", "smother_tome"),
    ("I shoot at the thing in the air.", "EM+drone", "fight", "scavenger_drone", "engage@scavenger_drone"),
    # outside it, the reader's own reading: a preposition, an owner, a place, a description of someone else, a verb
    # that strikes no one, a plan
    ("I open up on the redhead.", "EM", "fight", "elara", "idle"),
    ("I open up on the cursed book.", "EM", "fight", "tome", "ask_tome"),
    ("I heave the thing up.", "EM", "force", "tome", "idle"),
    ("I smash Elara's chains.", "EM", "force", "elara", "engage@jammed_iris"),
    ("I pry open Marcus's locker.", "EM", "force", "marcus", "engage@jammed_iris"),
    ("I hide behind the Tome.", "EM", "sneak", "tome", "engage@jammed_iris"),
    ("I take cover behind the Tome.", "EM", "sneak", "tome", "engage@jammed_iris"),
    ("I shoot the old lady.", "EM", "fight", "marcus", "engage@jammed_iris"),
    ("I hurl insults at Marcus.", "EM", "force", "marcus", "chat_npc"),
    ("If he moves, I shoot him.", "EM", "fight", "marcus", "chat_npc"),
    ("I shoot the lamp above the old soldier.", "EM", "fight", "marcus", "engage@jammed_iris"),
    ("I hit the wall.", "EM", "fight", "tome", "engage@jammed_iris"),
    ("I cover the lamp.", "EM", "sneak", "tome", "engage@jammed_iris"),
    ("Marcus... if you lie to me again... I shoot you.", "EM", "fight", "marcus", "threaten_npc"),
    ("I kill Marcus in my head.", "EM", "fight", "marcus", "chat_npc"),
    ("I kick the warden hard.", "EM+turret", "force", "marcus", "engage@warden_turret"),
    ("The crate is next to the Tome. I smash it.", "EM", "force", "tome", "engage@jammed_iris"),
    # a strict-form blow stands whatever the model read
    ("I stab Marcus with my scalpel.", "EM", "fight", "marcus", "attack_npc@marcus"),
    ("I shoot Marcus.", "EM", "talk", "marcus", "attack_npc@marcus"),
    ("I shoot Marcus.", "EM", "search", "none", "attack_npc@marcus"),
    ("I smash the Tome.", "EM", "talk", "tome", "strike_tome"),
    ("I smother the Tome.", "EM", "talk", "tome", "smother_tome"),
    # an act at a person that strikes no one is words to them, never an act on the door
    ("I search Marcus.", "EM", "search", "marcus", "chat_npc"),
    ("I lift Elara.", "EM", "force", "elara", "chat_npc"),
    # ... so too when the model perceives the act aimed at nothing
    ("I lift Elara.", "EM", "force", "none", "chat_npc"),
    ("I search Marcus.", "EM", "search", "none", "chat_npc"),
    ("I climb over Marcus.", "EM", "advance", "none", "chat_npc"),
    ("I hurl insults at Marcus.", "EM", "fight", "none", "chat_npc"),
    ("Marcus, I'm warning you. I shoot.", "EM", "fight", "none", "threaten_npc"),
    ("Elara, I kick the door.", "EM", "force", "none", "engage@jammed_iris"),
    # the reader refines the model only with an explicit object in the scene
    ("I shoot the drone.", "EM+drone", "fight", "marcus", "engage@scavenger_drone"),
    ("I'm sorry. Marcus, I shoot you.", "EM", "fight", "elara", "attack_npc@marcus"),
    # the Tome is only ever named, never described; no pronoun but "you" said to it by name stands for it
    ("I kick the stupid thing on the floor.", "EM", "force", "tome", "engage@jammed_iris"),
    ("I smash the relic.", "EM", "force", "tome", "engage@jammed_iris"),
    ("I smash the log book.", "EM", "force", "tome", "engage@jammed_iris"),
    ("I burn a book.", "EM", "fight", "tome", "idle"),
    ("Tome, I'm sorry. I cover you with my coat.", "EM", "sneak", "tome", "ask_tome"),
    ("I see rust on the Tome. I smash it.", "EM", "force", "tome", "engage@jammed_iris"),
    ("Marcus has the Tome. I hit him.", "EM", "fight", "marcus", "chat_npc"),
    ("Marcus, I'm sorry. I shoot you.", "EM", "fight", "marcus", "threaten_npc"),
    # the act's kind comes from the sentence that names what it is done to, never from an earlier one
    ("I take cover. I shoot the Tome.", "EM", "fight", "tome", "strike_tome"),
    ("I duck and cover. I shoot the Tome.", "EM", "fight", "tome", "strike_tome"),
    ("I cover my face. I smash the Tome.", "EM", "force", "tome", "strike_tome"),
    ("I hide. I hit the Tome.", "EM", "fight", "tome", "strike_tome"),
    ("I hide. I smash the Tome.", "EM", "force", "tome", "strike_tome"),
    ("I take cover behind the crate. I shoot the Tome.", "EM", "fight", "tome", "strike_tome"),
    ("I take cover. I shoot the Tome.", "EM+drone", "fight", "tome", "strike_tome"),
    ("I hide. I smash Marcus.", "EM", "force", "marcus", "attack_npc@marcus"),
    ("I hit. I cover the Tome.", "EM", "sneak", "tome", "smother_tome"),
    # the round-seven model-path probes: a model's act on a person or the Tome stands only in the closed grammar of
    # the same kind, in the sentence that names them; else the reader's own, least consequential reading
    ("I cover my eyes. I smash the Tome.", "EM", "force", "tome", "strike_tome"),
    ("I hide quickly. I kick the Tome.", "EM", "force", "tome", "strike_tome"),
    ("I cover the lamp. I smash the thing.", "EM", "force", "tome", "engage@jammed_iris"),
    ("I smash the lamp. I cover the thing.", "EM", "sneak", "tome", "engage@jammed_iris"),
    ("I shoot. I hit Marcus.", "EM", "fight", "marcus", "attack_npc@marcus"),
    ("I hide. I punch Marcus.", "EM", "fight", "marcus", "attack_npc@marcus"),
    ("I hide. I punch Marcus.", "EM", "talk", "marcus", "threaten_npc"),
    ("I swing at the Tome.", "EM", "fight", "none", "engage@jammed_iris"),
    ("I throw my coat over the Tome.", "EM", "sneak", "none", "engage@jammed_iris"),
    ("I lift the Tome.", "EM", "force", "none", "idle"),
    ("I hug Marcus.", "EM", "force", "none", "chat_npc"),
    ("I grab Elara.", "EM", "force", "none", "chat_npc"),
    ("I drag Marcus behind me.", "EM", "force", "none", "chat_npc"),
    ("I carry Elara.", "EM", "force", "none", "chat_npc"),
    ("I sneak past Marcus.", "EM", "sneak", "none", "chat_npc"),
    ("I use the medkit on Marcus.", "EM", "use", "none", "chat_npc"),
    ("I use the medkit on Marcus.", "EM", "use", "marcus", "chat_npc"),
    ("I examine Elara's wound.", "EM", "search", "none", "engage@jammed_iris"),
    ("I punch Marcus's ghost.", "EM", "fight", "none", "engage@jammed_iris"),
    ("Marcus, I shoot the drone.", "EM+drone", "fight", "none", "engage@scavenger_drone"),
    ("Elara, hold this. I lift the crate.", "EM", "force", "none", "engage@jammed_iris"),
    ("I tell Marcus to lift the crate.", "EM", "force", "none", "chat_npc"),
    ("Marcus lifts the crate.", "EM", "force", "none", "chat_npc"),
    ("Elara, push the crate.", "EM", "force", "none", "chat_npc"),
    ("I shoot Marcus.", "EM", "fight", "elara", "attack_npc@marcus"),
    ("I shoot Marcus.", "EM+drone", "fight", "scavenger_drone", "attack_npc@marcus"),
    ("I shoot Marcus.", "EM", "flee", "none", "attack_npc@marcus"),
    ("I smother the Tome.", "EM", "fight", "tome", "smother_tome"),
    ("I smash the Tome.", "EM", "sneak", "tome", "strike_tome"),
    ("I stab Marcus with my scalpel.", "EM", "use", "none", "attack_npc@marcus"),
    ("I stab Marcus with my scalpel.", "EM", "use", "marcus", "attack_npc@marcus"),
    ("I hit the guard.", "EM+turret", "fight", "marcus", "attack_npc@marcus"),
    ("I smash the thing.", "EM+drone", "force", "tome", "engage@scavenger_drone"),
    ("I hit the book of matches.", "EM", "fight", "tome", "ask_tome"),
    ("I hit the man in the mirror.", "EM", "fight", "marcus", "chat_npc"),
    ("I shoot the old man hard.", "EM", "fight", "marcus", "chat_npc"),
    ("I shoot the old woman hard.", "EM", "fight", "marcus", "chat_npc"),
    ("I shoot the drone pilot.", "EM+drone", "fight", "marcus", "engage@scavenger_drone"),
    ("I shoot the turret's warden.", "EM+turret", "fight", "marcus", "engage@warden_turret"),
    ("I punch the stranger. Nope.", "EM", "fight", "marcus", "engage@jammed_iris"),
    ("I hide the old thing in my pack.", "EM", "sneak", "tome", "engage@jammed_iris"),
    ("I hide the old relic in my pack.", "EM", "sneak", "tome", "engage@jammed_iris"),
    ("I cover the lamp thing.", "EM", "sneak", "tome", "engage@jammed_iris"),
    ("I shoot the coward in the face.", "EM", "fight", "elara", "attack_npc@elara"),
    ("I shoot the coward in my face.", "EM", "fight", "elara", "engage@jammed_iris"),
    ("I shoot the kid and run away.", "EM", "fight", "elara", "attack_npc@elara"),
    ("I shoot the guard drone hard.", "EM+drone", "fight", "marcus", "engage@scavenger_drone"),
    ("I shoot the guard, drone.", "EM+drone", "fight", "marcus", "engage@scavenger_drone"),
    ("Elara and Marcus. I shoot you.", "EM", "fight", "marcus", "threaten_npc"),
    ("Elara and Marcus. I shoot you.", "EM", "talk", "elara", "threaten_npc"),
    ("I shield my eyes. I smash the Tome.", "EM", "force", "tome", "strike_tome"),
])
def test_a_models_perceived_target_stands_only_in_the_closed_grammar(scenario, director_model, text, scene, intent,
                                                                       target, act):
    """With a local model as the transducer, the target and intent it perceives stand for words, and for an act on a
    person or the Tome only when the act's sentence is in the closed grammar with them as its direct object, by name
    or by a description the model resolved ("the old soldier" is Marcus). Otherwise the reader's least consequential
    reading stands in its place."""
    from voidlit.senses.state import Intent, SensoryState

    e = staged(scenario, director_model, scene)
    e.transducer = Perceives(SensoryState(intent=Intent(intent), target=target, aggression=0.8))
    mark = len(e.checks.records)
    e.turn(text)
    t, acts = e.last_turn, [r.action.removeprefix("player.") for r in e.checks.records[mark:]
                            if r.action.startswith("player.")]
    hit = {"engage": t.obstacle.id if t.obstacle else None, "attack_npc": t.flee_npc}.get(acts[0])
    assert acts[0] + (f"@{hit}" if hit else "") == act


def test_a_heuristic_reading_never_passes_for_a_perception(scenario, director_model):
    """The heuristics' own target, a name spoken to, is not what an act with no object of its own is done to: "Elara,
    run! I hide." hides from the drone, as the local model's fallback reads it too."""
    from voidlit.voice.llama_local import LocalTransducer

    class Broken:
        def chat(self, *a, **kw):
            raise RuntimeError("no model")

    e = staged(scenario, director_model, "EM+drone")
    e.transducer = LocalTransducer(Broken())
    mark = len(e.checks.records)
    e.turn("Elara, run! I hide.")
    assert [r.action for r in e.checks.records[mark:] if r.action.startswith("player.")] == ["player.engage"]
    assert e.last_turn.obstacle.id == "scavenger_drone" and "elara" in e.state.npcs_present


@pytest.mark.parametrize("text,person", [("I kick Marcus.", "marcus"), ("I shove Elara.", "elara"),
                                         ("I shoot Marcus.", "marcus")])
def test_a_blow_at_a_person_is_told_as_one(scenario, director_model, text, person):
    """A force verb at a person is an attack, and is told as one at them, never as a shove at the station."""
    e = staged(scenario, director_model, "EM")
    frame = e.turn(text)
    told = next(b.text for b in frame.blocks if b.role == "narration")
    assert e.last_turn.flee_npc == person
    assert scenario.npcs[person].name in told and "station" not in told.split(".")[0], told


@pytest.mark.parametrize("text", ["I lift the Tome.", "I read the Tome.", "I climb the Tome.", "I use the Tome."])
def test_a_hand_on_the_tome_that_neither_strikes_nor_hides_it_is_told_as_that(scenario, director_model, text):
    """Anything else done to the Tome changes nothing, and is told as just that: never as a shove at the station or as
    the player holding still."""
    e = staged(scenario, director_model, "EM")
    frame = e.turn(text)
    told = next(b.text for b in frame.blocks if b.role == "narration")
    assert e.last_turn.variant == "tome_idle" and told.startswith("The Tome lets you"), told


class Keys(OfflineVoice):
    """The bank key each turn's outcome line is told by, with the keys that were tried for it."""

    def _pick(self, *keys, register=""):
        if keys and keys[0].startswith("action."):
            self.picked = next((k for k in keys if any(self.banks.get(k, []))), None), keys
        return super()._pick(*keys, register=register)


@pytest.mark.parametrize("clear", [False, True], ids=["obstacle", "no_obstacle"])
@pytest.mark.parametrize("target", ["none", "jammed_iris", "elara", "tome"])
@pytest.mark.parametrize("intent", [i.value for i in __import__("voidlit.senses.state", fromlist=["Intent"]).Intent])
def test_every_intent_at_every_target_has_its_own_line(scenario, director_model, intent, target, clear):
    """Whatever the player does to whatever it is done to, the outcome is told by a line written for it: a variant
    (a blow at a person, a hand on the Tome, ...) by its own line, never by a plain outcome line, and nothing by the
    catch-all "You hold still" (the player did not)."""
    from voidlit.senses.state import Intent, SensoryState

    voice = Keys(scenario.narration, seed=0)
    voice.picked = None
    e = Spy(scenario, director=Director(director_model, seed=0), seed=0, voice=voice,
            transducer=Perceives(SensoryState(intent=Intent(intent), target=target, aggression=0.2)))
    e.start()
    e.state.npcs_present = ["elara", "marcus"]
    if clear:
        e.state.obstacles.clear()
    e.turn("...")
    key, tried = voice.picked
    variants = [k for k in tried if k.split(".")[-1] not in ("fail", "mixed", "success", "critical", "none")]
    assert not variants or key in variants, (key, tried)
    assert key != "action.any.none", tried
