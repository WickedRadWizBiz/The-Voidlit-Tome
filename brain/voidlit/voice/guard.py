"""The words the rules hold every line to: the mechanics check, the truth verifier's lexicons, and the checks on an
NPC's model-written words.

The narration and the Tome's own words are authored (the offline voice); a local model writes only the NPCs' own
dialogue (voice/llama_local.py). As far as the player knows there are no dice: any line that names a die, a roll, a
check or a modifier, or (outside an NPC's own words) any digit at all, leaks a mechanic. A model's NPC line also may
not talk of chance or pass for another speaker. The lexicons and the text plumbing here are the truth verifier's
(``tome/ledger.py``), which audits every authored line.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_MECHANICS = re.compile(
    r"\b(dice|die rolls?|d4|d6|d8|d10|d12|d20|d100|modifiers?|pips?|hit points?|saving throws?|skill checks?|"
    r"ability checks?|stat checks?|natural (?:one|twenty|1|20)|nat ?(?:1|20)|you rolled|rolled an?|"
    r"roll(?:ed|s)? (?:a|an|the) (?:\d+|one|two|three|four|five|six|twenty)|difficulty class|game mechanics?|"
    r"(?:your|the|that|this|my) (?:rolls?|checks?|stats?|scores?))\b",
    re.IGNORECASE)
_DC = re.compile(r"\bDC\s*\d*\b")
_DIGIT = re.compile(r"\d")
# the words of dice, in any sense ("was it a good roll?"); "die" only as a thing ("the die"), never "or die"
DICE_WORDS = re.compile(r"\b(?:dice|(?:a|an|the|one|that|this|each|every|your|my|our|their|his|her|its) die|rolls?|"
                        r"rolled|rolling|checks?|scores?|stats?|modifiers?|bonus\w*|penalt\w*)\b", re.I)


def leaks_mechanics(text: str, allow_digits: bool = False) -> bool:
    if _MECHANICS.search(text) or _DC.search(text):
        return True
    return not allow_digits and bool(_DIGIT.search(text))


# ------------------------------------------------------------------------------------------------ lexicons
# Chance at all: dice decide, so "nothing is random" is a falsehood, and the Tome says nothing about chance.
_CHANCE = (r"random\w*|(?:un)?luck\w*|chance\w*|odds|fate\w*|fortun\w*|misfortun\w*|coinciden\w*|accident\w*|"
           r"happenstance|fluke\w*|gambl\w*|destin(?:y|ies|ed)|meant to be|dice|predetermin\w*|preordain\w*|"
           r"foreordain\w*|ordained|foretold|inevitab\w*|written in the stars")
CHANCE = re.compile(rf"\b(?:{_CHANCE})\b", re.I)
_NEG = r"no|not|n't|nothing|none|never|nobody|no one|without"
# a denial of chance: a negation near a chance word, either way round
DICE_DENIAL = re.compile(rf"(?:\b(?:{_NEG})\b|n't\b)[^.!?]{{0,40}}?\b(?:{_CHANCE})\b|"
                         rf"\b(?:{_CHANCE})\b[^.!?]{{0,40}}?\b(?:{_NEG}|no part|no hand|no say|no place)\b", re.I)

# Talk of dice, rolls or luck in an NPC's own words ("Was it a good roll?", "Lucky shot."), which the mechanics check
# lets through: nobody in the game draws the player's eye to the dice. (Fate and accidents are the station's to name.)
NPC_CHANCE = re.compile(r"\b(?:dice|(?:a|an|the|one|that|this|each|every|your|my|our|their|his|her|its) die|rolls?|"
                        r"rolled|rolling|(?:un)?luck\w*|odds|chances?|random\w*|fluke\w*|gambl\w*|(?:mis)?fortun\w*)\b",
                        re.I)
# What lets an NPC's line pass for another block on screen: a line break (the screen starts a new block on one), a
# speaker's label the way the screen heads a block ("THE TOME:", "Elara Voss:"), as a name in capitals anywhere or as
# one to three capitalised words opening the line or one of its sentences, and the Tome named as the one speaking
# ("the Tome says ...", "Book: ...").
LINE_BREAK = re.compile(r"[\n\r\v\f\x85\u2028\u2029]")
SPEAKER_LABEL = re.compile(r"\b[A-Z][A-Z'’-]+(?:\s+[A-Z][A-Z'’-]+){0,2}\s*:|"
                           r"(?:^|[.!?…]\s+)(?:[A-Z][\w'’-]*\s+){0,2}[A-Z][\w'’-]*\s*:")
_TOME_NAME = r"(?:tome|book|monolith)"
_SPEAKING = r"(?:says|said|speaks|spoke|whispers|whispered|answers|answered|replies|replied|writes|wrote|tells|told)"
TOME_SPEAKS = re.compile(rf"\b{_TOME_NAME}\s*:|\b{_TOME_NAME}\s+{_SPEAKING}\b|\b{_SPEAKING}\s+the\s+{_TOME_NAME}\b",
                         re.I)


def poses_as_another(text: str) -> bool:
    """Whether an NPC's line, as it would be shown, could pass for another speaker's block or the narration."""
    return bool(LINE_BREAK.search(text) or SPEAKER_LABEL.search(text) or TOME_SPEAKS.search(text))


# Emotions (the OCC types and common labels) and needs. The Tome's own words carry none of them.
_AFFECT = (r"joy\w*|distress\w*|hope(?:ful|s|d)?|hoping|fear\w*|afraid|scared|frighten\w*|fright|terrif\w*|terror|"
           r"dread\w*|panic\w*|anxi\w*|worr\w*|nervous|satisf\w*|relie(?:f|ved)|disappoint\w*|resent\w*|gloat\w*|"
           r"pity|pitie[sd]|proud|pride|sham(?:e|ed)|ashamed|admir(?:e|es|ed|ing|ation)|reproach\w*|gratif\w*|"
           r"remorse\w*|grat(?:eful|itude)|ang(?:er|ry|rier|rily)|furious|fury|rage|enrag\w*|love[sd]?|loving|"
           r"hat(?:e|es|ed|red|ing)|happ(?:y|ier|ily|iness)|sad(?:ly|ness|der)?|lonel\w*|griev\w*|grief|sorrow\w*|"
           r"mourn\w*|miser\w*|despair\w*|desperat\w*|jealous\w*|envious|envie[sd]|envy|delight\w*|disgust\w*|"
           r"contempt\w*|scorn\w*|disdain\w*|fond\w*|glad\w*|upset|bitter\w*|mood\w*|feeling\w*|feels?|felt|"
           r"emotion\w*|excit\w*|thrill\w*|eager\w*|calm\w*|content(?:ed|ment)|pleased|hurt(?:s|ing)?|"
           r"pain(?:s|ful)?|suffer\w*|ache[sd]?|aching|yearn\w*|longing|miss(?:es|ed|ing)?|weary|wearied|tired|"
           r"exhaust(?:ed|ion)|weak(?:ness|ened)?|starv\w*|hunger\w*|hungr\w*|famish\w*|crav\w*|thirst\w*|needy|"
           r"greed\w*")
AFFECT = re.compile(rf"\b(?:{_AFFECT})\b", re.I)
# first-person reports, negated ones included ("I'm not afraid"), and the verbs of feeling and need
AFFECT_SELF = re.compile(
    rf"\b(?:I|I'm|I am|I was|I've been|I feel|I felt|we|we're|we are)(?:\s+(?:not|never|so|very|really|quite|too|"
    rf"just|almost|a little|a bit|rather|truly|always|still|getting|growing|becoming))*\s+(?:{_AFFECT})\b|"
    r"\bI (?:do not |don't |did not |didn't |never |no longer |cannot |can't )?(?:fear|dread|crave|hate|love|"
    r"despise|adore|loathe|trust|distrust|pity|resent|envy|need|hunger|thirst|ache|yearn|long|miss|suffer|starve|"
    r"mourn|grieve|like|enjoy|relish|regret|worry|care)\b", re.I)
_JUDGED = (r"fool\w*|idiot\w*|wretch\w*|worthless|pathetic|clever|brave|bold|weak\w*|coward\w*|liar|traitor|"
           r"useless|stupid|good|kind|smart|wise|strong|special|disappointment|disgrace|failure|nothing|mine|thief|"
           r"pitiful|doomed|hopeless|helpless|naive|reckless|cruel|selfish|greedy|ungrateful|faithful|loyal|"
           r"precious|dear|sweet|lovely|beautiful|ugly|vermin|worm|insect|filth\w*|scum|trash|pest|parasite|"
           r"child|amusing|boring|interesting|clumsy|slow|soft")
OPINION_OF_PLAYER = re.compile(
    rf"\byou(?:'re| are| were|'ve been| have been| seem| look| sound| remain)\s+(?:(?:a|an|such|so|very|too|quite|"
    rf"nothing but|no better than|just|only|truly|really|my|little|poor|mere)\s+)*(?:{_JUDGED})\b|"
    r"\byou (?:disgust|sicken|delight|please|disappoint|amuse|bore|impress|tire|anger|frighten|scare|interest|"
    r"intrigue|repel|revolt|charm|fascinate|worry|trouble|annoy|irritate|offend|insult)\s+me\b|"
    r"\b(?:you|poor|little|stupid|foolish|silly|brave|clever|sweet|dear|wretched|pathetic|miserable)\s+"
    r"(?:fool|idiot|wretch|thing|creature|child|coward|worm|insect|vermin|scum|liar|traitor)\b|"
    rf"\bwhat an? (?:{_JUDGED})\b|"
    r"\bhow (?:brave|clever|foolish|pathetic|stupid|sweet|kind|bold|weak|sad) (?:of )?you\b|"
    r"\bI (?:do not |don't |did not |didn't |never |cannot |can't |could not |couldn't )?(?:like|love|hate|trust|"
    r"distrust|pity|admire|despise|adore|loathe|respect|resent|envy|scorn|forgive|blame|mock|need|fear|prefer|"
    r"enjoy|tolerate|believe|doubt)s? you\b|\bI (?:think|find|consider|deem|judge|count|call|believe|know) you\b",
    re.I)
# judging words and mockery: free text may not use them about anyone (the player is the only one to judge)
JUDGEMENT = re.compile(r"\b(?:fool\w*|idiot\w*|wretch\w*|worthless|pathetic|coward\w*|liars?|traitor\w*|useless|"
                       r"stupid|pitiful|naive|reckless|vermin|worms?|insects?|filth\w*|scum|trash|pests?|parasit\w*|"
                       r"clever|brave|wise|precious|lovely|beautiful|ugly|doomed|hopeless|helpless|weakling\w*|"
                       r"laugh\w*|mock\w*|sneer\w*|jeer\w*|scoff\w*|smirk\w*)\b", re.I)
# what an opinion of the player is made of, when narration pins one on the Tome
OPINION_WORDS = re.compile(r"\b(?:despis\w*|loath\w*|pit(?:y|ies|ied)|scorn\w*|disdain\w*|mock\w*|sneer\w*|"
                           r"distrust\w*|trust\w*|admir(?:e|es|ed)|resent\w*|approv\w*|disapprov\w*|judg\w*|"
                           r"hat(?:e|es|ed|red)|lov(?:e|es|ed)|fond\w*|disgust\w*|contempt\w*|amus\w*)\b", re.I)

# Necessity: "no other way", "your only hope", "you must", "or die". Allowed literally only when a fact licenses it.
_WAY = (r"(?:way|ways|path|paths|road|roads|route|routes|choice|choices|option|options|alternative|alternatives|"
        r"hope|chance|door|doors|exit|exits|answer|light|escape)")
NO_ALTERNATIVE = re.compile(
    rf"\b(?:no other {_WAY}|no {_WAY} (?:but|except|save|other than|besides)|(?:the|your|my|our|their|his|her|its) "
    rf"(?:one and )?only {_WAY}|only (?:one|a single) {_WAY}|(?:one|single|sole|last) {_WAY} (?:left|remaining)|"
    rf"(?:the|your|my|our) (?:one|sole) {_WAY}|{_WAY} (?:left|remaining) (?:is|are)|no (?:choice|choices|"
    rf"alternative|alternatives|option|options|escape|way around|way back|going back|turning back|way through|way on)|"
    rf"nothing else for it|nothing else (?:to do|left|will do)|(?:there is|there's|there are|there remains) no "
    rf"(?:other )?{_WAY}|(?:you|we|I)(?:'ll| will| would)? (?:must|have to|need to|have got to|got to)|"
    rf"(?:you|we|I)'ve got to|(?:or|else) (?:you(?:'ll| will)? )?(?:die|perish|rot|be lost)|no more "
    rf"(?:choices|options|ways))\b", re.I)

NEGATION = re.compile(r"\b(?:not|no|never|nothing|none|nobody|no one|nowhere|neither|nor|without|cannot|nay|naught|"
                      r"nought)\b|n't\b", re.I)

# ------------------------------------------------------------------------------------------------ text plumbing
_TYPO = str.maketrans({"’": "'", "‘": "'", "‛": "'", "`": "'", "“": '"', "”": '"', "„": '"', "–": "-", "—": "-",
                       "…": "..."})
_SPLIT = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')]))\s+")


def normalize(text: str) -> str:
    """Typographic apostrophes, quotes and dashes made plain; whitespace collapsed."""
    return " ".join(text.translate(_TYPO).split())


def sentences(text: str) -> list[str]:
    return [s for s in _SPLIT.split(normalize(text)) if s.strip()]


def _bare(sentence: str) -> str:
    return normalize(sentence).lower().rstrip(" .!?\"')").strip()


def _verbatim(sentence: str) -> re.Pattern:
    """A verbatim line standing whole in a text: at its start or after a sentence's end (or one already taken
    out, marked "|"), and closed by its own kind of ending (a statement never turns into a question)."""
    end = r"\?+" if normalize(sentence).rstrip("\"')").endswith("?") else r"[.!]+"
    return re.compile(rf"(?:^\s*|(?<=[.!?\"')|])\s+){re.escape(_bare(sentence))}(?:{end}[\"')]*(?=\s|$)|$)")


def strip_core(text: str, core: Iterable[str]) -> str:
    """What is left of ``text`` once every verbatim line is taken out: the words nothing vouches for."""
    low = normalize(text).lower()
    for c in sorted({c for c in core if _bare(c)}, key=len, reverse=True):
        low = _verbatim(c).sub(" | ", low, count=1)
    return " ".join(low.replace("|", " ").split())


_STOP = frozenset("the a an and or but of to in on at by for with from into onto is are was were be been it its "
                  "this that "
                  "these those there here as so not no you your i my me we our they them their he she his her what "
                  "while only every one".split())
_CONTENT = re.compile(r"[a-z][a-z']+")


def covers(required: str, text: str, threshold: float = 0.5) -> bool:
    """Whether ``text`` is about ``required``: at least ``threshold`` of its content words appear. A licence test
    (a necessity claim must be about the fact that licenses it), never a test that something was said."""
    words = {w.strip("'") for w in _CONTENT.findall(normalize(required).lower()) if w not in _STOP and len(w) > 2}
    if not words:
        return True
    have = {w.strip("'") for w in _CONTENT.findall(normalize(text).lower())}
    return len(words & have) / len(words) >= threshold

