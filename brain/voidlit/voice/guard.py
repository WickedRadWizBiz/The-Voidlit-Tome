"""Output guard for the local LLM voice: the LLM decorates; only the rules assert.

A Tome line on the LLM path is its speech acts, spliced in verbatim, plus at most two short sentences of *framing*
the model wrote, each a command, a form of address or a question. A narration is the *core* sentences the rules
decided this turn, verbatim, plus at most two short sentences of *colour* about the place. This module checks what
the model wrote around the verbatim lines: anything outside the allowed shapes and words is rejected and the offline
voice speaks instead. The offline banks are the reference and never trip it.

As far as the player knows there are no dice: any line that names a die, a roll, a check or a modifier, or (for
narration) any digit at all, is rejected too. The lexicons here are also the truth verifier's (``tome/ledger.py``).
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
# the words of dice, in any sense: what a model writes may not use them at all ("was it a good roll?")
DICE_WORDS = re.compile(r"\b(?:dice|die|dies|rolls?|rolled|rolling|checks?|scores?|stats?|modifiers?|bonus\w*|"
                        r"penalt\w*)\b", re.I)
# numbers in words: what a model writes may not count anything either
NUMBER_WORDS = re.compile(r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\w+teen|twenty|"
                          r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundreds?|thousands?|millions?|numbers?|"
                          r"numerals?|digits?|count|counts|counted|counting|tally|tallies)\b", re.I)


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
# words that reframe what stands beside them (a dream, a lie, a doubt, a reversal); free text may not use them
HEDGE = re.compile(r"\b(?:dream\w*|seem\w*|appear\w*|imagin\w*|illusion\w*|pretend\w*|lie|lies|lied|lying|liars?|"
                   r"false\w*|untrue|unreal|fake\w*|trick\w*|wrong|mistak\w*|maybe|perhaps|possibl\w*|probabl\w*|"
                   r"might|almost|nearly|barely|hardly|scarcely|apparent\w*|suppos\w*|alleged\w*|claim\w*|rumou?r\w*|"
                   r"says?|said|think\w*|thought|believ\w*|guess\w*|true|truly|truth\w*|honest\w*|real|really|"
                   r"sure\w*|certain\w*|doubt\w*|trust\w*|know\w*|knew|as if|as though|or so|unless|until|except|"
                   r"instead|yet|still|again|undo\w*|undone|revers\w*|anyway|however|but|though|although|whereas|"
                   r"only|once|if|whether|or|should|ought|nonsense|rubbish|absurd\w*|ridiculous|ludicrous|"
                   r"doubtful|unlikely|impossibl\w*)\b", re.I)
# what an action came to: the rules decide it and the core sentences say it; a model's own words never do
OUTCOME = re.compile(r"\b(?:fail\w*|miss(?:ed|es)?|lost|lose|loses|losing|win|wins|won|winning|succe\w*|works?|"
                     r"worked|broke|broken|break\w*|crack\w*|wound\w*|bleed\w*|blood\w*|bled|hit|hits|struck|strik\w*|"
                     r"land(?:s|ed)|skid\w*|dead|death|kill\w*|harm\w*|safe\w*|unharmed|surviv\w*|open\w*|shut\w*|"
                     r"clos(?:ed|es|ing)|clear\w*|gone|beaten|defeat\w*|overcome|decid\w*|decision\w*|holds?|held|"
                     r"giv(?:e|es|en|ing)|gave|yield\w*|victor\w*|triumph\w*|reward\w*|punish\w*|prize\w*|"
                     r"fight\w*|fought|battle\w*|wreck\w*|lifeless|corpse\w*|bod(?:y|ies)|slain|slump\w*|fallen|"
                     r"collaps\w*|ends?|ended|ending|finish\w*)\b|\b(?:is|are|was|it's) (?:over|done)\b", re.I)
# the places and choices a scene offers are the rules' to name, not colour's
OPTIONS = re.compile(r"\b(?:ways?|path\w*|road\w*|route\w*|exit\w*|doors?|doorway\w*|hatch\w*|airlock\w*|escap\w*|"
                     r"choice\w*|choos\w*|option\w*|alternativ\w*|hope\w*|rest|rests|resting|respite|shelter\w*|"
                     r"refuge|sanctuary)\b", re.I)
# colour is the place as it is now: it tells no past and foretells nothing
TENSE = re.compile(r"\b(?:was|wasn't|were|had|has|have|did|does|been|will|won't|would|shall|could|can|may)\b", re.I)
SECOND_PERSON = re.compile(r"\b(?:you|your|yours|yourself|yourselves|you're|you'll|you've|you'd|thee|thy|thou)\b",
                           re.I)
FIRST_PERSON = re.compile(r"\b(?:I|I'm|I'll|I've|I'd|me|my|mine|myself|we|we're|we'll|we've|we'd|us|our|ours|"
                          r"ourselves)\b", re.I)
SPEAKER = re.compile(r"\b(?:I|I'm|I'll|I've|I'd|we|we're|we'll|we've|we'd)\b", re.I)
# the Tome itself, as narration can name it
TOME_REF = re.compile(r"\b(?:tome\w*|book\w*|glass\w*|monolith\w*|slab\w*|pages?|words?|letters?|ink)\b|"
                      r"\bits (?:light|glow|surface)\b|\bthe light you carry\b", re.I)

# ------------------------------------------------------------------------------------------------ text plumbing
_TYPO = str.maketrans({"’": "'", "‘": "'", "‛": "'", "`": "'", "“": '"', "”": '"', "„": '"', "–": "-", "—": "-",
                       "…": "..."})
_SPLIT = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')]))\s+")
_WORDS = re.compile(r"[a-z][a-z']*", re.I)


def normalize(text: str) -> str:
    """Typographic apostrophes, quotes and dashes made plain; whitespace collapsed."""
    return " ".join(text.translate(_TYPO).split())


def sentences(text: str) -> list[str]:
    return [s for s in _SPLIT.split(normalize(text)) if s.strip()]


def _bare(sentence: str) -> str:
    return normalize(sentence).lower().rstrip(" .!?\"')").strip()


def _verbatim(sentence: str) -> re.Pattern:
    """A core sentence standing whole in a text: at its start or after a sentence's end (or one already taken
    out, marked "|"), and closed by its own kind of ending (a statement never turns into a question)."""
    end = r"\?+" if normalize(sentence).rstrip("\"')").endswith("?") else r"[.!]+"
    return re.compile(rf"(?:^\s*|(?<=[.!?\"')|])\s+){re.escape(_bare(sentence))}(?:{end}[\"')]*(?=\s|$)|$)")


def missing_core(text: str, core: Iterable[str]) -> list[str]:
    """The core sentences ``text`` does not say verbatim, each as a whole sentence (case, whitespace, typographic
    quotes and the final punctuation aside). Writing one inside a longer sentence does not count."""
    low = normalize(text).lower()
    return [c for c in core if _bare(c) and not _verbatim(c).search(low)]


def strip_core(text: str, core: Iterable[str]) -> str:
    """What is left of ``text`` once every verbatim core sentence is taken out: the words the model added."""
    low = normalize(text).lower()
    for c in sorted({c for c in core if _bare(c)}, key=len, reverse=True):
        low = _verbatim(c).sub(" | ", low, count=1)
    return " ".join(low.replace("|", " ").split())


_STOP = frozenset("the a an and or but of to in on at by for with from into onto is are was were be been it its "
                  "this that "
                  "these those there here as so not no you your i my me we our they them their he she his her what "
                  "while only every one".split())
_CONTENT = re.compile(r"[a-z][a-z']+")


def content_words(text: str, min_len: int = 4) -> set[str]:
    """The words that carry a text's meaning (possessives reduced to their noun: "reactor's" -> "reactor")."""
    words = {re.sub(r"'s$", "", w.strip("'")) for w in _CONTENT.findall(normalize(text).lower())}
    return {w for w in words if w not in _STOP and len(w) >= min_len}


def covers(required: str, text: str, threshold: float = 0.5) -> bool:
    """Whether ``text`` is about ``required``: at least ``threshold`` of its content words appear. A licence test
    (a necessity claim must be about the fact that licenses it), never a test that something was said."""
    words = {w.strip("'") for w in _CONTENT.findall(normalize(required).lower()) if w not in _STOP and len(w) > 2}
    if not words:
        return True
    have = {w.strip("'") for w in _CONTENT.findall(normalize(text).lower())}
    return len(words & have) / len(words) >= threshold


def _avoided(low: str, avoid: Iterable[str]) -> bool:
    """Whether a lowercased text uses any avoided word or phrase (a simple plural counts)."""
    return any(re.search(rf"\b{re.escape(w.lower())}(?:s|es)?\b", low) for w in avoid if w)


def _lexical(text: str) -> list[str]:
    """Rules any text the model wrote breaks, whatever its shape."""
    checks = (("number", _DIGIT), ("number", NUMBER_WORDS), ("mechanics", _MECHANICS), ("mechanics", _DC),
              ("mechanics", DICE_WORDS), ("outcome", OUTCOME), ("chance", CHANCE),
              ("affect", AFFECT), ("affect", AFFECT_SELF), ("opinion", OPINION_OF_PLAYER), ("opinion", JUDGEMENT),
              ("necessity", NO_ALTERNATIVE), ("negation", NEGATION), ("hedge", HEDGE))
    return [rule for rule, pat in checks if pat.search(text)]


# ------------------------------------------------------------------------------------------------ framing (speech)
# The first word of a framing command or address. Everything after it is at most a form of address.
FRAME_OPENERS = frozenset(
    "listen come closer mind go keep feed find bring oh walk quiet wait careful slowly hush look hear stay step "
    "tread follow heed watch move hurry steady breathe rest take carry ask be now ah so easy gently softly friend "
    "bearer thief little stop run turn climb press lean".split())
ADDRESSES = ("friend", "bearer", "little light", "thief")
# a question that takes something for granted: why it happened, or that it did
PRESUPPOSING = re.compile(r"\b(?:why|how come|did|didn't|was|wasn't|were|weren't|had|hadn't|has|hasn't|have|haven't|"
                          r"\w+ed)\b", re.I)
FRAMING_WORDS = 20
QUESTION_WORDS = 8


def framing_problems(text: str, avoid: Iterable[str] = (), addresses: Iterable[str] = ADDRESSES) -> list[str]:
    """Why a model-written framing cannot stand beside the Tome's verbatim speech acts (empty if it can).

    At most two finished sentences and twenty words; each a short question in the present (no "why", nothing that
    happened), or commands and forms of address only (every clause a form of address or opening with a whitelisted
    word); no number, mechanics, outcome, chance, feeling or need, opinion of the player, necessity, negation or
    hedge; no speaker ("I", "we"); and none of ``avoid`` (the scene's topics, people, secrets and obstacles). A form
    of address is never itself an avoided word ("little light")."""
    text = normalize(text)
    parts = sentences(text)
    out = []
    if not parts or len(parts) > 2 or len(_WORDS.findall(text)) > FRAMING_WORDS:
        out.append("framing_length")
    if not re.search(r"[.!?][\"')]*$", text):
        out.append("framing_shape")                  # unfinished, it would run on into the Tome's own words
    names = sorted((a.lower() for a in addresses if a), key=len, reverse=True)
    for sent in parts:
        low = sent.lower()
        bare = low                                  # forms of address stand in as one neutral word
        for a in names:
            bare = re.sub(rf"\b{re.escape(a)}\b", "\x00", bare)
        if low.rstrip("\"' ").endswith("?"):
            if not 0 < len(_WORDS.findall(bare)) <= QUESTION_WORDS or PRESUPPOSING.search(bare):
                out.append("framing_shape")
        else:
            clauses = [_WORDS.findall(c) for c in re.split(r"[,;:\-]+", bare) if c.strip(" .!?\"'")]
            if not clauses or any(c and c[0] not in FRAME_OPENERS for c in clauses):
                out.append("framing_shape")
        out += _lexical(low)
        if SPEAKER.search(bare):
            out.append("speaker")
        if _avoided(bare, avoid):
            out.append("avoided_word")
    return list(dict.fromkeys(out))


# ------------------------------------------------------------------------------------------------ colour (narration)
COLOUR_SENTENCES = 2
COLOUR_WORDS = 30


def colour_problems(text: str, avoid: Iterable[str] = ()) -> list[str]:
    """Why what a model added around the core sentences is not pure colour (empty if it is).

    At most two sentences and thirty words about the place as it is now (no past, no future); no "you" or "I";
    nothing about the Tome; no number, mechanics, outcome, chance, feeling, opinion, necessity, negation or hedge;
    none of the places and choices the scene offers; and none of ``avoid`` (the scene's topics, people, secrets and
    obstacles, and this turn's core words, so colour can never speak to what the rules decided)."""
    text = normalize(text)
    low = text.lower()
    parts = [s for s in sentences(text) if _WORDS.search(s)]
    out = []
    if len(parts) > COLOUR_SENTENCES or len(_WORDS.findall(text)) > COLOUR_WORDS \
            or any(len(_WORDS.findall(s)) < 3 for s in parts):  # a bare "Ha." or "Hardly." comments on the rest
        out.append("colour_length")
    for rule, pat in (("second_person", SECOND_PERSON), ("first_person", FIRST_PERSON), ("tome", TOME_REF),
                      ("necessity", OPTIONS), ("tense", TENSE)):
        if pat.search(text):
            out.append(rule)
    out += _lexical(text)
    if _avoided(low, avoid):
        out.append("avoided_word")
    return list(dict.fromkeys(out))
