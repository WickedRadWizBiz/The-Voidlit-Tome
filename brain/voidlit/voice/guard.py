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
import unicodedata
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

# The words of a check's outcome, of dice, luck and numbers, and of the rules, in an NPC's own words ("Success,
# barely.", "Nat one.", "Lucky shot.", "Your number has come up.", "Yahtzee!", "You won.", "Plus one to grit.", "Level
# up.", "Double six again."): nobody in the game draws the player's eye to the dice, so a model's line that holds one,
# in any sense ("We die here.", "Check the door.", "We pass the lift."), gives way to the NPC's authored line. So does
# a number a die could show, alone or doubled ("Seven again.", "Double ones."), in any sense ("Bay seven."), but in a
# compound ("Twenty-one days.", "A one-way door."). (Fate and accidents are the station's to name.)
_DIE_FACES = (r"two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
              r"seventeen|eighteen|nineteen|twenty")
NPC_CHANCE = re.compile(r"\b(?:succe(?:ss|ed)\w*|fail\w*|check\w*|crits?|critical\w*|natural\w*|nat|rolls?|rolled|"
                        r"rolling|dice|die|dies|(?:un)?luck\w*|numbers?|sixes|yahtzee|jackpot|odds|chances?|random\w*|"
                        r"fluke\w*|gambl\w*|(?:mis)?fortun\w*|coins?|tails|probab\w*|scores?|stats?|modifiers?|"
                        r"bonus\w*|penalt\w*|botch\w*|fumbl\w*|boxcars|box cars|snake[- ]?eyes|outcomes?|hit ?points?|"
                        r"saving throws?|difficulty|won|wins?|winning|winners?|victor(?:y|ies|ious)|bingo|"
                        r"pass(?:es|ed)?|"
                        r"(?:dis)?advantage\w*|level(?:s|ed|led)? up|game over|(?:plus|minus) (?:one|\d+|"
                        rf"{_DIE_FACES})|grit|guile|insight|resolve|doubles?|ones|twos|threes|fours|fives|"
                        rf"(?<!-)(?:{_DIE_FACES})(?!-))\b", re.I)
# A number standing for itself, as a die shows one ("You got a twenty.", "Seventeen. That's what you got."): a number
# after "a" or "the", or with nothing after it in its clause. A count of things is not one ("It waited forty days.");
# "one" is too much the pronoun ("no one", "the one") to tell but after "a" or another number ("a one", "twenty one").
_NUMBER = (r"(?:\d+|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
           r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)")
BARE_NUMBER = re.compile(rf"\b(?:an?|the)\s+{_NUMBER}\b|\ba\s+one\b(?!-)|"
                         rf"\b{_NUMBER}(?:[-\s]+(?:{_NUMBER}|one))*\s*(?:[.,;:!?]|$)", re.I)
# What lets an NPC's line pass for another block on screen: a line break (the screen starts a new block on one), and
# the Tome named as the one speaking ("the Tome says ...", "The Tome murmurs: ...", "As the Tome would say, ...",
# "Quoth the Tome", "The Tome has spoken.").
LINE_BREAK = re.compile(r"[\n\r\v\f\x85\u2028\u2029]")
_LABEL_END = r"\s*(?::|-+\s|–|—|>+|»|\|)"
_TOME_NAME = r"(?:tome|book|monolith|narrator|narration)"      # the Tome, or the voice it narrates in
_SPEAKING = (r"(?:say|says|said|speak|speaks|spoke|spake|whisper|whispers|whispered|answer|answers|answered|reply|"
             r"replies|replied|write|writes|wrote|tell|tells|told|"
             r"murmurs|murmured|mutters|muttered|hisses|hissed|breathes|breathed|intones|intoned|rasps|rasped|"
             r"declares|declared|warns|warned|adds|added|continues|continued)")
TOME_SPEAKS = re.compile(r"\bquoth\b|\b(?:has|have|hath) spoken\b|"
                         rf"\b{_TOME_NAME}(?:['’]s)?\b(?:[\s,]+\w+){{0,2}}{_LABEL_END}|"
                         rf"\b{_TOME_NAME}(?:['’]s)?[\s,]+(?:\w+[\s,]+){{0,2}}{_SPEAKING}\b|"
                         rf"\b{_SPEAKING}\s+the\s+{_TOME_NAME}\b|(?:-+|–|—)\s*(?:the\s+)?{_TOME_NAME}\s*[.!]?\s*$|"
                         r"\b(?-i:TOME|BOOK|MONOLITH)\b", re.I)


def fold(text: str) -> str:
    """Text as it would look on screen, look-alikes undone: NFKC (fullwidth letters and colons to plain ones) and no
    invisible formatting characters (zero-width spaces and joiners, soft hyphens, direction overrides)."""
    return "".join(c for c in unicodedata.normalize("NFKC", text) if unicodedata.category(c) != "Cf")


# The closed shape of an NPC's line from a model (``plain_npc_line``): plain sentences of words in ASCII letters (an
# apostrophe in one, a hyphen joining two), each word followed by a space, a comma, a sentence's end, or an em dash
# before the next word. No colon, semicolon, pipe, angle bracket, slash, equals sign, asterisk, bracket, parenthesis or
# digit can stand in one, nor a dash or a quote mark that is not an em dash or an apostrophe.
_PLAIN_WORD = r"[A-Za-z]+(?:['’][A-Za-z]+)*['’]?(?:-[A-Za-z]+(?:['’][A-Za-z]+)*['’]?)*"
PLAIN_LINE = re.compile(rf"{_PLAIN_WORD}(?:(?:[,.!?]+ | | ?— ?){_PLAIN_WORD})*[.!?]*")
SHOUTED = re.compile(r"\b[A-Z]{2,}\b")          # a word in capitals, as the screen heads a block ("THE TOME")
# Whom a speaker's label could name, besides the people of the scene: the Tome by any name, the voice it narrates in,
# and the titles a speaker could go by.
TITLES = frozenset({"tome", "book", "monolith", "narrator", "narration", "voice", "player", "sir", "madam", "captain",
                    "sergeant", "doctor", "dr", "mr", "mrs", "ms", "miss", "lady", "lord", "commander", "warden",
                    "system", "master", "gm", "dm"})
_OPENING = re.compile(r"((?:[A-Za-z'’-]+ ){0,2}[A-Za-z'’-]+) ?[,.!?—]")
# A speaker's form: a name or title, then a verb of speech, at the start ("Elara says run.", "Elara Voss whispers,
# run.", "Marcus shrugs. Fine."); or anyone's verb of speech before an em dash ("Elara Voss whispers — run.").
_SPEECH = (rf"{_SPEAKING[:-1]}|shout|shouts|shouted|yell|yells|yelled|call|calls|called|cry|cries|cried|growl|growls|"
           r"growled|snap|snaps|snapped|ask|asks|asked|beg|begs|begged|sigh|sighs|sighed|laugh|laughs|laughed|shrug|"
           r"shrugs|shrugged|nod|nods|nodded|smile|smiles|smiled|grunt|grunts|grunted|mutter|murmur|hiss|hissed|"
           r"breathe|warn|add|continue|quote|quotes|quoted)")
_SPEAKER = re.compile(rf"((?:[A-Za-z'’-]+ ){{1,3}}){_SPEECH}\b", re.I)
_SAYS_DASH = re.compile(rf"\b{_SPEECH}\s*—", re.I)
# ... or the speaker named after the words ("Run, says Elara.", "Run, Elara says."); a stage direction at the start, a
# name or title and one verb of what they do ("Marcus winks. Run.", "Hale grins, run."); and a line that is only a
# check's outcome ("You lose.", "Direct hit.", "One.", "Lose a turn.")
_SAID_BY = re.compile(rf"[,.!?—]\s*(?:{_SPEECH}\s+((?:[A-Za-z'’-]+ ?){{1,3}})|((?:[A-Za-z'’-]+ ){{1,3}}){_SPEECH})"
                      r"[.!?]*$", re.I)
_DIRECTION = re.compile(r"((?:[A-Za-z'’-]+ ){1,3})[a-z]+s(?: (?:in|out|away|back|up|down|over|closer))?[.!,]")
_OUTCOME_LINE = re.compile(r"(?:(?:you|a|that's a|clean|direct|near) )?(?:defeat|hit|miss|missed|one|lose|lost|died|"
                           r"dead|lose a turn|skip a turn)[.!?]*", re.I)


def plain_npc_line(line: str, names: Iterable[str] = ()) -> bool:
    """Whether an NPC's line from a model, folded onto one line, is in the closed shape a model's line may take on
    screen (``PLAIN_LINE``): no word in capitals, no word of a check's outcome, of dice, luck or the rules
    (``NPC_CHANCE``), no number a die could show (``BARE_NUMBER``), the Tome never named as the one speaking, no
    speaker's form (a name or title then a verb of speech at the start: "Elara says run.", "Marcus shrugs. Fine."; a
    verb of speech before an em dash: "Elara Voss whispers — run."; or after the words: "Run, says Elara."), no stage
    direction ("Marcus winks. Run."), no line that is only a check's outcome ("You lose.", "Direct hit.", "One."),
    and no opening a speaker's label could be: a name
    (words of ``names``) or a title alone, or a name in full ("Elara Voss"), then any punctuation ("Elara, run.",
    "Marcus — down!"; not "Stay behind Elara.")."""
    if not PLAIN_LINE.fullmatch(line) or SHOUTED.search(line) or NPC_CHANCE.search(line) or \
            BARE_NUMBER.search(line) or TOME_SPEAKS.search(line) or _SAYS_DASH.search(line):
        return False
    if _OUTCOME_LINE.fullmatch(line):
        return False
    known = TITLES | {w for name in names for w in name.lower().split()}
    for who in (_SPEAKER.match(line), _DIRECTION.match(line), _SAID_BY.search(line)):
        if who and ({w.lower() for w in (who.group(1) or who.group(2)).split()} - {"the"}) <= known:
            return False
    opening = _OPENING.match(line)
    if opening is None:
        return True
    words = opening.group(1).split()
    named = {w.lower() for w in words} - {"the"}
    return not (named and named <= known or len(words) > 1 and all(w[0].isupper() for w in words))

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

