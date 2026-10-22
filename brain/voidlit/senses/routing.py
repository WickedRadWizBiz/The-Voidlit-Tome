"""Whom and what one line of player input is aimed at, read one sentence at a time.

The rules are structural. Each sentence is read on its own (who it is spoken to, what the player does in it and to
what, whether it asks something) and the readings are then combined:

* A **vocative** directs a sentence's words, its talk and its questions, never an act. It is a name at the head of the
  sentence or of a quotation in it, after an opener, before punctuation or a question word ("Hey Tome, ...", "Marcus -
  ...", "Tome what ...?", 'I shout "Tome, ..."'), last after a comma ("..., Elara?"), set off by commas ("I smash
  you, Tome, and run", "Drop it, Marcus, or I shoot"), or the one the player speaks to in the third person ("I ask the
  Tome who ..."). A sentence that is only a name ("Tome?", "Marcus.") asks nothing itself, and is spoken to along with
  the sentence next to it that names no one of its own. A sentence with no vocative is heard by whoever the line last
  spoke to.
* A **consequential act**, one that can drive a person from the scene or cost the Tome a light (a blow at a person, a
  blow at the Tome, a hand over its glow), happens only in a closed grammar (``_Reader._closed``): its sentence is, past
  whom it speaks to, ("And", "Without a word," or an adverb of its list at most before) "I" + one verb of a short list
  (``STRIKES``; on the Tome also ``SMOTHERS``), an adverb of its list at most between them ("I calmly shoot"), + the
  person or the Tome as the direct object + a trailing adverbial: what it is struck with, any noun phrase but a person,
  the Tome or a figure of speech ("with my slug rifle", not "with kindness" or "with a guilt trip"), or one of a closed
  list ("hard", "once", "three times", "point-blank", "down", "in the thigh" on their own body, never "in my head",
  "into the wall", "under my boot", "in my pack" or "up" after a smother only, "and run" or another short escape, never
  "and wake up screaming"). The object is named in that very sentence: a person's name, with "the" at most before it
  ("the warden", not "the hologram Marcus"); the Tome's, "the", "this" or "my" + "Tome", "book" or "monolith", or
  "Tome" alone, with at most words of a closed list between ("the cursed old Tome"; never "a book", "another Tome",
  "the log book" or "the hologram Tome", which are not the Tome at all); "you" (or "your") only for the one that very
  sentence speaks to by name, when it names no one else and the line speaks to no one jointly ("Marcus, I stab you.";
  never "Marcus and Elara, I shoot you.", "Tome, Marcus, I smash you." or "You lied, Tome. I smash you."); or their
  own body or glow ("Marcus's face", "Marcus' jaw", "the Tome's light"). No other pronoun ever stands for them ("it",
  "its", "him", "her", "them", "this", "that"): an act on one is an act on an obstacle or nothing, as on any object
  not in the scene. Any plan, wish, condition, promise, negation, retraction or fragment that takes it back anywhere in
  the line (would, could, should, need to, about to, will, 'll, swear, maybe, if, when, unless, until, next time, the
  second, "..., or I ...", not, "...", "Nope.", "Scratch that.", "Says Elara.", "Is that okay?", "Just kidding.", "In
  theory.", "That's the plan.", "Me?", ...) and it is not done to them; nor is it through a preposition ("at", "over",
  "toward", "behind", "of" ...), to a possessive's owner ("Elara's chains"), to a thing thrown at them, or to a
  figurative object ("I hurl insults at Marcus").
* Everything else that names a person or the Tome takes the least consequential reading. Aimed at a person in any way
  ("I lift Elara", "I search Marcus", "I climb over Marcus") or naming one with aggression, it is words to them: a
  threat when it is aggressive and they are the one spoken to, else talk they hear; never an act on the obstacle at
  hand. So it is with the Tome, but for a hand on it that neither strikes nor hides it, which changes nothing ("I lift
  the Tome", "I throw the Tome"), and the Tome as the mere place of a quiet act, which leaves the act on the obstacle
  at hand ("I hide behind the Tome"). A possessive's owner only hears the act ("I smash Elara's chains" works the
  door).
* Any other **act** (fight, force, sneak, search, use, ...) is something the player does in their own sentence ("I
  <act>", or a bare command when no one hears the sentence: "Elara, quick! Hide!" tells Elara to hide), as permissive
  as it ever was: its object is an obstacle the phrase after its verb names ("I shoot past the Tome at the drone",
  "I sneak past the drone"), else the obstacle at hand; what is thrown "at" something is not what is hit; a verb that
  needs an object in the scene is no act without one ("I burn with rage", "I throw up"). A name spoken to in another
  sentence never retargets an act. An act said as a plan, a wish or on a condition ("if ..., I ...", "when it turns,
  I ...", "... or I shoot", "one more step and I shoot", "you move, I shoot", "I shoot him if he moves", "I'll / I
  could / I should ...") is never done now: it is a threat when what it would be done to is the one spoken to (or
  nothing else is named), and plain talk otherwise. So is an act of things in general said to someone ("Marcus, I
  shoot drones for a living"), or one after a question in its sentence ("I ask the Tome where Marcus is, then I shoot
  him" asks, and shoots no one). Something handed to a person is an offer to them, not an act ("I give Marcus the
  flare"). One act a turn: the first sentence that has one.
* A **question** put to no one by name goes, in the second person ("Who are you?"), to the lone person present when
  there is one; when it is impersonal ("Where is the lifeboat?"), or no one person is there to take it, to the
  narrator, which is the Tome. A question that names someone present in passing is theirs ("Is Elara lying?"). The
  Tome answers what is put to it whatever else the line does. Plain words put to no one beside it ("Stay close.") are
  still the lone person's to hear. A sentence that opens "When the drone turns, ..." or "Should he move, ..." says
  when, and asks nothing. A question to the Tome with no topic of its own borrows one only through a word that refers
  back ("Elara, Marcus is armed. Tome, is that true?"), from the words before it, and never the name of someone else
  spoken to.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from .state import Intent

_WORD = re.compile(r"[a-z']+")

# Intent cues: what the player does, by the word for it. Multi-word cues match as phrases, the rest as whole words.
INTENT_CUES: dict[Intent, tuple[str, ...]] = {
    Intent.FLEE: ("run away", "flee", "retreat", "fall back", "get out", "bolt", "escape back", "turn back"),
    Intent.FIGHT: ("attack", "shoot", "stab", "kill", "fight", "strike", "fire at", "punch", "slash", "swing at", "hit",
                   "open fire", "gun down", "blast", "club", "burn", "set fire to", "tackle", "hurt", "beat", "slap",
                   "strangle", "choke"),
    Intent.FORCE: ("force", "pry", "break", "smash", "kick", "lift", "shove", "bash", "ram", "wrench", "cut through",
                   "tear", "climb", "haul", "brace", "jump", "crush", "shatter", "slam", "destroy", "pound", "hammer",
                   "throw", "hurl"),
    Intent.SNEAK: ("sneak", "hide", "slip past", "creep", "tiptoe", "quietly", "pick the lock", "distract",
                   "steal", "pickpocket", "crawl", "stay low", "shadow", "unseen", "cover", "smother", "shade"),
    Intent.SEARCH: ("search", "examine", "inspect", "analyze", "analyse", "scan", "hack", "investigate",
                    "study", "decode", "check", "listen", "read", "rummage", "loot", "figure out", "trace"),
    Intent.USE: ("use", "activate", "apply", "inject", "light the", "turn on", "plug", "equip", "drink",
                 "flare", "medkit", "switch on", "power up"),
    Intent.REST: ("rest", "sleep", "camp", "catch my breath", "bandage", "recover", "patch up", "breathe",
                  "sit down", "take a moment"),
    Intent.ADVANCE: ("go on", "move on", "continue", "proceed", "head to", "onward", "next", "descend",
                     "go deeper", "go through", "enter", "keep going", "press on", "go forward", "leave"),
    Intent.TALK: ("say", "ask", "tell", "talk", "speak", "greet", "whisper", "persuade", "threaten",
                  "lie to", "promise", "offer", "convince", "beg", "answer", "reply", "call out", "hello"),
    Intent.LOOK: ("look", "look around", "where am i", "describe", "what do i see", "survey", "observe"),
}
# The closed grammar of a consequential act: the verbs that strike a person or the Tome, and those that smother the
# Tome's glow, each a whole word in its base form after "I" (``_Reader._closed``). Nothing else is a blow at a person
# or the Tome, nor a hand over its light: not "throw" or "hurl" (what is thrown is not what is hit), not "fire at" or
# "swing at" (a person after a preposition is never the object).
STRIKES = ("attack", "hit", "strike", "shoot", "stab", "punch", "kick", "smash", "slash", "club", "tackle", "hurt",
           "burn", "kill", "fight", "blast", "bash", "beat", "slap", "shove", "ram", "crush", "shatter", "break",
           "destroy", "pound", "hammer", "strangle", "choke", "tear", "cut", "gun down", "beat up", "set fire to")
SMOTHERS = ("cover", "smother", "shade", "hide", "muffle", "wrap", "dim", "shroud", "cloak", "conceal")
_KIND = {**{v: "strike" for v in STRIKES}, **{v: "smother" for v in SMOTHERS}}
# The words that only ever hide a light count as an act only when the Tome is their object ("I dim the Tome"; "The
# corridor is dim." does nothing).
_SMOTHER_ONLY = ("dim", "muffle", "shroud", "wrap", "tuck", "cloak", "conceal")
# Verbs that are an act only on something in the scene: "I burn the Tome", "I throw a rock at the drone"; not "I burn
# with rage", "I throw up", "I burn the note".
_TRANSITIVE = {"throw", "hurl", "burn", "set fire to", "hurt", "tackle", "beat", "slap", "strangle", "choke"}
# Who a description may stand for, when a model resolves one ("I shoot the old soldier": Marcus): a person only. The
# Tome is only ever named (``_TOME_WORDS``), never described: "the stupid thing" or "the relic" is not the Tome.
PERSON_NOUNS = {"man", "woman", "guy", "girl", "boy", "lady", "gentleman", "soldier", "guard", "warden", "engineer",
                "redhead", "stranger", "traitor", "liar", "coward", "bastard", "fool", "idiot", "survivor", "kid",
                "brute", "thug", "officer", "captain", "sergeant", "mechanic", "prisoner", "veteran", "mercenary",
                "person", "fellow"}
# The Tome by name: "the", "this" or "my" + one of its names ("tome", "book", "monolith"), or "Tome" alone, with at
# most three of these words between ("the cursed old Tome", "this damn book", its title "the Voidlit Tome"). Any
# other determiner or word ("a book", "another Tome", "every book", "the log book", "the hologram Tome", "Marcus's
# Tome") names something else.
_TOME_WORDS = (r"cursed|black|damn|damned|stupid|glowing|wretched|old|dark|bloody|heavy|cold|warm|hateful|evil|lying|"
               r"talking|voidlit")
_TOME_RUN = re.compile(rf"(?:(?<![\w'’-])(?:{_TOME_WORDS})\s+){{0,3}}$")
_TOME_ARTICLE = re.compile(r"(?<![\w'’-])(?:the|this|my)\s+$")

QUESTION_WORDS = {"what", "where", "who", "why", "how", "which", "when", "is", "are", "can", "could", "do", "does",
                  "did", "will", "would", "should", "tell", "explain"}
_AUXILIARIES = {"is", "are", "was", "were", "am", "do", "does", "did", "will", "would", "can", "could", "should",
                "shall", "may", "might", "must", "has", "have", "had", "isn't", "won't", "can't", "didn't"}
# Words that may open an address before the name of the one spoken to ("Hey Tome, ...", "Tell me, Tome, ...",
# "Dear Tome, ...", "Excuse me, Tome, ..."), the few that may come before a name said last ("..., O Tome?"), and the
# question words that may follow a bare name ("Tome what does Elara want?").
VOCATIVE_LEADS = (r"(?:hey|hi|oi|oh|o|ah|so|well|now|then|there|listen|look|okay|ok|right|alright|please|and|but|say|"
                  r"tell me|dear|my|excuse me|pardon me|pardon|sorry|hello)")
TRAILING_LEADS = r"(?:hey|oh|o|my|dear)"
WH_WORDS = r"(?:what|where|who|whose|why|how|which|when)"
# Speaking to someone in the third person: "I ask the Tome who Marcus is", "I turn to the Tome and ask ...".
ADDRESS_VERBS = (r"(?:ask|asks|asked|question|questions|questioned|tell|tells|told|say to|said to|whisper to|"
                 r"whispered to|turn to|turned to|call to|called to|beg|begged|plead with|pleaded with|"
                 r"demand(?:s|ed)?(?:\s+of|\s+that)?)")
ASKING = {"ask", "asks", "asked", "question", "questions", "questioned"}
# A question asked without a question's form: "I demand to know where ...", "tell me about ...", "I wonder whether".
INDIRECT = re.compile(rf"\b(?:know|tell me|wonder(?:ing)?|ask(?:s|ed|ing)?|find out)\s+(?:{WH_WORDS}|whether|if|"
                      r"about)\b")
# Where, in a question, the asking begins: only the words before it can tell what the player does.
QUESTION_OPENS = re.compile(r"\b(?:what|where|who|whose|why|how|which|when|is|are|am|can|could|do|does|did|will|"
                            r"would|should|shall|may|might|has|have)\b")
LOOK_AROUND = re.compile(r"\b(?:where am i|where are we|what do i see|what do we see)\b")
SECOND_PERSON = re.compile(r"\b(?:you|your|yours|yourself|you're|you've|you'll|you'd)\b")

_SENTENCE = re.compile(r"[^.!?]*(?:[.!?]+|$)")
_QUOTED = re.compile(r"[\"“][^\"“”]*[\"”]?")
_OPEN_QUOTE = re.compile(r"(?:(?<=\s)\"|“)\s*")
# Who does the act: the player ("I", "we", "let me"), and how it stands: done now, said as a plan (promised, or on a
# condition: "if he is, I ...", "when it turns, I ...", "... or I shoot you", "I shoot him if he moves"), or not done.
_SUBJECT = re.compile(r"\b(?:i'm|i'll|i|we're|we'll|we|let me|let's|let us)(?![\w'])")
_NEGATED = re.compile(r"\b(?:not|never|no longer|cannot)\b|n't\b")
_FUTURE = re.compile(r"'ll\b|\b(?:will|shall|gonna|going to|would|could|might|should|must|may|needs? to|"
                     r"ha(?:ve|s|d) to|got to|gotta|wants? to|wanna|about to|wish)\b")
_CONDITION = re.compile(r"\b(?:(?<!even )if|unless|until|till|when|whenever|once(?!\s+(?:more|again))|as soon as|"
                        r"(?<!at )the (?:moment|second|minute|instant)|next time|in case|maybe|perhaps|swear|"
                        r"should)\b")
# A condition said before the player's "I" without a condition's word: "Drop it or I shoot", "One more step and I
# shoot", "You move, I shoot" (not "Now or never: I shoot", "Elara and I shoot").
_CONDITION_BEFORE = re.compile(r"\b(?:or|else|otherwise)\W*$|(?:\w+\W+){2,}and\W*$|"
                               r"^\W*(?:you|he|she|they|it)\s+[a-z']+[^,]*,\s*$")
_CONDITION_AFTER = re.compile(r"\b(?:(?<!even )if|unless|until|till|when|whenever|as soon as|next time|"
                              r"the (?:moment|second|minute|instant)|once\s+(?:it|he|she|they|you|we|i|the|that|this|"
                              r"my|his|her|its|their))\b")
# What a verb of looking into something asks after: "I check if the door is open" checks, it does not wait for it.
_ASKS_AFTER = re.compile(r"\s*(?:to see\s+)?(?:if|whether)\b")
_CLAUSE_END = re.compile(r"[;:]|\b(?:and|then|but|while|before|after)\b")
_SAYS_TO_DO = re.compile(r"\b(?:ask|tell|order|beg|want|need|get|let|make)s?\s+(?:\w+\s+){1,3}to\s+$|"
                         r"\b(?:watch|watches|see|sees|saw|hear|hears|heard|let|lets)\s+(?:\w+\s+){1,2}$")
_COMMAND_LEAD = re.compile(r"[\s\"'“”(]*(?:(?:\w+ly|then|now|just|and|so|first|quick)\s+)*")
# The object of an act: the words after its verb (past any particle) up to the next clause or prepositional phrase.
_PARTICLES = re.compile(r"(?:\s*,)?(?:\s+(?:at|on|upon|into|onto|up|down|away|open|apart|hard|again|over|in|out|"
                        r"toward|towards|against|for))*\s*")
_OBJECT_END = re.compile(r"[,;:.!?\"”]|\s(?:with|using|near|beside|by|to|while|and|then|but|because|so|or|if|when|"
                         r"before|after|until|as|for)\b")
# In that phrase, a name after one of these is where the act happens, not what it is done to ("I hide behind the
# Tome", "I shoot past Marcus at the drone"); after "of", "from" or "off" it is the owner of what is ("a page from the
# Tome" is the Tome's own, "the chains of Elara" are not Elara); after any other it is what the act is aimed at ("at
# Marcus", "over the Tome", "toward Elara"), which is never its object. The last of them before a name governs it.
_PLACE_WORDS = r"behind|past|from|around|round|through|across|under|beneath|off|of"
_GOVERNS = re.compile(rf"\b(?:{_PLACE_WORDS}|at|on|upon|into|onto|toward|towards|against|over|above|below|"
                      r"holding|with|using)\b")
_PLACES = set(_PLACE_WORDS.split("|"))
_AIMED_AT = re.compile(r"\bat\s+(?:the|a|an|that|this|those|these|his|her|him|them|its|their|my|your|it|me)\b")
_POSSESSIVE = re.compile(r"['’]s\b|(?<=s)['’](?!\w)")
# What of a person or the Tome is the person or the Tome itself, so that an act on it is an act on them ("I punch
# Marcus's face", "I smother the Tome's glow"); anything else they own is not ("I smash Elara's chains").
_BODY = {"face", "head", "jaw", "nose", "mouth", "eye", "eyes", "ear", "arm", "arms", "leg", "legs", "knee", "knees",
         "kneecap", "kneecaps", "thigh", "thighs", "hand", "hands", "foot", "feet", "chest", "torso", "gut", "stomach",
         "belly", "throat", "neck", "back", "shoulder", "shoulders", "body", "side", "ribs", "skull", "brain", "shin",
         "shins", "ankle", "wrist", "finger", "fingers", "cheek", "lip", "lips", "teeth", "groin", "kidney", "temple",
         "forehead", "elbow", "hip", "spine"}
_TOME_PARTS = {"glow", "light", "lights", "glass", "page", "pages", "cover", "covers", "spine", "surface", "face",
               "lock", "clasp", "binding"}
# the few words that may come before such a part in the closed grammar ("his left leg", "the Tome's cracked glass";
# never "his fake leg" or "Marcus's holographic face")
_PART_WORDS = (r"left|right|upper|lower|bare|bloody|ugly|smug|stupid|lying|fat|big|damn|good|bad|broken|wounded|"
               r"injured|black|cracked|glowing|cursed|dark")
_REFERS_BACK = re.compile(r"\b(?:it|its|that|this|those|these|they|them|so|true)\b")
_GENERIC = {"nothing", "anything", "everything", "nobody", "anybody", "anyone", "everyone", "someone", "something",
            "things", "people"}
_DETERMINERS = {"the", "a", "an", "my", "your", "his", "her", "its", "our", "their", "this", "that", "these", "those"}
_PRONOUNS = {"you": "you", "your": "you", "yourself": "you", "it": "it", "its": "it", "itself": "it", "him": "he",
             "his": "he", "himself": "he", "her": "she", "herself": "she", "them": "they", "their": "they"}
_OWNING = {"your", "its", "his", "their"}       # and "her" before a noun ("her chains", not "her in the leg")
# What makes "Tome" alone a noun phrase of something else (``_Reader._is_tome``): a determiner or an owner before it
_DETERMINER = r"the|a|an|this|that|these|those|my|your|his|her|its|our|their|some|any|another|every|each|no|one|other"
_NOT_NOUNS = {"in", "on", "at", "into", "onto", "again", "hard", "now", "down", "up", "back", "away", "off", "out",
              "over", "once", "twice", "too", "first", "dead", "there", "here", "point", "from", "through", "across"}
_MEN, _WOMEN = {"man", "boy", "guy", "gentleman", "fellow", "he"}, {"woman", "girl", "lady", "she"}
# Something handed to a person is an offer to them, not an act ("I give Marcus the flare", "Marcus, I give you the
# flare", "I give the lamp to Elara"); handed to the Tome (fed to it) it is still an act.
_GIVE = re.compile(r"(?<![\w'])(?:give|gives|hand|hands|offer|offers|pass|passes|slide|slides|toss|tosses|throw|"
                   r"throws)(?![\w'])")


# The closed grammar of an act on a person or the Tome (``_Reader._closed``): its sentence is, past whom it speaks to,
# "I" + one of its verbs (an adverb of its list may come between) + the person or the Tome as the direct object + a
# trailing adverbial of its list ("hard", "three times", "with my slug rifle", "in the face", "and run"). A plan, a
# wish, a condition, a denial, a retraction, or a fragment that takes it back or makes it a story, anywhere in the line
# (not only in its sentence: "Marcus... if you lie to me again... I shoot you.", "I shoot Marcus. Just kidding.", "I
# shoot Marcus. Nope.", "I shoot Marcus, says Elara."), and no act in the line is done to them. "as hard as I can",
# "with all my might" and "once" said of the blow ("I shoot Marcus once.") are none of these; "once" before a clause
# ("once he moves", "Once, I ...") is.
_RETRACTS = (r"nope|nah|psych|psyche|sike|(?:ha){2,}|(?:he){2,}|lmao|rofl|yeah,? right|"
             r"(?:scratch|cancel|strike|forget) (?:that|it)|never ?mind|on second thoughts?|second thoughts|"
             r"take (?:it|that) back|said (?:nobody|no one|no-one)|says who|"
             r"is (?:that|it|this) (?:okay|ok|alright|all right|fine|cool|allowed)|undo|tempting|"
             r"just (?:a thought|an idea|saying|thinking)|thinking about it|"
             r"(?:that's|that is) what i(?:'m| am) thinking|"
             r"in (?:my|a|the|your) (?:story|tale|novel|game|dreams?|fantasy|version)")
_RETRACTION = re.compile(rf"[\W\s]*(?:just\s+)?(?:{_RETRACTS})[\W\s]*")     # a sentence that only takes one back
_MARKS = re.compile(
    r"\b(?:would|could|(?<!my )(?<!his )(?<!her )might|should|shall|must|may|(?<!as i )(?<!i possibly )can|cannot|"
    r"will|gonna|going to|about to|needs? to|ha(?:ve|s|d) to|got to|gotta|wants?|wanted|wanna|wish|swear|maybe|"
    r"perhaps|if|unless|until|till|when|whenever|as soon as|in case|next time|"
    r"once(?=\s+(?!(?:more|again|and|then|or|with|in|on|into|to|across|over|for|now)\b)[a-z])|"
    r"the (?:second|moment|minute|instant)|or|else|otherwise|before|after|not|never|no|as if|"
    r"kidding|joking|jokes?|jk|lol|hypothetical(?:ly)?|in theory|theoretical(?:ly)?|imagin\w*|pretend\w*|suppos\w*|"
    r"let(?:'s| us) say|say (?:i|we)|dream\w*|fantas\w*|someday|some day|one day|tomorrow|later|plans?|planned|"
    r"what (?:i|we)(?:'d)? (?:do|did)|in (?:my|your) (?:head|mind|heart|imagination|thoughts)|"
    rf"{_RETRACTS})\b|"
    r"(?:^|[.!?;:—–-]\s*)once\b(?!\s+(?:more|again)\b)|(?:^|[.!?,;:—–-]\s*)(?:says|said)\s+[a-z]|"
    r"[.!?]\s*wait\b|n't\b|'ll\b|'d\b|\.{2,}|…")      # "wait" after a sentence ("I shoot Marcus. Wait."), not before
# "you" for more than one, or for one of several spoken to together ("Both of you, ...", "Marcus, you and Elara: ...",
# "Elara and you"): it stands for no one in the closed grammar
_JOINT = re.compile(r"\b(?:both(?=\s*(?:[,.!?;:—–-]|$|of\s+you\b))|you both|you two|the two of you|all of you|you all|"
                    r"y'all|you guys|you people|either of you|you\s+and\s+(?!(?:i|me|my|mine|your|yours)\b)|"
                    r"and\s+you)\b")
_ADVERBS = (r"quickly|swiftly|suddenly|hard|harder|fast|viciously|savagely|brutally|violently|silently|quietly|"
            r"repeatedly|wildly|blindly|desperately|carefully|precisely|cleanly|immediately|instantly|finally|again|"
            r"now|right now|at once|then|just|also|simply|once more|once again|again and again|twice|first|too|"
            r"calmly|angrily|deliberately|slowly|coldly|gently|firmly|furiously|casually|once")
_VERBS = "|".join(v.replace(" ", r"\s+") for v in sorted(_KIND, key=len, reverse=True))
# what may lead the player's "I": "And", "Then,", "Calmly,", "Without a word," (up to two of them)
_LEADS = rf"and|so|{_ADVERBS}|without (?:hesitation|warning|mercy|a word|a sound|a second thought)"
# a particle after the verb: "down" after a blow ("I cut down Marcus"), "up" after a smother ("I cover up the Tome")
_CLOSED = re.compile(rf"[^\w,;:]*(?:(?:{_LEADS})\s*,?\s+){{0,2}}i\s+(?:(?:{_ADVERBS})\s+)?(?P<verb>{_VERBS})"
                     r"(?:\s+(?P<particle>up|down))?(?![\w'])")
_PARTICLE = {"strike": "down", "smother": "up"}
_STOP = (r"(?:at|on|upon|in|into|onto|to|toward|towards|over|above|below|under|beneath|behind|past|from|around|round|"
         r"through|across|against|near|beside|by|of|off|with|using|and|or|but|then|if|when|while|so|as|for|my|your|his|"
         r"her|its|our|their|him|them|it|you|me|i|he|she|they|we|this|that|these|those|not|no|until|unless|before|"
         r"after|holding|next|hard|again|now|twice|once|too|first|down|up|out|away|aside|apart|open|shut|dead|"
         r"cold|flat)(?![\w'])")
_DESCRIBING = re.compile(rf"(?!{_STOP})([a-z-]+)(?![\w'])\s*")
# What a blow is struck with, or a light hidden under: any noun phrase ("my slug rifle", "a shard of glass", "the butt
# of my rifle") but one that names a person or the Tome (``_trails``), or one said figuratively ("with kindness", "with
# the truth", "with everything").
_FIGURATIVE = (r"kindness|truth|questions?|words?|insults?|love|looks?|glares?|stares?|gaze|logic|facts?|sarcasm|"
               r"silence|charm|smiles?|compliments?|praise|kiss(?:es)?|hugs?|lies|lie|accusations?|arguments?|news|"
               r"jokes?|threats?|curses?|rumou?rs?|gossip|eyes|mind|thoughts?|voice|tongue|wit|reason|evidence|"
               r"lines?|remarks?|comebacks?|ideas?|offers?|stor(?:y|ies)|speech|everything|anything|something|"
               r"nothing|all|puns?|fear|terror|guilt|shame|reality|treatment|lawsuits?|winks?|nods?|cameras?|photos?|"
               r"pictures?|snapshots?|truths|retorts?|grins?|riddles?|cooking|counterarguments?|music|songs?|skills?")
_NP_DET = r"(?:(?:my|a|an|the|this|that|both|all|his|her|your|its|one of my|some)\s+)"
_NP_WORD = (r"(?!(?:and|or|but|then|if|when|while|so|as|for|in|into|on|onto|at|to|with|using|until|unless|before|"
            r"after|from|of|over|across|against|through|i|you|he|she|it|we|they|him|her|them|me|again|now|hard|is|are|"
            r"was|were|be|which|who|because|right|square|squarely|straight|point|times|" + _FIGURATIVE +
            r")(?![\w'-]))[a-z][a-z'-]*")
_NOUN_PHRASE = rf"{_NP_DET}*(?:{_NP_WORD}\s+){{0,3}}{_NP_WORD}(?:\s+of\s+{_NP_DET}*(?:{_NP_WORD}\s+){{0,2}}{_NP_WORD})?"
_TRAIL_WORDS = (rf"{_ADVERBS}|away|apart|dead|cold|flat|senseless|unconscious|as (?:hard|fast) as i (?:possibly )?can|"
                r"(?:really|very|so) hard|over and over|completely|entirely|fully|totally|"
                r"(?:\d+|two|three|four|five|six|seven|eight|nine|ten|a few|a couple of|several|a dozen|many) times|"
                r"point[- ]blank|at (?:close|short|point[- ]blank) range|in cold blood|from behind|head[- ]on|"
                r"on the spot|right away|right there|then and there|"
                r"without (?:hesitation|warning|mercy|a word|a sound|a second thought)|"
                r"(?:to|into) (?:death|pieces|bits|shreds|a pulp|the (?:ground|floor|deck|wall))")
# where on them a blow lands, on their own body only ("in the face", "square in the jaw", "in the back of the head";
# never "in my head"); against what around them ("into the wall", "against a rock"), with a word of a closed list at
# most before it ("into the cold wall", never "in the holo room" or "in the next room"); and, for a hand over a light
# only, what it is hidden in ("in my pack", "under the floor"; "I crush Marcus in my arms" is no blow)
_WHERE = (r"floor|ground|wall|deck|room|corridor|pit|shaft|vent|bulkhead|hull|dirt|water|railing|stairs|hatch|rock|"
          r"rocks|stone|boulder|pillar|beam|pipe|table|ceiling")
_WHERE_WORDS = (r"metal|steel|iron|cold|hard|rusted|rusty|nearest|concrete|wet|dirty|filthy|grimy|bloody|damn|stone|"
                r"sharp|jagged|big|large|bare")
_CONTAINERS = (r"pack|backpack|bag|satchel|pocket|coat|jacket|cloak|blanket|cloth|sleeve|shirt|vest|robe|rags?|tarp|"
               r"sack|pouch|crate|box|locker|case|kit|rucksack|duffel|arms|lap|hands|palms|floor|floorboards|rubble|"
               r"debris|bunk|cot|bed")
_BODY_PART = rf"(?:(?:{_PART_WORDS})\s+)?(?:{'|'.join(sorted(_BODY))})"
_TRAIL_PP = (rf"(?:with|using)\s+{_NOUN_PHRASE}|"
             r"(?:(?:right|square|squarely|straight|full|directly)\s+)?(?:in|on|across|to|into|through|over|between)\s+"
             rf"(?:the|his|her|their|its|both)\s+(?:{_BODY_PART}|heart)(?:\s+of\s+(?:the|his|her|their)\s+{_BODY_PART})?|"
             r"(?:in|into|on|onto|across|against|over|through|to|off|down)\s+(?:the|a|an)\s+"
             rf"(?:(?:{_WHERE_WORDS})\s+)?(?:{_WHERE})")
_HIDDEN_IN = (r"(?:in|inside|into|under|beneath|within)\s+(?:(?:my|a|an|the|this|that|some)\s+)?"
              rf"(?:[a-z-]+\s+)?(?:{_CONTAINERS})")
# what trails only a blow ("down", "back", "full of holes", "under my boot"), or only a hand over a light ("up",
# "behind my back")
_TRAIL_ITEMS = {"strike": rf"\s*,?\s*(?:{_TRAIL_WORDS}|{_TRAIL_PP}|down|back|aside|full of holes|"
                          r"under (?:my|the) (?:boots?|heels?|foot|feet))(?![\w'])",
                "smother": rf"\s*,?\s*(?:{_TRAIL_WORDS}|{_TRAIL_PP}|{_HIDDEN_IN}|up|behind my back)(?![\w'])"}
# the one way on past the blow: a short escape ("and run"), never a story ("and wake up screaming")
_TRAIL_TAIL = r"(?:\s*,?\s*(?:and then|and|then)\s+(?:i\s+)?(?:run(?: away)?|flee|duck|hide|back away))?[^\w,;:]*"


class Closed(NamedTuple):
    """A sentence in the closed grammar of a blow or a smother: what it does and to whom (a person's or the Tome's id,
    or None for a description a model may resolve: "the old soldier"), and that description's head noun."""

    kind: str               # "strike" or "smother"
    intent: Intent          # its verb's
    target: str | None
    noun: str | None = None


class Act(NamedTuple):
    """What the player does in a line: the first sentence that does anything."""

    intent: Intent
    kind: str | None        # "strike" or "smother": what it does to the Tome if the Tome is its object
    target: str | None      # its object: the Tome, a person or an obstacle; None when it names nothing in the scene
    words: str = ""         # its sentence, as typed
    owner: str | None = None    # the person whose thing it is done to ("I kick Marcus's door in"), who hears it


class Reading(NamedTuple):
    """A line of player input, read sentence by sentence and combined."""

    act: Act | None
    target: str                             # the act's object, else whom the words are for (one asked first), else
                                            # the first named in passing (the Tome last), else "none"
    heard: tuple[tuple[str, str], ...]      # each one spoken to (the Tome among them) and the words they hear, in order
    asked: frozenset[str]                   # the ones something is asked of
    tome_question: str                      # what the Tome is asked, questions put to no one among them; "" if nothing
    about: str | None                       # the one a pronoun in that question stands for ("Tome, is he lying?")
    narrator: bool                          # a question put to no one in particular: the Tome's
    threat: bool                            # a blow promised to the one spoken to ("Tome, I will destroy you")
    aside: str = ""                         # second-person questions the lone person takes, which the Tome hears too
    unaddressed: str = ""                   # plain words put to no one: neither an act nor a question the Tome takes
    said: bool = False                      # an act only said: a plan, a threat, or done to a "you" no one is
    offer: str | None = None                # the person something is handed to ("I give Marcus the flare")
    context: str = ""                       # what a vague question to the Tome refers back to ("is that true?"):
                                            # the words up to it, no name of one spoken to; "" when it refers back
                                            # to nothing ("Tome, is it safe? Elara, is it?" borrows no "Elara")
    closed: tuple[Closed, ...] = ()         # its sentences in the closed grammar of a blow or a smother, in order

    @property
    def addressed_tome(self) -> bool:
        return self.narrator or self.target == "tome" or any(tid == "tome" for tid, _ in self.heard)

    def hears(self, tid: str) -> str:
        return " ".join(words for who, words in self.heard if who == tid)


@dataclass
class _Sentence:
    start: int
    text: str
    to: str | None = None                   # whom it speaks to by name: its vocative, or a name said alone beside it
    body: tuple[int, int] = (0, 0)          # its words outside the vocative (offsets within the sentence)
    third: bool = False                     # it speaks of the one it speaks to in the third person ("I ask the Tome")
    asking: bool = False                    # ... and asks them something
    name_only: bool = False
    hearer: str | None = None               # whom its words are for
    question: bool = False
    narrator: bool = False
    aside: bool = False                     # a question in the second person that the lone person takes
    own: frozenset[str] = frozenset()       # all it speaks to by name itself, each set off ("Marcus, Elara, run.")

    @property
    def words(self) -> str:
        return self.text[self.body[0]:self.body[1]]


def sentences(low: str) -> list[tuple[int, str]]:
    """The sentences of a line, each with where it starts (its closing punctuation kept)."""
    return [(m.start(), m.group()) for m in _SENTENCE.finditer(low) if re.search(r"[a-z]", m.group())]


def is_question(text: str) -> bool:
    """A question mark; a question word opening the words, or opening them after a short opener ("Listen, what are
    you?"); or a question asked indirectly ("I demand to know where ...")."""
    body = text.lstrip("\"“' ")
    if "?" in body or INDIRECT.search(body):
        return True
    if re.match(r"(?:should|were|had)\b[^,]*,\s*(?:i|we)\b", body):
        return False                            # "Should he move, I shoot him." says when, and asks nothing
    for part in (body, body.split(",", 1)[1] if "," in body[:24] else ""):
        words = _WORD.findall(part)
        # "When does it turn" asks; "When the drone turns, I shoot it." says when
        if words and words[0] in QUESTION_WORDS and (words[0] != "when" or words[1:2] and words[1] in _AUXILIARIES):
            return True
    return False


def _alternation(aliases: Sequence[str], tid: str) -> str:
    return "|".join(map(re.escape, sorted({tid, *(a.lower() for a in aliases)}, key=len, reverse=True)))


def _gender(aliases: Sequence[str]) -> str | None:
    words = {a.lower() for a in aliases}
    return "he" if words & _MEN else "she" if words & _WOMEN else None


def describes(noun: str, tid: str, aliases: Sequence[str]) -> bool:
    """Whether a description's head noun ("the old soldier") can stand for ``tid``: a person, by one of their names or
    a word for a person of their sex (or of none given). Never the Tome, which is only ever named (``_TOME_WORDS``):
    "the stupid thing", "the relic" or "the log book" is not the Tome."""
    if tid == "tome":
        return False
    return noun in {a.lower() for a in aliases} or noun in PERSON_NOUNS and _gender([noun]) in (None, _gender(aliases))


def mentions(low: str, targets: Mapping[str, Sequence[str]]) -> list[tuple[int, int, str]]:
    """Every one and everything in the scene named in a line, in order: of two names that start together the longer
    ("the warden turret", not the warden), and none inside another. A name that two go by, one of them as a word of a
    longer name of its own, is that one's ("the warden" with the warden turret there is the turret, not Marcus)."""
    found = sorted(((m.start(), -(m.end() - m.start()), not any(a in x.split() and a != x for x in names), tid)
                    for tid, aliases in targets.items() for names in [{tid, *(x.lower() for x in aliases)}]
                    for a in names for m in re.finditer(rf"(?<![a-z']){re.escape(a)}(?![a-z])", low)))
    out: list[tuple[int, int, str]] = []
    for start, neg, _, tid in found:
        if not out or start >= out[-1][1]:
            out.append((start, start - neg, tid))
    return out


def first_named(named: Sequence[tuple[int, int, str]]) -> str:
    """The first one or thing named in passing (of ``mentions``). The Tome, always at hand, only when nothing else is
    named."""
    ids = [tid for _, _, tid in named]
    return next((tid for tid in ids if tid != "tome"), "tome" if ids else "none")


class _Reader:
    """One line, read sentence by sentence (``read``)."""

    def __init__(self, text: str, targets: Mapping[str, Sequence[str]], people: Collection[str] | None) -> None:
        low = text.lower()
        self.text = text if len(text) == len(low) else low     # the words as typed, where lowercasing kept offsets
        self.low, self.targets = low, targets
        self.unquoted = _QUOTED.sub(lambda m: " " * len(m.group()), low)    # words said aloud are said, not done
        self.speakable = list(targets) if people is None else [t for t in targets if t == "tome" or t in people]
        self.people = (set(people) if people is not None
                       else {t for t in targets if _gender(targets[t])})
        self.lone = next(iter(people)) if people is not None and len(people) == 1 else None
        self.names = {tid: _alternation(targets[tid], tid) for tid in self.speakable}
        self.any_name = "|".join(self.names.values()) or r"(?!x)x"
        # those named, but for a name of the Tome's that names something else ("a book", "the log book"); a name said
        # alone ("Book?") is still a call to the one it names
        self.called = mentions(low, targets)
        self.named = [x for x in self.called if x[2] != "tome" or self._is_tome(*x[:2])]
        # a trailing adverbial, which may hold the name of the one spoken to, set off ("I smash you, Tome, and run")
        said_to = rf"\s*,\s*(?:{TRAILING_LEADS}\s+)?(?:the\s+)?(?:{self.any_name})(?![\w'])"
        self.trail = {kind: re.compile(rf"(?:{item}|{said_to})*{_TRAIL_TAIL}") for kind, item in _TRAIL_ITEMS.items()}
        self.sents = [self._parse(start, text) for start, text in sentences(low)]
        # a plan, a wish, a condition, a denial, a retraction or a fragment anywhere in the line, a question of a word
        # or two that names no one ("Me? I shoot Marcus."): no act in the line is done to a person or the Tome
        self.void = bool(_MARKS.search(low)) or any(
            "?" in s.text and len(_WORD.findall(s.text)) <= 2 and not self._named_in(s) for s in self.sents)

    # ------------------------------------------------------------------ one sentence
    def _parse(self, start: int, sent: str) -> _Sentence:
        """A sentence, with whom it speaks to: its vocative, and every name it sets off by punctuation."""
        s = self._vocative(start, sent)
        if not s.third:
            s.own = frozenset({s.to} - {None} | {tid for tid, name in self.names.items() if re.search(
                rf"(?:^|[,:;!?.—–-])[\W\s]*(?:{VOCATIVE_LEADS}\W+)*(?:and\s+)?(?:the\s+)?(?:{name})"
                r"\s*(?:[,:;!?.—–-]|$)", sent)})
        return s

    def _is_tome(self, a: int, b: int) -> bool:
        """Whether a name of the Tome's at ``a:b`` names it (``_TOME_WORDS``): "the", "this" or "my" before it, past
        at most three words of the closed list ("the cursed old book"), or "Tome" alone with no determiner or owner of
        its own ("Hey Tome", "Damn Tome!"); not "a book", "another Tome", "the log book" or "the hologram Tome"."""
        before = self.low[:a]
        before = before[:_TOME_RUN.search(before).start()]
        if _TOME_ARTICLE.search(before):
            return True
        return self.low[a:b] == "tome" and not re.search(
            rf"(?:(?<![\w'’-])(?:{_DETERMINER})|['’]s?)\s+(?:(?!{_STOP})[a-z-]+\s+){{0,2}}$", before)

    def _vocative(self, start: int, sent: str) -> _Sentence:
        s = _Sentence(start, sent, body=(0, len(sent)))
        lead = len(sent) - len(sent.lstrip("\"“”' "))
        anyone = self.any_name
        if re.fullmatch(rf"[\W\s]*(?:(?:{VOCATIVE_LEADS}\W+)+(?:{anyone})|(?:the\s+)?(?:{anyone}))\b"
                        rf"(?:[\W\s]+(?:and\s+)?(?:the\s+)?(?:{anyone})\b)*[\W\s]*", sent):
            ids = [tid for a, _, tid in self.called if start <= a < start + len(sent) and tid in self.names]
            s.to, s.name_only = (ids[0], True) if ids else (None, False)
            return s
        # at its head, or at the head of a quotation in it ('I shout "Tome, where is the lifeboat?"'), whose words
        # before it are still the player's own
        for head in (lead, *(q.end() for q in _OPEN_QUOTE.finditer(sent, lead + 1))):
            for tid, name in self.names.items():
                m = re.match(rf"(?:(?:{VOCATIVE_LEADS}\W+)+(?:{name})|(?:the\s+)?(?:{name}))"
                             rf"(?:\s*(?:[,:;!?.]|-+|–|—)|(?=\s+{WH_WORDS}\b))", sent[head:])
                if m:
                    s.to, s.body = tid, (head + m.end() if head == lead else 0, len(sent))
                    return s
        for tid, name in self.names.items():
            # no article: "Look, the Tome!" points at it
            m = re.search(rf",\s*(?:{TRAILING_LEADS}\s+)?(?:{name})\s*[?!.]*[\"”']*\s*$", sent)
            if m:
                s.to, s.body = tid, (0, m.start())
                return s
            # or set off by commas, saying who that is: "I smash you, Tome, and run.", "Drop it, Marcus, or I shoot."
            if re.search(rf",\s*(?:{TRAILING_LEADS}\s+)?(?:the\s+)?(?:{name})\s*,", sent):
                s.to = tid
                return s
        hits = [(m, tid) for tid, name in self.names.items() if (m := re.search(
            rf"\b({ADDRESS_VERBS})\s+(?:the\s+)?(?:{name})\b(?:\s*[,:]?\s*and\s+(ask|say|tell|whisper)\b)?", sent))]
        if hits:
            m, tid = min(hits, key=lambda h: h[0].start())
            s.to, s.body, s.third = tid, (m.end(), len(sent)), True
            s.asking = m.group(1) in ASKING or m.group(2) == "ask"
        return s

    # ------------------------------------------------------------------ the line
    def read(self) -> Reading:
        sents = self.sents
        for i, s in enumerate(sents):           # a name said alone is spoken to with the sentence beside it
            if not s.name_only or s.to is None or len(sents) == 1:
                continue
            near = [sents[j] for j in (i + 1, i - 1) if 0 <= j < len(sents)]
            free = next((n for n in near if n.to is None and not n.name_only), None)
            if free is None:
                s.to = None                     # a name in passing between two that speak to others
            else:
                free.to = s.to
        current = None
        for s in sents:                         # each one spoken to hears what follows, up to the next one's words
            current = s.to if s.to is not None else current
            s.hearer = current if not (s.name_only and s.to is None) else None
            s.question = is_question(s.words) and not (s.name_only and len(sents) > 1)  # "Tome? I smash you." asks
            if s.hearer is None and s.question:
                self._unnamed_question(s)
        act, act_sentence, threat, said, offer = self._act()
        heard: dict[str, list[str]] = {}
        for s in sents:
            if s.hearer is not None:
                heard.setdefault(s.hearer, []).append(self._typed(s))
        asked = frozenset(s.hearer for s in sents if s.hearer and (s.question or s.asking))
        tome = [s for s in sents if s.narrator or (s.hearer == "tome" and s.question)]
        if not tome and any(s.hearer == "tome" and s.asking for s in sents):
            tome = [s for s in sents if s.hearer == "tome"]
        first_asked = next((s.hearer for s in sents if s.hearer in asked), None)
        if act is not None:                     # of those named in passing, only a person: "I smash Elara's chains"
            person = next((tid for _, _, tid in self.named if tid in self.people and tid != "tome"), None)
            target = act.target or first_asked or next(iter(heard), None) or person or "none"
        else:
            target = first_asked or next(iter(heard), None) or first_named(self.named)
        plain = [s for s in sents if s.hearer is None and not (s.narrator or s.name_only or s is act_sentence)]
        # a vague question borrows a topic only through a word that refers back ("is that true?")
        upto = max((i for i, s in enumerate(sents) if s in tome), default=-1) if any(
            _REFERS_BACK.search(s.words) for s in tome) else -1
        return Reading(act, target, tuple((tid, " ".join(w)) for tid, w in heard.items()), asked,
                       " ".join(s.words.strip() for s in tome), self._about(tome), any(s.narrator for s in sents),
                       threat, " ".join(s.text.strip() for s in sents if s.aside),
                       " ".join(self._typed(s) for s in plain), said, offer,
                       " ".join(s.words.strip() for s in sents[:upto + 1] if not s.name_only),
                       tuple(c for s in sents if (c := self._closed(s))))

    def _typed(self, s: _Sentence) -> str:
        """A sentence as the player typed it."""
        return self.text[s.start:s.start + len(s.text)].strip()

    def _unnamed_question(self, s: _Sentence) -> None:
        """A question with no one named to take it: the lone person's in the second person, the narrator's when
        impersonal; not one at all when the player looks around or says it aloud; the one's it names in passing."""
        if LOOK_AROUND.search(s.text) or self.unquoted[s.start:s.start + len(s.text)] != s.text or \
                re.fullmatch(rf"[\W\s]*(?:{VOCATIVE_LEADS}\b[\W\s]*)+", s.text):     # "Hello?" calls out, asks nothing
            return
        if any(tid in self.people and tid != "tome" for _, _, tid in self._named_in(s)):
            return                              # "Is Elara lying?": hers
        if SECOND_PERSON.search(s.text) and self.lone is not None:
            s.hearer, s.aside = self.lone, True
        else:
            s.narrator = True

    def _named_in(self, s: _Sentence) -> list[tuple[int, int, str]]:
        return [x for x in self.named if s.start <= x[0] < s.start + len(s.text)]

    def _about(self, asked: list[_Sentence]) -> str | None:
        """The one a pronoun in a question to the Tome stands for: "he" for the last man named before it, "she" for the
        last woman, "it" for the last thing."""
        for s in asked:
            for m in re.finditer(r"\b(?:he|him|his|she|her|it|its|they|them)\b", s.words):
                kind = {"him": "he", "his": "he", "her": "she", "its": "it", "them": "they"}.get(m.group(), m.group())
                found = self._antecedent(kind, s.start + s.body[0] + m.start())
                if found and found != "tome":       # asked of the Tome, it is "you"; "it" is something else
                    return found
        return None

    def _antecedent(self, kind: str, before: int, since: int = 0) -> str | None:
        """The last one named between ``since`` and ``before`` that the pronoun can stand for: "it" a thing (the Tome
        among them), "he" or "she" a person of that sex (or of none given), "they" any person."""
        return next(iter(self._antecedents(kind, before, since)), None)

    def _antecedents(self, kind: str, before: int, since: int = 0, owners: bool = True) -> list[str]:
        """Every one named between ``since`` and ``before`` that the pronoun can stand for, the last named first; with
        ``owners`` false, "it" and "they" not for one named only as an owner ("Elara's chains ... I smash them")."""
        found = []
        for _, b, tid in reversed([x for x in self.named if since <= x[0] and x[1] <= before]):
            person = tid in self.people and tid != "tome"
            if not owners and kind in ("it", "they") and _POSSESSIVE.match(self.low, b):
                continue
            if tid not in found and (kind == "it" and not person or kind == "they" and person or
                                     kind in ("he", "she") and person and _gender(self.targets[tid]) in (kind, None)):
                found.append(tid)
        return found

    def _candidates(self, kind: str, s: _Sentence, prev: _Sentence | None, at: int) -> list[str]:
        """Whom a pronoun at ``at`` in a sentence may stand for: those named before it in the sentence, else in the
        sentence before."""
        found = self._antecedents(kind, s.start + at, s.start, owners=False)
        if not found and prev is not None:
            found = self._antecedents(kind, prev.start + len(prev.text), prev.start, owners=False)
        return found

    def _referent(self, kind: str, s: _Sentence, prev: _Sentence | None, at: int) -> str | None:
        """The one a pronoun stands for, read leniently: of an obstacle and a person it could stand for, the obstacle;
        else the last one named; else one named after it in its sentence ("I shoot over his head, Marcus"). "it" is
        never the Tome: an act on it is one on an obstacle, or on nothing ("The Tome glows. I smash it." works the door
        at hand, "The door is next to the Tome. I kick it." the door)."""
        found = [tid for tid in self._candidates(kind, s, prev, at) or
                 self._antecedents(kind, s.start + len(s.text), s.start + at) if kind != "it" or tid != "tome"]
        return next((tid for tid in found if tid not in self.people), None) if len(found) > 1 and \
            any(tid not in self.people for tid in found) else next(iter(found), None)

    def _you(self, s: _Sentence) -> str | None:
        """Whom "you" stands for in the closed grammar: the one its own sentence speaks to by name ("Marcus, I stab
        you.", "I smash you, Tome, and run."), when it speaks to no one else and the line speaks to no one jointly (not
        "Tome, Marcus, ...", "Both of you, listen. Marcus, ...", "Elara and you"; no other name can stand in the closed
        grammar's sentence besides). Never one spoken to in another sentence ("You lied to me, Tome. I smash you.",
        "Tome! I smash you.")."""
        return s.to if s.to is not None and s.own == {s.to} and not _JOINT.search(self.low) else None

    # ------------------------------------------------------------------ the act
    def _act(self) -> tuple[Act | None, _Sentence | None, bool, bool, str | None]:
        """The line's one act and the sentence it is in; whether a blow is promised to someone spoken to, whether an
        act is only said, and the person something is handed to (who then hears that sentence)."""
        threat = said = False
        offer = None
        for i, s in enumerate(self.sents):
            if s.name_only:
                continue
            prev = self.sents[i - 1] if i else None
            to = self._offer(s, prev)
            if to is not None:
                offer = offer or to
                s.hearer = s.hearer or to
                continue
            found = self._act_in(s, prev)
            if isinstance(found, Act):
                return found._replace(words=self._typed(s)), s, threat, said, offer
            threat = threat or found == "threat"
            said = said or found is not None
        return None, None, threat, said, offer

    def _span(self, s: _Sentence) -> tuple[str, int, int]:
        """A sentence's own words (said aloud in quotes, blanked) and where the player's part of them lies: all of it
        in the third person ("I turn to the Tome and ..."), up to where a question in it begins."""
        lo, hi = (0, len(s.text)) if s.third else s.body
        words = self.unquoted[s.start:s.start + hi]
        if s.question:
            m = QUESTION_OPENS.search(words, lo)
            hi = m.start() if m else hi
        return words, lo, hi

    def _act_in(self, s: _Sentence, prev: _Sentence | None) -> Act | str | None:
        """The act a sentence does, if any; else "threat" when it only promises a blow to the one it speaks to, or
        "words" when an act is only said: as a plan or on a condition, of things in general to someone, or done to a
        "you" when no one is spoken to.

        An act lands on a person or the Tome only in the closed grammar (``_closed``). An act on an obstacle is read as
        it always was. Anything else aimed at a person (or, with aggression, naming one) is words to them, never an act
        on the obstacle at hand: "I hurl insults at Marcus", "I lift Elara", "I climb over Marcus". So is an
        aggressive act that names the Tome; a hand on it that neither strikes nor hides it changes nothing ("I lift the
        Tome"), and the Tome as a mere place leaves the act on the obstacle at hand ("I hide behind the Tome")."""
        if s.to is None and LOOK_AROUND.search(s.text):
            return Act(Intent.LOOK, None, None)                       # "where am I?" asks no one: it looks around
        if _RETRACTION.fullmatch(s.words):
            return None                                               # "Strike that." takes back; it strikes nothing
        closed = self._closed(s)
        if closed is not None and closed.target is not None:
            return Act(closed.intent, closed.kind, closed.target)
        words, lo, hi = self._span(s)
        found = []
        for intent, cue, only_tome, pat in _CUES:
            for m in pat.finditer(words, lo, hi):
                how = self._how(words[lo:m.start()], words[m.end():hi], s.hearer is not None, intent is Intent.SEARCH)
                if not how:
                    continue
                obj, note, owner, named = self._object(s, prev, words, m.end(), hi)
                if only_tome and obj != "tome" or cue in _TRANSITIVE and obj is None and \
                        not [role for _, role in named if role != "owner"]:
                    continue                    # "The corridor is dim", "I burn with rage", "I throw up"
                found.append((m.start(), m.end(), intent, cue, how, obj, note, owner, named))
        if not found:
            return None
        counts = {intent: sum(f[2] is intent for f in found) for intent in _ORDER}
        best = max(_ORDER, key=lambda x: counts[x])                   # ties go to the first in the cue table
        _, end, intent, cue, how, obj, note, owner, named = min((f for f in found if f[2] is best),
                                                                key=lambda f: f[0])
        kind = _KIND.get(cue)
        aggressive = intent is Intent.FIGHT or kind == "strike"
        aimed = [(tid, role) for tid, role in named if role not in ("owner", "instrument")]
        if how == "plan":                       # a threat only to the one it is said to; else talk of a plan
            who = obj or next((tid for tid, _ in aimed), None)
            if who is None and note in (None, "bare") and not any(tid != s.hearer for _, _, tid in self._named_in(s)):
                who = s.hearer                  # "Marcus, if you move, I shoot.": at him, nothing else named
            return "threat" if aggressive and who is not None and who == s.hearer else "words"
        if note == "nobody" or (note == "generic" and s.hearer is not None):
            return "words"
        if note == "bare" and aggressive and s.hearer is not None:
            return "threat"                     # "Marcus, I'm warning you. I shoot.": at no thing, said to him
        owner = owner if owner in self.people and owner != obj else None
        if intent is Intent.LOOK or obj is not None and obj != "tome" and obj not in self.people:
            return Act(intent, kind, obj, owner=owner)                # a look, or an obstacle: as permissive as ever
        on = [tid for tid, _ in aimed]
        if aggressive:                          # named with aggression after its verb, and not as an owner
            on += [tid for tid in self._involved(s, words, end, hi) if tid not in on]
        who = next((tid for tid in on if tid != "tome"), None)
        if who is None and "tome" in on:
            roles = {role for tid, role in aimed if tid == "tome"}
            if not aggressive and kind is None and roles == {"object"}:
                return Act(intent, None, "tome")                      # lifted, read, thrown: nothing comes of it
            if not aggressive and roles == {"place"}:
                return Act(intent, kind, None)                        # where it happens: the obstacle at hand
            who = "tome"
        if who is None:
            return Act(intent, kind, None, owner=owner)
        threat = aggressive and who == s.hearer
        s.hearer = s.hearer or who              # words to them: they hear it, as they would one spoken to
        return "threat" if threat else "words"

    def _closed(self, s: _Sentence) -> Closed | None:
        """The sentence as a blow at a person or the Tome, or a hand over the Tome's light, in the closed grammar: past
        whom it speaks to, "I" + one of the verbs (``STRIKES``, ``SMOTHERS``) + the person or the Tome as the direct
        object + a trailing adverbial, with no plan, wish, condition, retraction or fragment that takes it back anywhere
        in the line (``_MARKS``). The object is named in the sentence itself: a person's name ("the" at most before
        it), the Tome's by its closed form (``_is_tome``: "the cursed old Tome"), "you" for the one the sentence speaks
        to by name (``_you``), or their own body or glow ("Marcus's face", "your glass", "the Tome's light"); no other
        pronoun ever is ("it", "him", "his face"); else a description ("the old soldier"), which only a model can
        resolve, and only to a person. A smother lands only on the Tome."""
        if s.third or s.name_only or "?" in s.text:
            return None
        text = self.unquoted[s.start:s.start + len(s.text)]
        if text != s.text or self.void:
            return None
        lo, hi = s.body
        m = _CLOSED.match(text, lo, hi)
        if m is None:
            return None
        verb = " ".join(m.group("verb").split())
        kind = _KIND[verb]
        if m.group("particle") not in (None, _PARTICLE[kind]):
            return None                         # "I hit up Marcus" asks him for something
        intent = _INTENT.get(verb) or _INTENT.get(verb.split()[0], Intent.FIGHT if kind == "strike" else Intent.SNEAK)
        for target, noun, end in self._direct_objects(s, text, m.end(), hi):
            if self._trails(s, kind, target, text, end, hi):
                if target is None:
                    return Closed(kind, intent, None, noun)
                return Closed(kind, intent, target) if target == "tome" or kind == "strike" and \
                    target in self.people else None
        return None

    def _direct_objects(self, s: _Sentence, text: str, at: int,
                        hi: int) -> Iterator[tuple[str | None, str | None, int]]:
        """What may be the direct object of a verb ending at ``at``: (whom it is, the head noun of a description, where
        it ends)."""
        rest = text[at:hi]
        if m := re.match(r"\s+(you|your|him|her|his|it|its|them|their)(?![\w'])", rest):
            # no pronoun but "you", or "your" before their own body or glow, for the one its sentence speaks to
            who = self._you(s) if m.group(1) in ("you", "your") else None
            part = re.match(rf"\s+(?:(?:{_PART_WORDS})\s+)?([a-z]+)(?![\w'])", rest[m.end():])
            if who is not None and m.group(1) == "you":
                yield who, None, at + m.end()
            elif who is not None and part and self._own_part(who, [part.group(1)]):
                yield who, None, at + m.end() + part.end()
            return
        name = next(((a - s.start, b - s.start, tid) for a, b, tid in self.named
                     if s.start + at <= a < s.start + hi), None)
        if name is not None:
            a, b, tid = name
            for a2, b2, tid2 in self.named:     # a name in full: "Marcus Hale"
                if tid2 == tid and a2 - s.start == b + 1 and text[b] == " ":
                    b = b2 - s.start
            # the Tome by its closed form ("the cursed old Tome", ``_is_tome``); a person by their name alone, or
            # "the" and it ("the warden"), never "the hologram Marcus" or "a cardboard Marcus"
            before = rf"\s+(?:(?:the|this|my)\s+)?(?:(?:{_TOME_WORDS})\s+){{0,3}}" if tid == "tome" else \
                r"\s+(?:the\s+)?"
            if re.fullmatch(before, text[at:a]):
                part = re.match(rf"['’]s?\s+(?:(?:{_PART_WORDS})\s+)?([a-z]+)(?![\w'])", text[b:hi])
                if part is None:
                    yield tid, None, b
                elif self._own_part(tid, [part.group(1)]):
                    yield tid, None, b + part.end()
            return
        if m := re.match(r"\s+(?:the|that|this|a|an)\s+", rest):
            end = at + m.end()
            for _ in range(3):                  # its head, after up to two words: "the old soldier"
                w = _DESCRIBING.match(text, end, hi)
                if w is None:
                    return
                yield None, w.group(1), w.end(1)
                end = w.end()

    def _trails(self, s: _Sentence, kind: str, target: str | None, text: str, at: int, hi: int) -> bool:
        """Whether all that follows a direct object at ``at`` is a trailing adverbial of a blow or a smother (``kind``:
        "hard", "again", "with my knife", "in the face", ", Tome," said to the one spoken to, "and run"; for a smother
        also "in my pack"), naming no one else in the scene, and with no "his" or "her" for a body not the target's
        ("I punch Marcus in her face")."""
        if not self.trail[kind].fullmatch(text, at, hi) or any(
                tid != s.to for a, _, tid in self.named if s.start + at <= a < s.start + hi):
            return False
        whose = re.search(r"\b(?:in|on|across|to|into|through|over|between)\s+(his|her|its)\b", text[at:hi])
        return not (whose and target in self.people and _PRONOUNS[whose.group(1)] != _gender(self.targets[target]))

    def _involved(self, s: _Sentence, words: str, at: int, hi: int) -> list[str]:
        """The people and the Tome the player's own words after ``at`` name, but as an owner or an instrument ("Elara's
        chains", "with the Tome")."""
        span = words[at:hi]
        return [tid for a, b, tid in self._in_phrase(s, at, span) if (tid == "tome" or tid in self.people)
                and self._governed(span, a, b, tid) not in ("owner", "instrument")]

    @staticmethod
    def _how(before: str, after: str, spoken_to: bool, asks: bool = False) -> str | None:
        """Whether the player does it ("now") or only says it as a plan ("plan": promised, wished, or on a condition
        before it or after it in its clause); None when it is not the player's act (someone else's, one asked of them,
        negated, or a command heard by the one spoken to: "Elara, quick! Hide!"). ``asks``: a verb of looking into
        something, whose "if" asks ("I check if the door is open")."""
        subjects = list(_SUBJECT.finditer(before))
        if not subjects:
            if spoken_to or not _COMMAND_LEAD.fullmatch(before):
                return None
            subject = None
            between = ""
        else:
            subject = subjects[-1]
            between = before[subject.end():]
            if _NEGATED.search(between) or _SAYS_TO_DO.search(between):
                return None
        clause = after[:m.start()] if (m := _CLAUSE_END.search(after)) else after
        pre = before[:subject.start()] if subject else before
        if _CONDITION.search(pre) or _CONDITION_BEFORE.search(pre) or _CONDITION.search(between) or \
                _CONDITION_AFTER.search(clause) and not (asks and _ASKS_AFTER.match(clause)):
            return "plan"
        if subject is not None and (subject.group() in ("i'll", "we'll") or _FUTURE.search(between)):
            return "plan"
        return "now"

    def _object(self, s: _Sentence, prev: _Sentence | None, words: str, at: int,
                hi: int) -> tuple[str | None, str | None, str | None, list[tuple[str, str]]]:
        """The act's own object (an id, or None when its words name nothing in the scene); a note: "generic" when the
        words name things in general ("drones", "nothing"), "nobody" for a "you" when no one is spoken to or a "him" or
        "her" no one named stands for; the one named as the owner of what it is done to; and the people and the Tome
        its words are aimed at, each with how (``_governed``).

        A possessive is not its owner ("his radio", "Elara's chains") but for what of them is themselves ("her face",
        "the Tome's glow"); a name after "behind", "past", "from" and the like is where the act happens, and the act
        is done to what follows "at" ("I shoot past the Tome at the drone"), else to the place itself only when it is
        an obstacle ("I sneak past the drone"). An obstacle aimed at is the object before a person or the Tome."""
        start = _PARTICLES.match(words, at).end()
        stop = _OBJECT_END.search(words, start, hi)
        phrase = words[at:stop.start() if stop else hi]                 # its particles kept: "at", "over" govern
        lead = start - at
        tokens = _WORD.findall(phrase[lead:])
        first = (tokens or [""])[0]
        roles = [(tid, self._governed(phrase, a, b, tid), a, b) for a, b, tid in self._in_phrase(s, at, phrase)]
        owner = None
        if first in _PRONOUNS:
            kind = _PRONOUNS[first]
            found = (s.hearer if not s.aside else None) if kind == "you" else self._referent(kind, s, prev, start)
            owning = first in _OWNING or first == "her" and len(tokens) > 1 and tokens[1] not in _NOT_NOUNS \
                and not tokens[1].endswith("ly")
            if owning and not self._own_part(found, tokens[1:4]):
                owner = found                   # "his radio": not him
            elif found is None and kind in ("you", "he", "she"):
                return None, "nobody", None, []     # a person no one here is ("Marcus, I hit her."): only words
            elif found is not None:
                roles.insert(0, (found, self._governed(phrase, lead, lead, found), lead, lead + len(first)))
        if any(role == "at" for _, role, _, _ in roles) or roles and _AIMED_AT.search(phrase, roles[0][3]):
            # "I throw the Tome at the drone": what is thrown is not what is hit
            roles = [(tid, "instrument" if role == "object" else role, a, b) for tid, role, a, b in roles]
        owner = owner or next((tid for tid, role, _, _ in roles if role == "owner"), None)
        named = [(tid, role) for tid, role, _, _ in roles if tid == "tome" or tid in self.people]
        objects = [tid for tid, role, _, _ in roles if role in ("object", "at", "aim")]
        things = [tid for tid in objects if tid != "tome" and tid not in self.people]
        if things or objects:
            return (things or objects)[0], None, owner, named
        # an obstacle where the act happens, or whose part it is done to ("the drone's sensor"), is still its object
        places = [tid for tid, role, _, _ in roles if role in ("place", "owner") and tid != "tome"
                  and tid not in self.people]
        if places:
            return places[0], None, owner, named
        if roles:
            return None, None, owner, named
        if all(t in _NOT_NOUNS or t.endswith("ly") for t in tokens):
            return None, "bare", owner, named   # "I shoot.", "I shoot first.": no object at all
        generic = first in _GENERIC or (first.endswith("s") and not first.endswith(("ss", "us", "is"))
                                        and len(first) > 3 and first not in _DETERMINERS)
        return None, "generic" if generic else None, owner, named

    def _in_phrase(self, s: _Sentence, at: int, phrase: str) -> list[tuple[int, int, str]]:
        """Those named in an act's object phrase, with where they are in it."""
        base = s.start + at
        return [(a - base, b - base, tid) for a, b, tid in self.named if base <= a < base + len(phrase)]

    def _own_part(self, tid: str | None, words: Sequence[str]) -> bool:
        """Whether a word for something of ``tid``'s is ``tid`` itself: a person's body, the Tome's glow or pages."""
        parts = _TOME_PARTS if tid == "tome" else _BODY if tid in self.people else set()
        return bool(parts & set(words))

    def _governed(self, phrase: str, a: int, b: int, tid: str) -> str:
        """How the one named at ``phrase[a:b]`` stands to the act: its "object"; what it is aimed "at", or aimed at
        otherwise ("aim": "over the Tome", "toward Marcus"); the "place" it happens ("behind the Tome"); or the "owner"
        of what it is done to ("Elara's chains", "the chains of Elara"). Their own body or glow is them ("Marcus's
        face", "out of Marcus's hand")."""
        preps = list(_GOVERNS.finditer(phrase, 0, a))
        prep = preps[-1].group() if preps else None
        if prep in ("with", "using"):
            return "instrument"                                 # "I hit the drone with the Tome"
        if _POSSESSIVE.match(phrase, b):
            if not self._own_part(tid, _WORD.findall(phrase[b:])[1:4]):
                return "owner"
            return "object" if prep is None else "at" if prep == "at" else "aim"
        if prep in ("of", "from", "off"):                       # "a page from the Tome" is the Tome's own
            before = _WORD.findall(phrase[:preps[-1].start()])
            if before and self._own_part(tid, before[-1:]):
                return "object"
            if prep == "of":
                return "owner"
        return "object" if prep is None else "at" if prep == "at" else "place" if prep in _PLACES else "aim"

    def _offer(self, s: _Sentence, prev: _Sentence | None) -> str | None:
        """The person the player hands something to in a sentence ("I give Marcus the flare", "I give you the flare",
        "I hand the lamp to Elara"), if any."""
        words, lo, hi = self._span(s)
        for m in _GIVE.finditer(words, lo, hi):
            if not self._how(words[lo:m.start()], "", s.hearer is not None):
                continue
            rest = words[m.end():hi]
            to = re.search(r"\bto\s+(?:the\s+)?([a-z']+)", rest)
            head = re.match(r"\s*(?:the\s+)?([a-z']+)\s+(?:the|a|an|my|this|that|some|your|our|"
                            r"(?!across\b|towards\b)\w+s)\b", rest)
            for found in filter(None, (head, to)):
                who = self._person(s, prev, m.end() + found.start(1), found.group(1))
                if who is not None:
                    return who
        return None

    def _person(self, s: _Sentence, prev: _Sentence | None, at: int, word: str) -> str | None:
        """The person present a word at ``at`` in a sentence stands for: their name, or "you", "him" or "her"."""
        if word in ("you", "him", "her", "them"):
            kind = _PRONOUNS[word]
            who = (s.hearer or self.lone) if kind == "you" and not s.aside else \
                self._antecedent(kind, s.start + at, since=s.start)
            if who is None and kind != "you" and prev is not None:
                who = self._antecedent(kind, prev.start + len(prev.text), since=prev.start)
        else:
            who = next((tid for a, _, tid in self.named if a == s.start + at), None)
        return who if who in self.people and who != "tome" else None


def _cue(pattern: str) -> re.Pattern:
    return re.compile(rf"(?<![\w'])(?:{pattern})(?![\w'])")


_ORDER = [i for i in INTENT_CUES if i is not Intent.TALK]
# (intent, its cue, an act only when the Tome is its object, pattern)
_CUES = [(intent, c, False, _cue(r"\s+".join(map(re.escape, c.split())))) for intent in _ORDER
         for c in INTENT_CUES[intent]]
_CUES += [(Intent.SNEAK, c, True, _cue(c)) for c in _SMOTHER_ONLY]
_INTENT = {c: intent for intent, c, _, _ in _CUES}


def read(text: str, targets: Mapping[str, Sequence[str]], people: Collection[str] | None = None) -> Reading:
    """Read a line of player input. ``targets``: every one and everything in the scene, by id, with the names it goes
    by (the Tome's id is "tome"). ``people``: those present who can be spoken to, besides the Tome; None when unknown,
    and then anyone named may be."""
    return _Reader(text.strip(), targets, people).read()
