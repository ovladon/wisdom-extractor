"""Canonicalization + lexical preprocessing (from the paper pipeline, v10_working).

Turns figurative proverbs into minimal comparable propositions ("claims") via a
cascade of structural regex rules, then applies lexical normalization used for
similarity computation. Rules are the v10 `improved_canonicalize` set.
"""
import re

QUOTES = "\"'“”‘’`"

STRUCTURAL_RULES = [
    # comparisons and preferences
    (r"(?i)^(?:it'?s )?better (?:to )?(.+?) than (?:to )?(.+?)\.?$", r"Better \1 than \2."),
    (r"(?i)^(.+?) is better than (.+?)\.?$", r"Better \1 than \2."),
    (r"(?i)^prefer (.+?) (?:over|to) (.+?)\.?$", r"Better \1 than \2."),
    # prohibitions and avoidance
    (r"(?i)^(?:you should )?(?:never|don'?t|do not|cannot|can'?t|avoid) (.+?)\.?$", r"Avoid \1."),
    (r"(?i)^(?:it'?s )?(?:bad|wrong|dangerous) to (.+?)\.?$", r"Avoid \1."),
    # conditional wisdom
    (r"(?i)^(?:when|if) (.+?), (?:then )?(.+?)\.?$", r"If \1, then \2."),
    (r"(?i)^where (?:there'?s|you (?:have|find)) (.+?), (?:there'?s|you (?:have|find)) (.+?)\.?$",
     r"If there is \1, there is \2."),
    # timing and process
    (r"(?i)^(?:the )?early (.+?) (?:gets?|catches?) (?:the )?(.+?)\.?$", r"Early action brings \2."),
    (r"(?i)^(?:practice|repetition) makes? (?:perfect|improvement)\.?$", r"Practice improves skill."),
    (r"(?i)^time (?:is|equals?) (?:money|wealth|value)\.?$", r"Time has value."),
    # cooperation and collective action
    (r"(?i)^(?:many|multiple|several) hands (?:make|create) (?:light work|easy work)\.?$",
     r"Cooperation reduces effort."),
    (r"(?i)^(?:together|unity) (?:we|is) (?:stand|strength)(?:, (?:divided|apart) (?:we )?fall)?\.?$",
     r"Unity creates strength."),
    # excess and moderation
    (r"(?i)^too (?:many|much) (.+?) (?:spoils?|ruins?) (?:the )?(.+?)\.?$", r"Excess of \1 harms \2."),
    (r"(?i)^(?:all work|only work) and no play makes? (.+?) (?:a )?(?:dull|boring) (?:boy|person)\.?$",
     r"Balance work and rest."),
    # patience and haste
    (r"(?i)^(?:haste|rushing|hurrying) (?:makes|creates|causes) (?:waste|mistakes|errors)\.?$",
     r"Haste causes problems."),
    (r"(?i)^(?:slow and )?steady wins (?:the )?race\.?$", r"Consistency beats speed."),
    (r"(?i)^(?:good things|patience) (?:comes?|rewards) (?:to )?(?:those who|people who) wait\.?$",
     r"Patience brings rewards."),
]

_COMPILED = [(re.compile(p), r) for p, r in STRUCTURAL_RULES]


def canonicalize(s):
    t = str(s).strip().strip(QUOTES).strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[“”‘’`]", '"', t)
    t = re.sub(r"[–—]", "-", t)
    # strip framing that doesn't affect meaning
    t = re.sub(r"^(old saying:|proverb:|they say:?|it is said:?|english equivalent:?)\s*", "", t, flags=re.I)
    t = re.sub(r"\s*(- \w+ proverb|- \w+ saying)\.?$", "", t, flags=re.I)
    for rx, rep in _COMPILED:
        if rx.search(t):
            t = rx.sub(rep, t)
            break
    t = t.strip()
    if t and t[-1] not in ".!?":
        t += "."
    return t


_SUBSTITUTIONS = [
    (re.compile(r"\b(?:a|an|the)\b"), ""),
    (re.compile(r"\b(?:is|are|was|were|being)\b"), "be"),
    (re.compile(r"\b(?:will|shall|would|should|could|can|may|might)\b"), "will"),
    (re.compile(r"\b(?:does|did|done)\b"), "do"),
    (re.compile(r"\b(?:has|had)\b"), "have"),
    (re.compile(r"\b(?:person|people|man|men|woman|women|individual)\b"), "person"),
    (re.compile(r"\b(?:home|house|dwelling)\b"), "home"),
    (re.compile(r"\b(?:money|wealth|riches|fortune|gold)\b"), "wealth"),
    (re.compile(r"\b(?:friend|buddy|companion|ally)\b"), "friend"),
    (re.compile(r"\b(?:enemy|foe|opponent|rival)\b"), "enemy"),
    (re.compile(r"\b(?:work|labor|labour|effort|toil)\b"), "work"),
    (re.compile(r"\b(?:speak|talk|say|tell|utter)\b"), "speak"),
    (re.compile(r"\b(?:good|fine|excellent|wonderful)\b"), "good"),
    (re.compile(r"\b(?:bad|terrible|awful|horrible)\b"), "bad"),
]


def preprocess_for_similarity(text):
    """Lexical normalization applied before vectorization (not stored)."""
    t = str(text).strip().lower()
    for rx, rep in _SUBSTITUTIONS:
        t = rx.sub(rep, t)
    t = re.sub(r"[^\w\s\.\!\?]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()
