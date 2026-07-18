"""Noise filtering and quality scoring (paper's cleaning pipeline, hardened).

keep(x) implements the paper's rule:
    keep(x) = [len >= 6] AND (NOT name_like OR proverb_cues) AND NOT boilerplate
plus boilerplate patterns learned from real failures in the v16-v18 databases
(e.g. Gutenberg transcriber notes ended up as "proverbs" in wisdom.db).
"""
import re

BOILERPLATE_RX = [
    re.compile(r"transcriber'?s?\s+notes?", re.I),
    re.compile(r"\bISBN\b|\bOCLC\b|\bdoi:", re.I),
    re.compile(r"retrieved from|this page was last edited|creative commons", re.I),
    re.compile(r"^\s*(see also|references|external links|notes|bibliography|further reading|contents)\b", re.I),
    re.compile(r"^\s*(chapter|section|volume|part)\s+[ivxlc\d]+\b.{0,20}$", re.I),
    re.compile(r"https?://|www\.", re.I),
    re.compile(r"print and punctuation errors", re.I),
    re.compile(r"^\s*\[?\d+\]?\s*$"),
    re.compile(r"proverb of the month", re.I),
    # web-nav / ad debris seen on scraped pages
    re.compile(r"\b(one-to-one|lessons? (with|online)|like a native|phrasebook|learn \w+ (online|fast)|"
               r"click here|sign up|subscribe|privacy policy|cookie|omniglot|advertisement)\b", re.I),
    # source-book headers: "Nathan Bailey (1721). Divers Proverbs…", "Thomas Fuller, Gnomologia (1732)"
    re.compile(r"^[A-Z][\w'’.]+\s+[A-Z][\w'’.]+\s*[,(].{0,80}\(1[4-9]\d\d\)"),
    re.compile(r"\b(Gnomologia|Almanack|Paroemiologia|Adagia|Divers Proverbs|Introductio ad prudentiam)\b"),
    re.compile(r"^\s*translation:?\s*$", re.I),
    re.compile(r"wikipedia|wikiquote|wiktionary", re.I),
]

ADVICE_RX = re.compile(
    r"\b(never|always|don'?t|do not|should|must|avoid|beware|better|if|when|who|he that|he who|those who)\b", re.I)


_CITATION_RX = [
    # trailing "Author, Name; Other (1875). "Title" . Werk . pp. 358-359." after a sentence
    re.compile(r"^(.*?[.!?])\s+(?:(?:in|von|van|de|la|d[ei])\s+)?[A-ZÄÖÜ][\w'’\-]+,\s+[A-Z].{0,200}\(\d{4}\).*$", re.S),
    re.compile(r"^(.*?[.!?])\s+(?:(?:in|von|van|de|la|d[ei])\s+)?[A-ZÄÖÜ][\w'’\-]+,\s+[A-Z].{0,300}?\bpp?\.\s*\d+.*$", re.S),
    re.compile(r"^(.*?[.!?])\s+[A-Z][^.!?]{0,80}\bpp?\.\s*\d+.*$", re.S),
    re.compile(r"^(.*?[.!?])\s*\[\d+\]\s*$", re.S),
]


_YEAR_PAREN_RX = re.compile(r"\((1[4-9]\d\d|20[0-2]\d)\)")
_YEAR_TAIL_RX = re.compile(r"[,;]\s*(1[4-9]\d\d|20[0-2]\d)\b")


def extract_attestation_year(t):
    """Earliest year found in a citation context: '(1875)' anywhere, or ', 1857' in a tail.

    Interpreted as an 'attested no later than' bound — the year of the source that
    printed the proverb, not the proverb's origin. Returns int year or None.
    """
    t = str(t)
    years = [int(m) for m in _YEAR_PAREN_RX.findall(t)]
    years += [int(m) for m in _YEAR_TAIL_RX.findall(t)]
    return min(years) if years else None


def strip_citations(t):
    """Cut trailing bibliographic attributions (common in Wikiquote/Gutenberg scrapes)."""
    t = str(t).strip()
    for rx in _CITATION_RX:
        m = rx.match(t)
        if m and len(m.group(1)) >= 10:
            t = m.group(1).strip()
    return t


def name_like(t):
    toks = t.split()
    if len(toks) > 3:
        return False
    title_toks = sum(1 for w in toks if w[:1].isupper())
    has_verb = ADVICE_RX.search(t) or re.search(r"\b(is|are|makes?|has|have|comes?|goes?)\b", t, re.I)
    return title_toks == len(toks) and not has_verb


def proverb_cues(t):
    return bool(re.search(r"[,;:]", t) or ADVICE_RX.search(t))


def is_boilerplate(t):
    return any(rx.search(t) for rx in BOILERPLATE_RX)


def keep(t):
    t = str(t).strip()
    if len(t) < 6 or len(t) > 400:
        return False
    if len(t.split()) < 2:
        return False
    if is_boilerplate(t):
        return False
    if name_like(t) and not proverb_cues(t):
        return False
    return True


def quality_score(t):
    """Paper's Q(x): punctuation + advice words + length cues (0-6)."""
    t = str(t)
    q = 0
    if re.search(r"[,;:]", t):
        q += 2
    if ADVICE_RX.search(t):
        q += 2
    if len(t.split()) >= 5:
        q += 1
    if len(t) >= 20:
        q += 1
    return q
