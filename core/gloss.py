"""English gloss extraction (v19.6).

Annotators can only judge meaning they can read. Many corpus rows carry an English
translation inside the text itself ("Af óskum eru allir eins ríkir. If wishes were
horses, beggars would ride.", "Literally: …", "(KRA) To think twice, to talk once.");
others are pure English; others have no English at all. This module extracts the best
English gloss where one exists — rows without a gloss are kept in the corpus but are
NOT served to annotators.
"""
import re

# high-frequency English function/content words — cheap language cue
_COMMON_EN = {
    "the", "a", "an", "is", "are", "am", "be", "been", "was", "were", "to", "of", "and",
    "in", "it", "you", "not", "he", "she", "who", "his", "her", "its", "for", "with",
    "on", "at", "by", "from", "as", "do", "does", "did", "done", "will", "shall", "would",
    "should", "can", "cannot", "may", "must", "has", "have", "had", "but", "if", "then",
    "than", "when", "where", "what", "which", "one", "two", "no", "never", "always",
    "man", "men", "woman", "good", "bad", "better", "best", "worse", "worst", "makes",
    "make", "made", "comes", "come", "goes", "go", "without", "every", "all", "own",
    "like", "or", "so", "they", "them", "their", "there", "this", "that", "those",
    "these", "we", "us", "our", "your", "my", "me", "him", "water", "dog", "dogs",
    "god", "home", "house", "time", "day", "old", "new", "first", "last", "little",
    "much", "many", "more", "most", "other", "another", "after", "before", "over",
    "under", "up", "down", "out", "into", "away", "far", "near", "long", "great",
    "small", "eat", "eats", "know", "knows", "get", "gets", "give", "gives", "take",
    "takes", "friend", "fool", "wise", "words", "word", "eye", "hand", "head", "heart",
    "world", "life", "death", "money", "work", "thing", "things", "nothing", "something",
    "everything", "himself", "itself", "well", "too", "very", "only", "even", "also",
    "off", "about", "against", "between", "each", "own", "same", "such", "some", "any",
}

_MARKER_RX = re.compile(
    r"(?:translation|english equivalent|english|meaning|literal(?:ly)?|lit\.|i\.e\.|that is)"
    r"\s*(?:and\s+)?[:\-–—]\s*(.+)", re.I)

_PAREN_RX = re.compile(r"\([^)]*\)")
_SEG_SPLIT_RX = re.compile(r"(?<=[.!?])\s+|\s+[—–]\s+|\s*\|\s*")


def english_score(s):
    """0..1: how English does this string look?"""
    words = re.findall(r"[a-zA-Z']+", str(s).lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in _COMMON_EN)
    ascii_ratio = sum(1 for ch in s if ord(ch) < 128) / max(1, len(s))
    return (hits / len(words)) * min(1.0, ascii_ratio * 1.15)


def _clean(g):
    g = _PAREN_RX.sub(" ", g)                       # drop "(Lappajärvi…)" "(KRA)" notes
    g = re.sub(r"^\W+|\s+", lambda m: " " if m.group(0).strip() == "" or m.start() else "", g)
    g = re.sub(r"\s+", " ", g).strip(" -–—:;,")
    return g.strip()


_FRAGMENT_RX = re.compile(r"^(and|or|but|nor|which|that|who|whose|whom|because|so)\b", re.I)
_CITATION_RX = re.compile(r"\b(19|20)\d{2}\b|\b(ISBN|vol\.|pp?\.|ed\.)\b")
_IDENT = None


def _identifier():
    """Statistical language identifier, loaded once. Absent -> caller falls back."""
    global _IDENT
    if _IDENT is None:
        try:
            from langid.langid import LanguageIdentifier, model
            _IDENT = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        except Exception:
            _IDENT = False
    return _IDENT or None


def _is_english_sentence(seg, min_conf=0.90, min_score=0.10):
    """Two independent cues must agree: a trained identifier and the word-list score.
    Also rejects fragments and bibliographic lines, which score as English but are
    not sayings."""
    if len(seg.split()) < 3 or not seg[:1].isupper():
        return False
    if _FRAGMENT_RX.match(seg) or _CITATION_RX.search(seg):
        return False
    ident = _identifier()
    if ident is None:
        return english_score(seg) >= 0.22
    lang, conf = ident.classify(seg)
    return lang == "en" and conf >= min_conf and english_score(seg) >= min_score


def first_english_sentence(text):
    """The first sentence of `text` that is confidently English, or None.

    Returns one sentence, not a run of them: a gloss that trails on into commentary
    gives judges extra material that differs between them.
    """
    t = str(text or "").strip()
    if not t:
        return None
    m = _MARKER_RX.search(t)
    if m:
        g = re.split(r"(?<=[.!?])\s+", _clean(m.group(1)))[0].strip()
        if _is_english_sentence(g):
            return g
    for seg in _SEG_SPLIT_RX.split(t):
        seg = _clean(seg)
        if _is_english_sentence(seg):
            return seg if seg[-1] in ".!?" else seg + "."
    return None


def extract_gloss(text, threshold=0.22, min_words=3):
    """Return the best English gloss of `text`, or None if none can be found.

    Priority: explicit markers ("Translation: …") → whole text if English →
    best English sentence-segment inside a bilingual text.
    """
    t = str(text).strip()
    if not t:
        return None

    m = _MARKER_RX.search(t)
    if m:
        g = _clean(m.group(1))
        if len(g.split()) >= min_words and english_score(g) >= threshold:
            return g

    stripped = _clean(t)
    whole_score = english_score(stripped)

    best, best_score = None, threshold - 1e-9
    for seg in _SEG_SPLIT_RX.split(t):
        seg = _clean(seg)
        if len(seg.split()) < min_words:
            continue
        sc = english_score(seg)
        if sc > best_score:
            best, best_score = seg, sc

    # bilingual text: an inner segment that is clearly more English than the whole wins
    if best is not None and best_score > whole_score + 0.12:
        if best[-1] not in ".!?":
            best += "."
        return best[0].upper() + best[1:]
    if whole_score >= threshold and len(stripped.split()) >= min_words:
        return stripped
    if best is not None:
        if best[-1] not in ".!?":
            best += "."
        return best[0].upper() + best[1:]
    return None
