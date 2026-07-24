#!/usr/bin/env python3
"""Coverage-driven canonicalization design.

Instead of hand-guessing rules, we test a set of GENERAL proverb-shape templates
(family matchers with variable slots, no single-proverb catchers) and rank them by
how much of the corpus each one actually matches. Coverage is the success metric:
what fraction of proverbs a structural rule can reach. Whatever the rules do not
cover is the honest scope for an LLM / learned canonicalizer.

Usage: WISDOM_DB_PATH=<db> python scripts/canon_coverage.py [--examples]
"""
import argparse, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.persistence import init_db, list_proverbs

# General families only. Each: (name, regex, rewrite). Slots are (.+?) — they match
# WHOLE FAMILIES, not one saying. Ordered rough-general first.
FAMILIES = [
    ("relative-agent (he who / whoever)",
     r"(?i)^(?:he|she|they|the one|those|whoever|whosoever|who(?:so)?ever) (?:who |that )?(.+?) (?:will|shall|is|are|gets?|finds?|reaps?|pays?|loses?|wins?|must|deserves?) (.+?)[.!?]?$",
     r"Whoever \1 will \2."),
    ("equative (X is Y)",
     r"(?i)^(?:a |an |the )?(.+?) (?:is|are) (?:a |an |the )?(.+?)[.!?]?$",
     r"\1 equals \2."),
    ("conditional (if/when X, Y)",
     r"(?i)^(?:if|when|whenever) (.+?),? (?:then )?(.+?)[.!?]?$",
     r"If \1, then \2."),
    ("comparative (better X than Y)",
     r"(?i)^(?:it'?s )?(?:better|worse) (?:to )?(.+?) than (.+?)[.!?]?$",
     r"Better \1 than \2."),
    ("negative-existential (no X without Y)",
     r"(?i)^(?:there is |there'?s )?no (.+?) without (.+?)[.!?]?$",
     r"No \1 without \2."),
    ("causation (X makes/brings/breeds Y)",
     r"(?i)^(?:a |an |the )?(.+?) (?:makes?|brings?|breeds?|begets?|causes?|leads to) (?:a |an |the )?(.+?)[.!?]?$",
     r"\1 causes \2."),
    ("prohibition (never/don't X)",
     r"(?i)^(?:you should )?(?:never|do not|don'?t|avoid|do not ever) (.+?)[.!?]?$",
     r"Avoid \1."),
    ("universal (every/all X has/are Y)",
     r"(?i)^(?:every|each|all|any) (.+?) (?:has|have|is|are|must|will) (.+?)[.!?]?$",
     r"Every \1 has \2."),
    ("similarity (like X, like Y / as X so Y)",
     r"(?i)^(?:like (.+?), like (.+?)|as (.+?),? so (.+?))[.!?]?$",
     r"As \1, so \2."),
    ("sequence (X before Y / first X then Y)",
     r"(?i)^(?:first (.+?),? (?:then |before )(.+?)|(.+?) before (.+?))[.!?]?$",
     r"\1 before \2."),
    ("necessity (you can't X without Y)",
     r"(?i)^(?:you )?(?:can'?t|cannot|must not) (.+?) without (.+?)[.!?]?$",
     r"Cannot \1 without \2."),
    ("worth/value (X is worth Y)",
     r"(?i)^(?:a |an |the )?(.+?) (?:is|are) worth (.+?)[.!?]?$",
     r"\1 is worth \2."),
    ("quantity (too much/many X ...)",
     r"(?i)^too (?:much|many) (.+?) (.+?)[.!?]?$",
     r"Excess of \1 \2."),
    ("possession (X has Y)",
     r"(?i)^(?:a |an |the |every )?(.+?) (?:has|have|carries|holds) (?:a |an |its )?(.+?)[.!?]?$",
     r"\1 has \2."),
    ("existence (where there is X, there is Y)",
     r"(?i)^where (?:there'?s|there is|you (?:have|find)) (.+?),? (?:there'?s|there is|you (?:have|find)) (.+?)[.!?]?$",
     r"If there is \1, there is \2."),
    ("obligation (X must/should Y)",
     r"(?i)^(?:a |an |the |one )?(.+?) (?:must|should|ought to|has to) (.+?)[.!?]?$",
     r"\1 must \2."),
    ("privative (one X does not make a Y)",
     r"(?i)^(?:one|a single) (.+?) (?:does not|doesn'?t|do not) make (?:a |an |the )?(.+?)[.!?]?$",
     r"One \1 does not make \2."),
]


def light_clean(t):
    t = str(t).strip().strip('"“”\'')
    t = re.sub(r"^(?:translation|literally|meaning|english equivalent|equivalent)\s*[:\-]\s*",
               "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", action="store_true")
    args = ap.parse_args()
    init_db()
    rows = list_proverbs(with_claims_only=True)
    texts = [light_clean(r.get("gloss") or r.get("text") or "") for r in rows]
    texts = [t for t in texts if 3 <= len(t.split()) <= 25]   # plausible proverb length
    N = len(texts)
    compiled = [(name, re.compile(pat), rw) for name, pat, rw in FAMILIES]

    covered = [False] * N
    print(f"Corpus tested: {N} readable proverbs\n")
    print(f"{'family':42s} {'matches':>8s} {'coverage':>9s} {'new':>7s}")
    print("-" * 70)
    for name, rx, rw in compiled:
        m = [i for i, t in enumerate(texts) if rx.match(t)]
        new = sum(1 for i in m if not covered[i])
        for i in m:
            covered[i] = True
        print(f"{name:42s} {len(m):8d} {100*len(m)/N:8.1f}% {new:7d}")
        if args.examples and m:
            for i in m[:2]:
                print(f"      e.g. {texts[i][:70]!r}")
    total = sum(covered)
    print("-" * 70)
    print(f"{'CUMULATIVE (any rule matches)':42s} {total:8d} {100*total/N:8.1f}%")
    print(f"\nUncovered (needs LLM / learned canonicalizer): {N-total} = {100*(N-total)/N:.1f}%")


if __name__ == "__main__":
    main()
