#!/usr/bin/env python3
"""Inter-annotator agreement report (Pelican revision, priority #1).

Reads a Wisdom Lab database and produces the statistics and tables the paper's
IAA subsection needs: per-annotator volumes, raw exact and binarized agreement on
multi-annotated pairs, ordinal Krippendorff's alpha, quadratic-weighted Cohen's kappa
for the two most prolific annotators, and a disagreement-examples table categorised
along the boundary types of the graded scheme.

Usage:  WISDOM_DB_PATH=/path/wisdom.db python scripts/iaa_report.py [--out DIR]
"""
import argparse, csv, os, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.persistence import init_db, list_constraints, list_proverbs
from core.annotation_quality import krippendorff_alpha_ordinal, _pair_key

SCALE = [-1, 0, 1, 2, 3, 4]
NAME = {4: "same rule", 3: "same advice", 2: "same theme",
        1: "related/diff lesson", 0: "unrelated", -1: "contradictory"}


def boundary_type(s1, s2):
    lo, hi = sorted((s1, s2))
    if {lo, hi} == {2, 3}: return "functional vs thematic (advice/theme boundary)"
    if {lo, hi} == {3, 4}: return "strict vs functional equivalence"
    if {lo, hi} == {1, 2}: return "thematic vs merely related"
    if {lo, hi} == {0, 1}: return "related vs unrelated"
    if lo == -1:           return "contradiction perception"
    if hi - lo >= 3:       return "large gap (possible error / ambiguity)"
    return "adjacent-level boundary"


def binarize(s):  # hard-link semantics: >=3 same idea, <=1 different, 2 excluded
    if s >= 3: return 1
    if s <= 1: return 0
    return None


def score_of(c):
    if c.get("score") is not None: return int(c["score"])
    return {"must": 4, "cannot": 0}.get(c.get("label"))


def weighted_kappa(pairs_ab):
    """Quadratic-weighted Cohen's kappa for rater pairs [(sa, sb), ...] on SCALE."""
    if len(pairs_ab) < 2: return None
    k = len(SCALE); idx = {v: i for i, v in enumerate(SCALE)}
    w = [[1 - ((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)]
    obs = [[0.0] * k for _ in range(k)]
    for sa, sb in pairs_ab:
        obs[idx[sa]][idx[sb]] += 1
    n = len(pairs_ab)
    ra = [sum(row) for row in obs]
    rb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    po = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k)) / n
    pe = sum(w[i][j] * ra[i] * rb[j] for i in range(k) for j in range(k)) / (n * n)
    return (po - pe) / (1 - pe) if pe != 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Bring an older database up to the current schema first, as the other scripts do.
    # Without this the report crashes on any archived or pre-migration copy rather than
    # upgrading it in place, which is exactly the copy someone reproducing a published
    # number would be holding.
    init_db()

    cons = [c for c in list_constraints() if score_of(c) is not None]
    texts = {p["id"]: (p.get("gloss") or p["text"]) for p in list_proverbs(excluded=True)}

    # One score per (pair, annotator): the latest, matching the consensus engine and
    # overlap_stats. A pair the same person judged twice is one opinion, not two, so
    # counting judgment rows would report more independent agreement than exists —
    # this report previously said 174 "multi-annotated" pairs where alpha, the release
    # trigger and the Admin panel all said 107.
    ordered = sorted(cons, key=lambda c: (c.get("created_at") or 0))
    latest = {}
    for c in ordered:
        latest[(_pair_key(c["a_id"], c["b_id"]), c.get("user") or "?")] = score_of(c)

    by_pair = defaultdict(list)   # pair -> [(score, user)], one entry per annotator
    for (pair, u), s in latest.items():
        by_pair[pair].append((s, u))

    multi = {p: v for p, v in by_pair.items() if len(v) >= 2}   # >= 2 DISTINCT annotators
    repeats = len(cons) - len(latest)   # rows superseded by the same annotator re-judging
    exact = sum(1 for v in multi.values() if len({s for s, _ in v}) == 1)
    bin_agree = bin_total = 0
    for v in multi.values():
        bs = [binarize(s) for s, _ in v]
        bs = [b for b in bs if b is not None]
        if len(bs) >= 2:
            bin_total += 1
            bin_agree += 1 if len(set(bs)) == 1 else 0

    alpha, n_units = krippendorff_alpha_ordinal(cons)

    vol = Counter((c.get("user") or "?") for c in cons)   # judgments made, per annotator
    top2 = [u for u, _ in vol.most_common(2)]
    shared = []
    for p, v in multi.items():
        d = {u: s for s, u in v}
        if len(top2) == 2 and top2[0] in d and top2[1] in d:
            shared.append((d[top2[0]], d[top2[1]]))
    wk = weighted_kappa(shared)

    # disagreement examples
    rows = []
    for p, v in sorted(multi.items()):
        scores = sorted({s for s, _ in v})
        if len(scores) > 1:
            rows.append({
                "text_a": texts.get(p[0], "?")[:100], "text_b": texts.get(p[1], "?")[:100],
                "scores": "/".join(f"{s} ({NAME[s]})" for s in scores),
                "gap": scores[-1] - scores[0],
                "boundary": boundary_type(scores[0], scores[-1]),
            })
    rows.sort(key=lambda r: -r["gap"])
    with open(os.path.join(args.out, "iaa_disagreements.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text_a", "text_b", "scores", "gap", "boundary"])
        w.writeheader(); w.writerows(rows)

    cat = Counter(r["boundary"] for r in rows)
    report = {
        "annotations_total": len(cons),
        "annotators": dict(vol),
        "pairs_annotated": len(by_pair),
        "pairs_double_rated": len(multi),      # >= 2 distinct annotators; equals alpha_units
        "repeat_judgments_superseded": repeats,
        "raw_exact_agreement": round(exact / len(multi), 4) if multi else None,
        "binarized_agreement(>=3 vs <=1)": round(bin_agree / bin_total, 4) if bin_total else None,
        "krippendorff_alpha_ordinal": round(alpha, 4) if alpha is not None else None,
        "alpha_units": n_units,
        "weighted_kappa_top2": round(wk, 4) if wk is not None else None,
        "kappa_shared_pairs": len(shared),
        "disagreements": len(rows),
        "disagreement_categories": dict(cat),
    }
    import json
    with open(os.path.join(args.out, "iaa_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\ndisagreement examples -> {os.path.join(args.out, 'iaa_disagreements.csv')}")


if __name__ == "__main__":
    main()
