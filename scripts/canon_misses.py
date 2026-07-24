#!/usr/bin/env python3
"""Canonicalization-miss finder — the first step of the annotations->rules loop.

Read-only. Uses the human consensus links as ground truth and finds where the
current canonicalization + similarity DISAGREES with people:

  MISS       : humans judged two proverbs the SAME idea, but their canonical
               claims are far apart (low similarity). A rule is missing that
               should bring them together -> candidate for a new rewrite rule.
  OVER-MERGE : humans judged two DIFFERENT, but their claims are near-identical.
               Surface form collapsed a real distinction -> a rule to restrain,
               or a genuinely hard pair for the method.

Usage: WISDOM_DB_PATH=<db> python scripts/canon_misses.py [--low 0.35] [--high 0.65] [--out misses.csv]
"""
import argparse, csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.persistence import init_db, list_proverbs, list_constraints
from core.annotation_quality import aggregate_constraints
from core.clustering import vectorize
from sklearn.metrics.pairwise import cosine_similarity


def find_misses(low=0.35, high=0.65):
    init_db()
    rows = {r["id"]: r for r in list_proverbs(with_claims_only=True)}
    pairs, _ = aggregate_constraints(list_constraints())
    ids = list(rows)
    if not ids:
        return [], []
    X, _ = vectorize([str(rows[i]["claim"]) for i in ids])
    pos = {pid: k for k, pid in enumerate(ids)}
    misses, over = [], []
    for p in pairs:
        a, b = p["a_id"], p["b_id"]
        if a not in pos or b not in pos:
            continue
        sim = float(cosine_similarity(X[pos[a]], X[pos[b]])[0, 0])
        rec = {"sim": round(sim, 3), "a_id": a, "b_id": b,
               "claim_a": rows[a]["claim"], "claim_b": rows[b]["claim"]}
        if p["label"] == "must" and sim < low:
            misses.append(rec)
        elif p["label"] == "cannot" and sim > high:
            over.append(rec)
    misses.sort(key=lambda r: r["sim"])
    over.sort(key=lambda r: -r["sim"])
    return misses, over


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low", type=float, default=0.35)
    ap.add_argument("--high", type=float, default=0.65)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    misses, over = find_misses(args.low, args.high)
    print(f"MISSES (judged SAME, similarity < {args.low}): {len(misses)} "
          f"— each is a candidate for a new canonicalization rule")
    for r in misses[:12]:
        print(f"  sim={r['sim']}  [{r['a_id']}] {r['claim_a'][:60]!r}  ~=  "
              f"[{r['b_id']}] {r['claim_b'][:60]!r}")
    print(f"\nOVER-MERGES (judged DIFFERENT, similarity > {args.high}): {len(over)}")
    for r in over[:8]:
        print(f"  sim={r['sim']}  [{r['a_id']}] {r['claim_a'][:60]!r}  vs  "
              f"[{r['b_id']}] {r['claim_b'][:60]!r}")
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["kind", "sim", "a_id", "b_id", "claim_a", "claim_b"])
            w.writeheader()
            for r in misses:
                w.writerow({"kind": "miss", **r})
            for r in over:
                w.writerow({"kind": "over_merge", **r})
        print("\nwrote", args.out)


if __name__ == "__main__":
    main()
