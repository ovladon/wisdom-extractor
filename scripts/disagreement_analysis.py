#!/usr/bin/env python3
"""Contested pairs: locating and characterising judge disagreement.

Disagreement is analysed as a measurement in its own right rather than discarded as
noise: the spread of independent judgments over a pair indicates how determinate the
equivalence is. This module ranks double-rated pairs by spread and tests whether the
contested ones differ systematically from the agreed ones.

Outputs:
  contested pairs  - every double-rated pair, ranked by spread of judgments
  characterisation - how contested pairs differ from agreed ones (family, region,
                     lexical similarity, score level, gloss length)

Usage:
  WISDOM_DB_PATH=<db> python scripts/disagreement_analysis.py [--out contested.csv]
"""
import argparse, csv, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.persistence import init_db, list_proverbs, list_constraints


def latest_per_user(constraints):
    """One score per (pair, judge): the most recent, mirroring the consensus engine."""
    latest = {}
    for c in constraints:
        s = c.get("score")
        if s is None:
            s = {"must": 4, "cannot": 0}.get(c.get("label"))
        if s is None:
            continue
        key = (tuple(sorted((int(c["a_id"]), int(c["b_id"])))), c.get("user") or "(anon)")
        latest[key] = int(s)
    per_pair = defaultdict(dict)
    for (pair, user), s in latest.items():
        per_pair[pair][user] = s
    return per_pair


def analyse(min_judges=2):
    init_db()
    rows = {r["id"]: r for r in list_proverbs(excluded=False)}
    per_pair = latest_per_user(list_constraints())

    # lexical similarity of the canonical claims, to test whether contested pairs
    # are simply the ones the surface method finds hard
    from core.clustering import vectorize
    from sklearn.metrics.pairwise import cosine_similarity
    ids = [i for i in rows if rows[i].get("claim")]
    X, _ = vectorize([str(rows[i]["claim"]) for i in ids])
    pos = {pid: k for k, pid in enumerate(ids)}

    recs = []
    for (a, b), votes in per_pair.items():
        if len(votes) < min_judges or a not in rows or b not in rows:
            continue
        vs = list(votes.values())
        spread = max(vs) - min(vs)
        ra, rb = rows[a], rows[b]
        sim = None
        if a in pos and b in pos:
            sim = round(float(cosine_similarity(X[pos[a]], X[pos[b]])[0, 0]), 3)
        recs.append({
            "a_id": a, "b_id": b, "n_judges": len(vs),
            "scores": "|".join(str(v) for v in sorted(vs, reverse=True)),
            "spread": spread, "mean": round(sum(vs) / len(vs), 2),
            "similarity": sim,
            "same_family": int(bool(ra.get("family")) and ra.get("family") == rb.get("family")),
            "same_region": int(bool(ra.get("region")) and ra.get("region") == rb.get("region")),
            "people_a": ra.get("people"), "people_b": rb.get("people"),
            "gloss_a": (ra.get("gloss") or ra.get("text") or "")[:160],
            "gloss_b": (rb.get("gloss") or rb.get("text") or "")[:160],
        })
    recs.sort(key=lambda r: (-r["spread"], -r["n_judges"]))
    return recs


def characterise(recs, contested_from=2):
    """Do contested pairs differ systematically from agreed ones?"""
    agreed = [r for r in recs if r["spread"] <= 1]
    contested = [r for r in recs if r["spread"] >= contested_from]
    out = {"n_double_rated": len(recs), "n_agreed": len(agreed),
           "n_contested": len(contested)}
    if not (agreed and contested):
        return out

    def avg(rs, k):
        vals = [r[k] for r in rs if r[k] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    out["similarity_agreed"] = avg(agreed, "similarity")
    out["similarity_contested"] = avg(contested, "similarity")
    out["same_family_agreed"] = avg(agreed, "same_family")
    out["same_family_contested"] = avg(contested, "same_family")
    out["same_region_agreed"] = avg(agreed, "same_region")
    out["same_region_contested"] = avg(contested, "same_region")
    out["gloss_len_agreed"] = round(
        sum(len(r["gloss_a"]) + len(r["gloss_b"]) for r in agreed) / (2 * len(agreed)), 1)
    out["gloss_len_contested"] = round(
        sum(len(r["gloss_a"]) + len(r["gloss_b"]) for r in contested) / (2 * len(contested)), 1)

    # is the contested/agreed split independent of cross-family status?
    try:
        from scipy.stats import fisher_exact, mannwhitneyu
        tab = [[sum(r["same_family"] for r in contested),
                len(contested) - sum(r["same_family"] for r in contested)],
               [sum(r["same_family"] for r in agreed),
                len(agreed) - sum(r["same_family"] for r in agreed)]]
        odds, p = fisher_exact(tab)
        out["family_effect_odds"] = round(float(odds), 3)
        out["family_effect_p"] = float(p)
        sa = [r["similarity"] for r in agreed if r["similarity"] is not None]
        sc = [r["similarity"] for r in contested if r["similarity"] is not None]
        if sa and sc:
            u, p2 = mannwhitneyu(sc, sa, alternative="two-sided")
            out["similarity_diff_p"] = float(p2)
    except Exception:
        pass

    # where on the scale does disagreement live?
    zones = defaultdict(int)
    for r in contested:
        m = r["mean"]
        zone = ("clearly different (<=1)" if m <= 1 else
                "boundary (1-2.5)" if m <= 2.5 else
                "boundary (2.5-3)" if m <= 3 else "clearly same (>3)")
        zones[zone] += 1
    out["contested_by_zone"] = dict(zones)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    recs = analyse()
    stats = characterise(recs)

    print(f"double-rated pairs: {stats['n_double_rated']}  "
          f"| agreed (spread<=1): {stats['n_agreed']}  "
          f"| contested (spread>=2): {stats['n_contested']}")
    print("\nMost contested pairs:")
    for r in recs[:args.top]:
        if r["spread"] < 2:
            break
        print(f"\n  spread {r['spread']} (scores {r['scores']}), similarity {r['similarity']}, "
              f"{r['people_a']} / {r['people_b']}")
        print(f"    A: {r['gloss_a'][:105]}")
        print(f"    B: {r['gloss_b'][:105]}")

    print("\n--- do contested pairs differ from agreed ones? ---")
    for k in ("similarity_agreed", "similarity_contested", "same_family_agreed",
              "same_family_contested", "same_region_agreed", "same_region_contested",
              "gloss_len_agreed", "gloss_len_contested", "family_effect_odds",
              "family_effect_p", "similarity_diff_p"):
        if k in stats:
            print(f"  {k:26} {stats[k]}")
    if "contested_by_zone" in stats:
        print("  contested pairs by consensus zone:")
        for z, n in sorted(stats["contested_by_zone"].items(), key=lambda kv: -kv[1]):
            print(f"     {z:26} {n}")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0]))
            w.writeheader(); w.writerows(recs)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
