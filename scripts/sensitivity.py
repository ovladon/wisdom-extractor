#!/usr/bin/env python3
"""Tau sensitivity sweep — robustness evidence for the clustering threshold.

For each tau, clusters a fixed sample (always including every annotated proverb)
and scores agreement with human consensus: fraction of same-idea (must) pairs that
land in one cluster, and of different (cannot) pairs kept apart.

Usage: WISDOM_DB_PATH=<db> python scripts/sensitivity.py [--sample 4000] [--out csv]
"""
import argparse, csv, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.persistence import init_db, list_proverbs, list_constraints
from core.annotation_quality import aggregate_constraints
from core.clustering import cluster_texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--taus", default="0.25,0.30,0.35,0.40,0.45,0.50")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_db()
    rows = {r["id"]: r for r in list_proverbs(with_claims_only=True)}
    agg, _ = aggregate_constraints(list_constraints())
    must = [(p["a_id"], p["b_id"]) for p in agg if p["label"] == "must"
            and p["a_id"] in rows and p["b_id"] in rows]
    cannot = [(p["a_id"], p["b_id"]) for p in agg if p["label"] == "cannot"
              and p["a_id"] in rows and p["b_id"] in rows]
    keep = {i for pr in must + cannot for i in pr}
    pool = [i for i in rows if i not in keep]
    random.seed(42)
    ids = sorted(keep) + random.sample(pool, max(0, min(args.sample - len(keep), len(pool))))
    texts = [str(rows[i]["claim"]) for i in ids]

    results = []
    for tau in (float(t) for t in args.taus.split(",")):
        labels, method = cluster_texts(texts, ids, tau=tau)   # UNCONSTRAINED on purpose:
        lab = dict(zip(ids, labels.tolist()))                 # measures raw geometry vs humans
        m_ok = sum(1 for a, b in must if lab[a] == lab[b])
        c_ok = sum(1 for a, b in cannot if lab[a] != lab[b])
        acc = (m_ok + c_ok) / max(1, len(must) + len(cannot))
        results.append({"tau": tau, "clusters": len(set(labels.tolist())),
                        "must_linked": round(m_ok / max(1, len(must)), 3),
                        "cannot_split": round(c_ok / max(1, len(cannot)), 3),
                        "constraint_accuracy": round(acc, 3), "method": method})
        print(results[-1])
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0]))
            w.writeheader(); w.writerows(results)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
