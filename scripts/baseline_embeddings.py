#!/usr/bin/env python3
"""Multilingual-embedding baseline vs char n-grams (Pelican revision, priority #4).

On human-annotated pairs, compares AUC of:
  A. char 3-5-gram TF-IDF cosine over canonical claims (the paper's method);
  B. multilingual sentence-embedding cosine over RAW texts (embeddings' home turf);
  C. multilingual sentence-embedding cosine over canonical claims.

Positive = same idea (binary must, or graded score>=3); negative = different
(binary cannot, or score<=1); graded score 2 pairs are excluded.

Usage:
  WISDOM_DB_PATH=<db> python scripts/baseline_embeddings.py [--pairs-csv matched.csv]
      [--model paraphrase-multilingual-MiniLM-L12-v2] [--out results.json]
"""
import argparse, csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.persistence import init_db, list_proverbs, list_constraints
from core.clustering import vectorize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import mannwhitneyu


def auc(sims, labs):
    sims, labs = np.array(sims), np.array(labs)
    n1, n0 = (labs == 1).sum(), (labs == 0).sum()
    if n1 == 0 or n0 == 0:
        return None, None
    u, p = mannwhitneyu(sims[labs == 1], sims[labs == 0], alternative="greater")
    return float(u / (n1 * n0)), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", help="CSV with a_id,b_id,label columns (binary set); "
                                        "default: graded pairs from the database")
    ap.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_db()
    rows = {r["id"]: r for r in list_proverbs(with_claims_only=True)}

    pairs = []
    if args.pairs_csv:
        with open(args.pairs_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                a, b = int(r["a_id"]), int(r["b_id"])
                lab = 1 if r["label"] == "must" else 0
                if a in rows and b in rows:
                    pairs.append((a, b, lab))
        source = os.path.basename(args.pairs_csv)
    else:
        for c in list_constraints():
            s = c.get("score")
            if s is None:
                s = {"must": 4, "cannot": 0}.get(c.get("label"))
            if s is None or s == 2:
                continue
            a, b = int(c["a_id"]), int(c["b_id"])
            if a in rows and b in rows:
                pairs.append((a, b, 1 if s >= 3 else 0))
        source = "graded annotations in DB (score>=3 vs <=1)"
    if not pairs:
        print("no usable pairs"); return
    labs = [l for _, _, l in pairs]

    # A: char n-gram over claims (corpus-fitted)
    ids = list(rows)
    claims = [str(rows[i]["claim"]) for i in ids]
    X, _ = vectorize(claims)
    pos = {pid: k for k, pid in enumerate(ids)}
    simA = [float(cosine_similarity(X[pos[a]], X[pos[b]])[0, 0]) for a, b, _ in pairs]

    # B/C: embeddings
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device="cpu")
    need = sorted({i for a, b, _ in pairs for i in (a, b)})
    raw = model.encode([rows[i]["text"] for i in need], normalize_embeddings=True,
                       show_progress_bar=False)
    clm = model.encode([str(rows[i]["claim"]) for i in need], normalize_embeddings=True,
                       show_progress_bar=False)
    epos = {pid: k for k, pid in enumerate(need)}
    simB = [float(np.dot(raw[epos[a]], raw[epos[b]])) for a, b, _ in pairs]
    simC = [float(np.dot(clm[epos[a]], clm[epos[b]])) for a, b, _ in pairs]

    out = {"pairs_source": source, "n_pairs": len(pairs),
           "n_pos": int(sum(labs)), "n_neg": int(len(labs) - sum(labs)),
           "model": args.model}
    for name, sims in [("char_ngram_claims", simA),
                       ("embedding_raw_texts", simB),
                       ("embedding_claims", simC)]:
        a_, p_ = auc(sims, labs)
        out[name] = {"AUC": round(a_, 4) if a_ else None,
                     "p": f"{p_:.2e}" if p_ else None}
    print(json.dumps(out, indent=2))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
