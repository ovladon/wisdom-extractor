"""Validation metrics: everything the paper reports, plus constraint-based evaluation.

- silhouette (cosine), on a stratified sample for speed
- bootstrap stability (mean Adjusted Rand Index over subsamples) — paper's protocol
- tau sensitivity sweep — reproduces the paper's sensitivity table
- coverage distribution and operational bins (universal / regional / culture-specific)
- permutation triangulation: cluster diversity (families, regions) vs random mixing
- NEW: constraint agreement — human must/cannot annotations scored against clustering,
  turning the Wisdom Lab annotation game into a real evaluation set.
"""
import numpy as np
import pandas as pd

from .clustering import vectorize, cluster_texts


def silhouette_cosine(texts, labels, sample=3000, seed=0):
    from sklearn.metrics import silhouette_score
    labels = np.asarray(labels)
    n = len(texts)
    if n < 3 or len(set(labels.tolist())) < 2:
        return None
    idx = np.arange(n)
    if n > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, sample, replace=False)
    sub_labels = labels[idx]
    if len(set(sub_labels.tolist())) < 2 or len(set(sub_labels.tolist())) >= len(idx):
        return None
    X, _ = vectorize([texts[i] for i in idx])
    try:
        return float(silhouette_score(X, sub_labels, metric="cosine"))
    except Exception:
        return None


def bootstrap_stability(texts, ids, tau, iterations=5, frac=0.8, seed=0, agglo_limit=4000):
    """Mean ARI between full-run labels (restricted) and labels from reclustered subsamples."""
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed)
    full_labels, _ = cluster_texts(texts, ids, tau=tau, agglo_limit=agglo_limit)
    n = len(texts)
    scores = []
    for _ in range(iterations):
        idx = rng.choice(n, int(n * frac), replace=False)
        sub_texts = [texts[i] for i in idx]
        sub_ids = [ids[i] for i in idx]
        sub_labels, _ = cluster_texts(sub_texts, sub_ids, tau=tau, agglo_limit=agglo_limit)
        scores.append(adjusted_rand_score(full_labels[idx], sub_labels))
    return float(np.mean(scores)) if scores else None


def sensitivity_sweep(texts, ids, taus=(0.25, 0.30, 0.35, 0.40, 0.45),
                      sample=3000, seed=0, agglo_limit=4000):
    """Paper's sensitivity table: clusters / silhouette / stability per tau (on a sample)."""
    rng = np.random.default_rng(seed)
    n = len(texts)
    idx = np.arange(n) if n <= sample else rng.choice(n, sample, replace=False)
    sub_texts = [texts[i] for i in idx]
    sub_ids = [ids[i] for i in idx]
    rows = []
    for tau in taus:
        labels, method = cluster_texts(sub_texts, sub_ids, tau=tau, agglo_limit=agglo_limit)
        sil = silhouette_cosine(sub_texts, labels, sample=len(sub_texts))
        ari = bootstrap_stability(sub_texts, sub_ids, tau, iterations=3, agglo_limit=agglo_limit)
        rows.append({"tau": tau, "n": len(sub_texts), "n_clusters": int(len(set(labels.tolist()))),
                     "silhouette_cosine": sil, "bootstrap_ARI_mean": ari, "method": method})
    return pd.DataFrame(rows)


def coverage_distribution(cluster_summary):
    """Distribution + operational bins over the per-cluster coverage column."""
    cov = cluster_summary["coverage"]
    return {
        "clusters": int(len(cov)),
        "coverage_mean": float(cov.mean()) if len(cov) else 0.0,
        "coverage_p90": float(cov.quantile(0.9)) if len(cov) else 0.0,
        "coverage_p99": float(cov.quantile(0.99)) if len(cov) else 0.0,
        "coverage_max": int(cov.max()) if len(cov) else 0,
        "universal_ge20": int((cov >= 20).sum()),
        "regional_10_19": int(((cov >= 10) & (cov < 20)).sum()),
        "culture_specific_lt10": int((cov < 10).sum()),
    }


def constraint_agreement(labels_by_id, constraints):
    """Score clustering against human annotations.

    must-link pair correct  <=> same cluster
    cannot-link pair correct <=> different clusters
    Returns accuracies + violation lists (ids) for review.
    """
    res = {"must_total": 0, "must_satisfied": 0, "cannot_total": 0, "cannot_satisfied": 0,
           "must_violations": [], "cannot_violations": []}
    for c in constraints:
        a, b, label = c["a_id"], c["b_id"], c["label"]
        la, lb = labels_by_id.get(a), labels_by_id.get(b)
        if la is None or lb is None:
            continue
        if label == "must":
            res["must_total"] += 1
            if la == lb:
                res["must_satisfied"] += 1
            else:
                res["must_violations"].append((a, b))
        elif label == "cannot":
            res["cannot_total"] += 1
            if la != lb:
                res["cannot_satisfied"] += 1
            else:
                res["cannot_violations"].append((a, b))
    res["must_accuracy"] = res["must_satisfied"] / res["must_total"] if res["must_total"] else None
    res["cannot_accuracy"] = res["cannot_satisfied"] / res["cannot_total"] if res["cannot_total"] else None
    return res


def permutation_triangulation(df, top_n=25, perms=200, seed=0):
    """Diversity of top clusters (distinct families / regions) vs random-mixing baseline.

    Percentile ~50 means the cluster is no more diverse than chance (genealogy/areal mixing);
    high percentiles suggest diffusion or convergence beyond proximity. Paper's protocol.
    """
    rng = np.random.default_rng(seed)
    d = df.dropna(subset=["cluster_id"]).copy()
    if d.empty:
        return pd.DataFrame()
    fam_pool = d["family"].fillna("?").to_numpy()
    reg_pool = d["region"].fillna("?").to_numpy()
    sizes = d.groupby("cluster_id").size()
    top = (d.groupby("cluster_id")["people"].nunique()
             .sort_values(ascending=False).head(top_n).index)
    rows = []
    for cid in top:
        sub = d[d["cluster_id"] == cid]
        k = len(sub)
        obs_f = sub["family"].fillna("?").nunique()
        obs_r = sub["region"].fillna("?").nunique()
        null_f = np.array([len(set(rng.choice(fam_pool, k, replace=False))) for _ in range(perms)])
        null_r = np.array([len(set(rng.choice(reg_pool, k, replace=False))) for _ in range(perms)])
        rows.append({
            "cluster_id": int(cid), "size": int(k),
            "distinct_cultures": int(sub["people"].nunique()),
            "distinct_families": int(obs_f), "distinct_regions": int(obs_r),
            "families_percentile_vs_random": float((null_f < obs_f).mean() * 100 + (null_f == obs_f).mean() * 50),
            "regions_percentile_vs_random": float((null_r < obs_r).mean() * 100 + (null_r == obs_r).mean() * 50),
        })
    return pd.DataFrame(rows).sort_values("families_percentile_vs_random", ascending=False)
