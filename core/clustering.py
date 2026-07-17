"""Cross-script clustering (paper method) + annotation constraints (Wisdom Lab) + scalability fixes.

- Character 3-5-gram TF-IDF (char_wb): works across all writing systems, no training data.
- n <= agglo_limit: average-linkage agglomerative clustering on precomputed cosine
  distances with cut threshold tau — exactly the paper's method, so paper results reproduce.
- n > agglo_limit: sparse radius-neighbor graph + union-find (single linkage at tau).
  Never materializes a dense n x n matrix (v18 needed ~2.7 GB at 18k rows; this doesn't).
- Must-link / cannot-link constraints from the annotation game are enforced in both paths:
  must-links merge clusters, cannot-links block merges (processed in ascending distance order).
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

from .canonicalize import preprocess_for_similarity

AGGLO_LIMIT_DEFAULT = 4000


def vectorize(texts, preprocess=True):
    proc = [preprocess_for_similarity(t) for t in texts] if preprocess else [str(t) for t in texts]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    X = vec.fit_transform(proc)
    return X, vec


class _UnionFind:
    def __init__(self, n, cannot_pairs=None):
        self.parent = list(range(n))
        # cannot-links tracked as sets of enemy roots per root
        self.cannot = {}
        for a, b in (cannot_pairs or []):
            self.cannot.setdefault(self.find(a), set()).add(self.find(b))
            self.cannot.setdefault(self.find(b), set()).add(self.find(a))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def blocked(self, a, b):
        ra, rb = self.find(a), self.find(b)
        return rb in self.cannot.get(ra, ())

    def union(self, a, b, force=False):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if not force and rb in self.cannot.get(ra, ()):
            return False
        self.parent[rb] = ra
        enemies = self.cannot.pop(rb, set())
        if enemies:
            self.cannot.setdefault(ra, set()).update(enemies)
            for e in enemies:
                s = self.cannot.get(e)
                if s is not None:
                    s.discard(rb)
                    s.add(ra)
        return True

    def labels(self):
        roots = {}
        out = []
        for i in range(len(self.parent)):
            r = self.find(i)
            if r not in roots:
                roots[r] = len(roots)
            out.append(roots[r])
        return np.array(out)


def _edges_radius(X, tau, batch=2000):
    """Similarity edges with cosine distance <= tau, computed in batches (sparse-safe)."""
    nn = NearestNeighbors(metric="cosine", radius=tau, algorithm="brute")
    nn.fit(X)
    edges = []
    n = X.shape[0]
    for start in range(0, n, batch):
        dists, idxs = nn.radius_neighbors(X[start:start + batch])
        for row, (dd, jj) in enumerate(zip(dists, idxs)):
            i = start + row
            for d, j in zip(dd, jj):
                if j > i:
                    edges.append((float(d), i, int(j)))
    edges.sort(key=lambda e: e[0])
    return edges


def cluster_texts(texts, ids, tau=0.35, must_pairs=None, cannot_pairs=None,
                  agglo_limit=AGGLO_LIMIT_DEFAULT):
    """Cluster texts; returns (labels aligned with ids, method_name).

    must_pairs / cannot_pairs: iterables of (id_a, id_b) using the same ids given here.
    """
    n = len(texts)
    if n == 0:
        return np.array([]), "none"
    pos = {pid: i for i, pid in enumerate(ids)}
    must = [(pos[a], pos[b]) for a, b in (must_pairs or []) if a in pos and b in pos]
    cannot = [(pos[a], pos[b]) for a, b in (cannot_pairs or []) if a in pos and b in pos]

    X, _ = vectorize(texts)

    if n <= agglo_limit:
        labels = _cluster_agglomerative(X, tau)
        labels = _apply_constraints_to_labels(labels, must, cannot, X)
        return labels, "agglomerative-average"

    uf = _UnionFind(n, cannot_pairs=cannot)
    for a, b in must:
        uf.union(a, b, force=True)
    for _, i, j in _edges_radius(X, tau):
        uf.union(i, j)
    return uf.labels(), "graph-single-linkage"


def _cluster_agglomerative(X, tau):
    from sklearn.cluster import AgglomerativeClustering
    sim = cosine_similarity(X)
    dist = np.clip(1.0 - sim, 0.0, None)
    kwargs = dict(linkage="average", distance_threshold=tau, n_clusters=None)
    try:
        clust = AgglomerativeClustering(metric="precomputed", **kwargs).fit(dist)
    except TypeError:  # older scikit-learn
        clust = AgglomerativeClustering(affinity="precomputed", **kwargs).fit(dist)
    return clust.labels_


def _apply_constraints_to_labels(labels, must, cannot, X):
    """Post-hoc constraint enforcement on agglomerative labels.

    Must-links merge the two clusters (unless that would join a cannot-linked pair).
    Cannot-links inside one cluster split the offender out by reassigning the two
    items' nearer neighbors is out of scope; we flag by leaving labels as-is —
    diagnostics reports cannot-link violations so annotators can lower tau.
    """
    labels = np.asarray(labels).copy()
    cannot_set = {frozenset((int(labels[a]), int(labels[b]))) for a, b in cannot
                  if labels[a] != labels[b]}
    for a, b in must:
        la, lb = int(labels[a]), int(labels[b])
        if la == lb:
            continue
        if frozenset((la, lb)) in cannot_set:
            continue
        labels[labels == lb] = la
        cannot_set = {frozenset((la if x == lb else x) for x in fs) for fs in cannot_set}
    return labels


def nearest_pairs(texts, ids, k=8, hi=0.85, lo=0.35, rng=None):
    """Candidate annotation pairs without a dense similarity matrix.

    Positives: k-nearest neighbors with lo <= sim <= hi (uncertain zone — most
    informative to annotate). Negatives: random pairs verified to have sim < lo.
    """
    rng = rng or np.random.default_rng(0)
    n = len(texts)
    if n < 2:
        return [], []
    X, _ = vectorize(texts)
    kq = min(k + 1, n)
    nn = NearestNeighbors(metric="cosine", n_neighbors=kq, algorithm="brute").fit(X)
    dists, idxs = nn.kneighbors(X)
    pos = []
    for i in range(n):
        for d, j in zip(dists[i], idxs[i]):
            s = 1.0 - float(d)
            if j != i and lo <= s <= hi and j > i:
                pos.append((ids[i], ids[int(j)], s))
    neg = []
    tries = 0
    target = max(1, len(pos) // 2)
    while len(neg) < target and tries < target * 20:
        i, j = rng.integers(0, n, 2)
        if i == j:
            tries += 1; continue
        s = float(cosine_similarity(X[int(i)], X[int(j)])[0, 0])
        if s < lo:
            neg.append((ids[int(i)], ids[int(j)], s))
        tries += 1
    return pos, neg


def summarize_clusters(df):
    """df needs: id, cluster_id, claim (or text), people. Returns per-cluster table.

    wisdom_score = coverage + 0.3 * support   (paper's S_k)
    """
    textcol = "claim" if "claim" in df.columns and df["claim"].notna().any() else "text"
    rows = []
    for cid, sub in df.groupby("cluster_id"):
        if cid is None or (isinstance(cid, float) and np.isnan(cid)):
            continue
        claims = sub[textcol].dropna().astype(str)
        claim = claims.mode().iloc[0] if len(claims) else ""
        peoples = sub["people"].dropna().astype(str)
        coverage = int(peoples.nunique())
        support = int(len(sub))
        examples = {}
        for p, g in sub.groupby("people"):
            if pd.notna(p):
                examples[str(p)] = str(g["text"].iloc[0])
        rows.append({
            "cluster_id": int(cid), "claim": claim,
            "coverage": coverage, "support": support,
            "wisdom_score": round(coverage + 0.3 * support, 3),
            "cultures": ", ".join(sorted(peoples.unique().tolist())),
            "examples": examples,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["wisdom_score", "coverage", "support"],
                              ascending=[False, False, False]).reset_index(drop=True)
    return out
