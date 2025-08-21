# diagnostics_v2.py — corrected proverb-level diagnostics for Wisdom Extractor v7.7
# Run:  python diagnostics_v2.py   (in the folder with the CSVs and extractor.py)
import gc, json, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from scipy.stats import spearmanr
from extractor import canonicalize  # reuse YOUR canonicalization, no reimplementation

RESULTS = {}

def load_claims(path="proverbs_clean_v2.csv"):
    df = pd.read_csv(path)
    RESULTS["clean_rows_total"] = int(len(df))                     # compare vs 18,049
    df = df.dropna(subset=["people", "saying"])
    df["basis"] = df.get("english_equivalent", "").fillna("")
    mask = df["basis"].astype(str).str.strip().eq("")
    df.loc[mask, "basis"] = df["saying"].astype(str)
    df = df[df["basis"].astype(str).str.strip() != ""]
    df["claim"] = df["basis"].apply(canonicalize)
    RESULTS["rows_entering_clustering"] = int(len(df))
    return df.reset_index(drop=True)

def cluster(df, tau):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          min_df=1, dtype=np.float32)
    X = vec.fit_transform(df["claim"])
    sim = cosine_similarity(X, dense_output=True).astype(np.float32)
    np.subtract(1.0, sim, out=sim)                                 # dist, in place
    np.fill_diagonal(sim, 0.0)
    try:
        cl = AgglomerativeClustering(linkage="average", metric="precomputed",
                                     distance_threshold=tau, n_clusters=None).fit(sim)
    except TypeError:
        cl = AgglomerativeClustering(linkage="average", affinity="precomputed",
                                     distance_threshold=tau, n_clusters=None).fit(sim)
    labels = cl.labels_
    del sim; gc.collect()
    return X, labels

def top10_by_coverage(df, labels):
    d = df.assign(cluster_id=labels)
    g = d.groupby("cluster_id").agg(claim=("claim", lambda s: s.mode().iloc[0]),
                                    coverage=("people", "nunique"),
                                    support=("claim", "size"))
    return g.sort_values("coverage", ascending=False).head(10)

# ---- main run at tau = 0.35 ----
df = load_claims()
X, labels = cluster(df, 0.35)
RESULTS["n_clusters_tau_0.35"] = int(len(set(labels)))             # compare vs 13,576
RESULTS["sum_support_equals_rows"] = bool(len(labels) == RESULTS["rows_entering_clustering"])

rng = np.random.default_rng(0)
n = X.shape[0]
samp = min(5000, n)
RESULTS["silhouette_proverb_level_sampled"] = float(
    silhouette_score(X, labels, metric="cosine",
                     sample_size=samp, random_state=0))
comp_vals = []
for lab in np.unique(labels):
    idx = np.where(labels == lab)[0]
    if len(idx) >= 2:
        D = cosine_distances(X[idx])
        comp_vals.append(D[np.triu_indices_from(D, k=1)].mean())
RESULTS["compactness_mean_intracluster_cosine_dist"] = float(np.mean(comp_vals))
RESULTS["clusters_with_2plus_members"] = int(len(comp_vals))

top10 = top10_by_coverage(df, labels)
RESULTS["top10_coverage_tau_0.35"] = top10["coverage"].tolist()    # compare vs 31,30,27...
RESULTS["top10_claims"] = top10["claim"].tolist()

# ---- honest stability: drop 10% of PROVERBS, recluster, match top-10 by claim ----
keep = rng.choice(n, size=int(n * 0.9), replace=False)
df_sub = df.iloc[keep].reset_index(drop=True)
_, labels_sub = cluster(df_sub, 0.35)
top10_sub = set(top10_by_coverage(df_sub, labels_sub)["claim"])
RESULTS["stability_top10_retained"] = float(
    len(set(top10["claim"]) & top10_sub) / 10.0)

# ---- tau = 0.25 comparison (resolves the 0.35 vs 0.25 question with data) ----
_, labels25 = cluster(df, 0.25)
RESULTS["n_clusters_tau_0.25"] = int(len(set(labels25)))
RESULTS["top10_coverage_tau_0.25"] = top10_by_coverage(df, labels25)["coverage"].tolist()

# ---- Spearman theme x proxy (your logic, n disclosed) ----
from interpret_v2 import theme_of
meta = pd.read_csv("people_metadata_v2.csv")
d = df.assign(cluster_id=labels)
g = d.groupby("cluster_id").agg(claim=("claim", lambda s: s.mode().iloc[0]),
                                cultures=("people", lambda s: sorted(set(s))))
rows = [{"people": p, "theme": theme_of(r.claim)}
        for r in g.itertuples() for p in r.cultures]
T = pd.DataFrame(rows).groupby(["people", "theme"]).size().unstack(fill_value=0)
M = meta.set_index("people").join(T, how="left").fillna(0)
corr = {}
for prox in ["maritime_orientation", "urbanization_level",
             "individualism_proxy", "uncertainty_avoidance_proxy"]:
    if prox in M.columns:
        s = pd.to_numeric(M[prox].map({"Low": 0, "Med": 1, "High": 2}), errors="coerce")
        for col in T.columns:
            rho, p = spearmanr(s, M[col], nan_policy="omit")
            corr[f"{col}~{prox}"] = {"rho": round(float(rho), 3),
                                     "p": round(float(p), 4),
                                     "n": int(s.notna().sum())}
RESULTS["spearman_theme_x_proxy"] = corr

with open("diagnostics_v2_results.json", "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, ensure_ascii=False, indent=2)
print(json.dumps(RESULTS, ensure_ascii=False, indent=2))
