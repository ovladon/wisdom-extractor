import numpy as np, pandas as pd, json, warnings
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr
def _safe_silhouette(X, labels):
    try:
        if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
            return float("nan")
        return float(silhouette_score(X, labels, metric="cosine"))
    except Exception:
        return float("nan")
def _compactness(X, labels):
    import numpy as np
    vals = []
    for lab in set(labels):
        idx = np.where(labels==lab)[0]
        if len(idx) >= 2:
            Xi = X[idx]
            from sklearn.metrics.pairwise import cosine_distances
            D = cosine_distances(Xi)
            triu = np.triu_indices_from(D, k=1)
            if triu[0].size > 0:
                vals.append(D[triu].mean())
    return float(np.mean(vals)) if vals else float("nan")
def _stability(vec, claims, labels, k=5, drop=0.1, random_state=0):
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    rng = np.random.default_rng(random_state)
    n = len(claims)
    keep_n = int(n*(1-drop))
    if keep_n < 3:
        return float("nan")
    idx = rng.choice(n, size=keep_n, replace=False)
    c2 = [claims[i] for i in idx]
    X2 = vec.fit_transform(c2)
    sim = cosine_similarity(X2); dist = 1 - sim
    try:
        cl = AgglomerativeClustering(linkage="average", metric="precomputed", distance_threshold=0.35, n_clusters=None).fit(dist)
    except TypeError:
        cl = AgglomerativeClustering(linkage="average", affinity="precomputed", distance_threshold=0.35, n_clusters=None).fit(dist)
    from collections import Counter
    tops = [c for c,_ in Counter(claims).most_common(min(k, len(claims)))]
    kept = sum(1 for t in tops if t in c2)
    return kept/float(min(k, len(claims)))
def compute_diagnostics(df_clusters: pd.DataFrame, df_coords: pd.DataFrame, meta: pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1)
    claims = df_clusters["claim"].astype(str).tolist()
    if len(claims) < 3:
        return {"silhouette": float("nan"), "compactness_mean_cosine_distance": float("nan"), "stability_drop10_topk_retained": float("nan"), "rule_of_thumb": "Too few clusters for diagnostics."}, {}
    X = vec.fit_transform(claims)
    labels = df_clusters["cluster_id"].values
    sil = _safe_silhouette(X, labels)
    comp = _compactness(X, labels)
    stab = _stability(vec, claims, labels, k=min(10, len(claims)), drop=0.1, random_state=0)
    def themes_from_claim(c):
        c = c.lower()
        if any(k in c for k in ["cooperat","many hands","together","friend","help"]):
            return "Cooperation"
        if any(k in c for k in ["avoid","never","haste","risk","beware","prudence","slow","steady"]):
            return "Prudence/Time"
        if any(k in c for k in ["honest","truth","lie","trust","deception"]):
            return "Trust/Honesty"
        if any(k in c for k in ["family","blood","home","mother","father"]):
            return "Kinship"
        if any(k in c for k in ["fortune","luck","opportun"]):
            return "Fortune"
        if any(k in c for k in ["speak","silence","word","tongue","listen"]):
            return "Speech"
        return "Other"
    df_clusters = df_clusters.copy()
    df_clusters["theme"] = df_clusters["claim"].map(themes_from_claim)
    rows = []
    for _, r in df_clusters.iterrows():
        cul = r["cultures"]
        if isinstance(cul, str):
            import ast
            try: cul = ast.literal_eval(cul)
            except Exception: cul = []
        for peep in (cul or []):
            rows.append({"people": peep, "theme": r["theme"]})
    out_corr = {}
    if rows and not meta.empty:
        theme = pd.DataFrame(rows)
        T = theme.groupby(["people","theme"]).size().unstack(fill_value=0)
        M = meta.set_index("people").join(T, how="left").fillna(0)
        for prox in ["maritime_orientation","urbanization_level","individualism_proxy","uncertainty_avoidance_proxy"]:
            if prox in M.columns:
                series = pd.to_numeric(M[prox].map({"Low":0,"Med":1,"High":2}).fillna(M[prox]), errors="coerce")
                for col in T.columns:
                    try:
                        rho, p = spearmanr(series, M[col], nan_policy="omit")
                        if rho is not None:
                            out_corr[f"{col}~{prox}"] = {"rho": float(rho), "p": float(p)}
                    except Exception:
                        pass
    trust = {
        "silhouette": sil,
        "compactness_mean_cosine_distance": comp,
        "stability_drop10_topk_retained": stab,
        "rule_of_thumb": "Silhouette ~ 0.2–0.5 is typical for short-text clustering; lower suggests mixing; higher suggests cleaner separation. Lower compactness is better (tighter clusters). Stability close to 1.0 indicates robust top claims."
    }
    return trust, out_corr
