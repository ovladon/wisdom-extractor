import argparse, json, re, pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from packaging import version
import sklearn

def canonicalize(s):
    t = str(s).strip().strip('"“”‘’')
    rules = [
        (r"(?i)^better (.+?) than (.+)$", r"Prefer \1 over \2."),
        (r"(?i)^(no|never|don’t|do not|cannot|can’t|avoid)\s+(.+)$", r"Avoid \2."),
        (r"(?i)^where there’?s (.+?), there’?s (.+)$", r"If there is \1, there is \2."),
        (r"(?i)^if (.+?), (.+)$", r"If \1, then \2."),
        (r"(?i)^many hands (.+)$", r"Cooperation \1."),
        (r"(?i)^too many (.+)$", r"Excess of \1 is harmful."),
        (r"(?i)^practice makes perfect$", r"Practice improves skill."),
        (r"(?i)^time is money$", r"Time has economic value."),
    ]
    for pat, rep in rules:
        if re.search(pat, t):
            t = re.sub(pat, rep, t)
            break
    t = re.sub(r"\s+", " ", t).strip()
    if t and t[-1] not in ".!?":
        t += "."
    return t

def compute_coords(claims, n_components=2, random_state=0):
    from sklearn.decomposition import TruncatedSVD, PCA
    try:
        import umap
        use_umap = True
    except Exception:
        use_umap = False
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1)
    X = vec.fit_transform(claims)
    svd = TruncatedSVD(n_components=min(50, max(2, min(X.shape)-1)), random_state=random_state)
    Xr = svd.fit_transform(X)
    if use_umap:
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        emb = reducer.fit_transform(Xr)
    else:
        emb = PCA(n_components=n_components, random_state=random_state).fit_transform(Xr)
    return emb

def run(csv_path, out_json, out_csv, coords_csv, distance_threshold=0.35):
    df = pd.read_csv(csv_path).dropna(subset=["people","saying"])
    df["basis"] = df.get("english_equivalent", "").fillna("")
    mask = df["basis"].astype(str).str.strip().eq("")
    df.loc[mask, "basis"] = df["saying"].astype(str)
    df = df[df["basis"].astype(str).str.strip()!=""]
    df["claim"] = df["basis"].apply(canonicalize)

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1)
    X = vec.fit_transform(df["claim"])
    sim = cosine_similarity(X)
    dist = 1 - sim

    kwargs = dict(linkage="average", distance_threshold=distance_threshold, n_clusters=None)
    skver = version.parse(sklearn.__version__)
    try:
        if skver >= version.parse("1.4"):
            clust = AgglomerativeClustering(metric='precomputed', **kwargs).fit(dist)
        else:
            clust = AgglomerativeClustering(affinity='precomputed', **kwargs).fit(dist)
    except TypeError:
        try:
            clust = AgglomerativeClustering(affinity='precomputed', **kwargs).fit(dist)
        except TypeError:
            clust = AgglomerativeClustering(metric='precomputed', **kwargs).fit(dist)

    df["cluster_id"] = clust.labels_

    out = []
    for cid, sub in df.groupby("cluster_id"):
        claim = sub["claim"].mode().iloc[0]
        coverage = sub["people"].nunique()
        support = len(sub)
        cultures = sorted(sub["people"].astype(str).unique().tolist())
        ex = sub.groupby("people")["saying"].apply(lambda s: s.iloc[0]).to_dict()
        score = coverage + 0.3*support
        out.append({"cluster_id": int(cid), "claim": claim, "wisdom_score": round(score,3),
                    "coverage": int(coverage), "support": int(support),
                    "cultures": cultures, "examples": ex})
    out = sorted(out, key=lambda r: (-r["wisdom_score"], -r["coverage"], -r["support"]))
    df_out = pd.DataFrame(out)
    df_out.to_csv(out_csv, index=False)
    with open(out_json,"w",encoding="utf-8") as f: json.dump(out,f,ensure_ascii=False,indent=2)

    try:
        emb = compute_coords(df_out["claim"].tolist(), n_components=2)
        df_coords = df_out[["cluster_id","claim","coverage","support","wisdom_score"]].copy()
        df_coords["x"] = emb[:,0]; df_coords["y"] = emb[:,1]
        df_coords.to_csv(coords_csv, index=False)
    except Exception as e:
        df_coords = pd.DataFrame(columns=["cluster_id","claim","coverage","support","wisdom_score","x","y"])
        df_coords.to_csv(coords_csv, index=False)
        print(f"[WARN] coord computation failed: {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="proverbs_clean_v2.csv")
    ap.add_argument("--out_json", default="wisdom_clusters.json")
    ap.add_argument("--out_csv", default="clusters.csv")
    ap.add_argument("--coords_csv", default="clusters_coords.csv")
    ap.add_argument("--distance_threshold", type=float, default=0.35)
    args = ap.parse_args()
    run(args.csv, args.out_json, args.out_csv, args.coords_csv, args.distance_threshold)
