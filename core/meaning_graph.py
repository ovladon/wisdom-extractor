import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _tfidf(texts):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
    return vec.fit_transform(texts), vec

def build_edges(df, text_col='text', paraphrase_thr=0.42):
    if len(df) == 0:
        return pd.DataFrame(columns=['u', 'v', 'rel', 'score'])
    X, _ = _tfidf(df[text_col].fillna('').astype(str).tolist())
    S = cosine_similarity(X)
    edges = []
    n = S.shape[0]
    ids = df['id'].tolist()
    for i in range(n):
        for j in range(i + 1, n):
            s = S[i, j]
            if s >= paraphrase_thr:
                edges.append((ids[i], ids[j], 'paraphrase', float(s)))
    return pd.DataFrame(edges, columns=['u', 'v', 'rel', 'score'])

def communities_from_edges(edges):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
            return parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for _, r in edges.iterrows():
        if r['rel'] != 'paraphrase':
            continue
        union(r['u'], r['v'])
    for k in list(parent.keys()):
        find(k)
    comp_id = {}
    idx = 0
    for node, root in sorted(parent.items(), key=lambda kv: kv[1]):
        if root not in comp_id:
            comp_id[root] = idx
            idx += 1
    return {node: comp_id[root] for node, root in parent.items()}

def nearest_pairs(df, text_col='text', k=10, hi=0.8, lo=0.3):
    X, _ = _tfidf(df[text_col].fillna('').astype(str).tolist())
    S = cosine_similarity(X)
    pairs_pos, pairs_neg = [], []
    n = S.shape[0]
    ids = df['id'].tolist()
    for i in range(n):
        order = np.argsort(-S[i])
        taken_pos = 0
        taken_neg = 0
        for j in order:
            if i == j:
                continue
            s = S[i, j]
            if s >= lo and s <= hi and taken_pos < k:
                pairs_pos.append((ids[i], ids[j], float(s)))
                taken_pos += 1
            if s < lo and taken_neg < max(1, k // 2):
                pairs_neg.append((ids[i], ids[j], float(s)))
                taken_neg += 1
            if taken_pos >= k and taken_neg >= max(1, k // 2):
                break
    return pairs_pos, pairs_neg
