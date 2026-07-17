"""2D semantic map of cluster claims (paper's Results & Visualisation tab).

char n-gram TF-IDF -> TruncatedSVD(50) -> UMAP if installed, else PCA.
"""
import numpy as np

from .clustering import vectorize


def compute_coords(claims, n_components=2, random_state=0):
    from sklearn.decomposition import TruncatedSVD, PCA
    X, _ = vectorize(claims)
    k = min(50, max(2, min(X.shape) - 1))
    Xr = TruncatedSVD(n_components=k, random_state=random_state).fit_transform(X)
    try:
        import umap
        emb = umap.UMAP(n_components=n_components, random_state=random_state).fit_transform(Xr)
    except Exception:
        emb = PCA(n_components=n_components, random_state=random_state).fit_transform(Xr)
    return np.asarray(emb)
