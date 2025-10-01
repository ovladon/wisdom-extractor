import numpy as np

def _norm(x): 
    return float(np.clip(x, 0, 1))

def survival_score(cluster_rows, weights=None):
    if weights is None:
        weights = dict(coverage=0.35, independence=0.25, temporal=0.20, stability=0.20)
    langs = len(set([r.get('language') for r in cluster_rows if r.get('language')]))
    fams  = len(set([r.get('family') for r in cluster_rows if r.get('family')]))
    regs  = len(set([r.get('region') for r in cluster_rows if r.get('region')]))
    coverage = np.mean([_norm(langs/20.0), _norm(fams/8.0), _norm(regs/6.0)])
    independence = _norm(fams/8.0)
    years = [(r.get('last_seen') or r.get('first_seen') or 0) - (r.get('first_seen') or 0) for r in cluster_rows]
    temporal = _norm((max(years) if years else 0)/500.0)
    stability = _norm(np.mean([r.get('cohesion',0.0) for r in cluster_rows]) if cluster_rows else 0.0)
    S = (weights['coverage']*coverage + weights['independence']*independence + weights['temporal']*temporal + weights['stability']*stability)
    return float(np.clip(S, 0, 1))
