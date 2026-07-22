"""Annotation aggregation and confidence (v19.1).

Turns raw pairwise annotations (possibly many per pair, from many annotators, possibly
conflicting) into per-pair consensus labels with confidence scores, via a lightweight
Dawid–Skene-style reliability model:

1. majority vote per unordered pair;
2. annotator reliability = smoothed agreement of their votes with pair majorities;
3. pair confidence = reliability-weighted vote share for the majority label
   (iterated a few times so reliabilities and majorities stabilize).

Pairs whose confidence falls below a threshold — or that are outright tied — are
"disputed": they are excluded from clustering constraints and served back to
annotators for re-annotation. More annotations therefore monotonically improve both
the constraint set and the evaluation set.
"""
from collections import defaultdict

PRIOR_CORRECT = 0.7    # Laplace-style prior reliability for annotators with little history
PRIOR_WEIGHT = 2.0     # pseudo-votes behind the prior
DISPUTED_BELOW = 0.65  # consensus confidence under which a pair counts as disputed


def _pair_key(a, b):
    return (a, b) if a <= b else (b, a)


def aggregate_constraints(constraints, iterations=3):
    """Ordinal (Pelican-scale) consensus with annotator reliability.

    Works natively on graded scores (4..0, -1); legacy binary labels map to 4/0.
    Returns (pairs, annotators):
      pairs: {a_id, b_id, consensus_score, label, n, votes_must, votes_cannot,
              agreement, confidence, disputed}
        label: 'must' if consensus >= 2.5, 'cannot' if <= 1.5, else None (theme zone)
      annotators: {user: {n, reliability}}  — reliability = smoothed ordinal closeness
        of the annotator's votes to consensus (1 = always on it, 0 = maximally far).
    """
    # keep only each annotator's LATEST vote per pair: repeated serves of the same
    # pair must update, not multiply, that person's voice
    latest = {}
    for c in constraints:
        sc = c.get("score")
        if sc is None:
            sc = {"must": 4, "cannot": 0}.get(c.get("label"))
        if sc is None:
            continue
        latest[(_pair_key(int(c["a_id"]), int(c["b_id"])), c.get("user") or "(anon)")] = float(sc)
    votes = defaultdict(list)   # pair -> [(score, user)]
    for (pair, user), sc in latest.items():
        votes[pair].append((sc, user))

    HALF = 2.5   # half of the -1..4 range; closeness = 1 - |vote-consensus|/HALF (floored 0)
    reliability = defaultdict(lambda: PRIOR_CORRECT)
    consensus, dispersion = {}, {}
    for _ in range(max(1, iterations)):
        for pair, vs in votes.items():
            wsum = vsum = 0.0
            for sc, user in vs:
                w = max(0.05, reliability[user])
                wsum += w; vsum += w * sc
            m = vsum / wsum
            d = (sum(max(0.05, reliability[u]) * (sc - m) ** 2 for sc, u in vs) / wsum) ** 0.5
            consensus[pair], dispersion[pair] = m, d
        agree = defaultdict(float); count = defaultdict(float)
        for pair, vs in votes.items():
            m = consensus[pair]
            for sc, user in vs:
                count[user] += 1.0
                agree[user] += max(0.0, 1.0 - abs(sc - m) / HALF)
        for user in count:
            reliability[user] = ((agree[user] + PRIOR_CORRECT * PRIOR_WEIGHT)
                                 / (count[user] + PRIOR_WEIGHT))

    pairs = []
    for pair, vs in sorted(votes.items()):
        m, d = consensus[pair], dispersion[pair]
        label = "must" if m >= 2.5 else ("cannot" if m <= 1.5 else None)
        conf = max(0.0, 1.0 - d / 5.0)          # d=0 -> 1; full-range split -> 0.5
        n_must = sum(1 for sc, _ in vs if sc >= 3)
        n_cannot = sum(1 for sc, _ in vs if sc <= 1)
        raw_agreement = max(n_must, n_cannot, len(vs) - n_must - n_cannot) / len(vs)
        pairs.append({
            "a_id": pair[0], "b_id": pair[1],
            "consensus_score": round(m, 3),
            "label": label, "n": len(vs),
            "votes_must": n_must, "votes_cannot": n_cannot,
            "agreement": round(raw_agreement, 4),
            "confidence": round(conf, 4),
            "disputed": conf < DISPUTED_BELOW,
        })
    annotators = {u: {"n": int(count.get(u, 0)), "reliability": round(reliability[u], 4)}
                  for u in ({user for _, user in sum(votes.values(), [])})}
    return pairs, annotators


def constraint_pairs_for_clustering(pairs, min_confidence=0.6, min_votes=1):
    """Filter aggregated pairs into (must_pairs, cannot_pairs) for cluster_texts()."""
    must, cannot = [], []
    for p in pairs:
        if p["disputed"] or p["label"] is None:
            continue
        if p["confidence"] < min_confidence or p["n"] < min_votes:
            continue
        (must if p["label"] == "must" else cannot).append((p["a_id"], p["b_id"]))
    return must, cannot


def pairs_needing_review(pairs, target_votes=3, max_confidence=0.8):
    """Pairs worth re-annotating: disputed, low-confidence, or under-voted.

    Sorted most-valuable-first (disputed, then lowest confidence, then fewest votes).
    """
    review = [p for p in pairs
              if p["disputed"] or p["confidence"] < max_confidence or p["n"] < target_votes]
    review.sort(key=lambda p: (not p["disputed"], p["confidence"], p["n"]))
    return review


def krippendorff_alpha_ordinal(constraints, scale=(-1, 0, 1, 2, 3, 4)):
    """Krippendorff's alpha (ordinal metric) over graded pair scores.

    Units = unordered pairs with >= 2 scored annotations. Returns (alpha, n_units)
    or (None, n_units) when undefined (fewer than 2 multi-annotated units or no
    variance). Legacy binary labels are mapped must->4, cannot->0 when score is
    missing, so mixed old/new data still yields a defined statistic.
    """
    from collections import defaultdict
    # latest vote per (annotator, pair): self-repeats must not masquerade as
    # inter-annotator agreement
    latest = {}
    for c in constraints:
        s = c.get("score")
        if s is None:
            s = {"must": 4, "cannot": 0}.get(c.get("label"))
        if s is None:
            continue
        latest[(_pair_key(int(c["a_id"]), int(c["b_id"])), c.get("user") or "(anon)")] = int(s)
    vals = defaultdict(list)
    for (pair, _user), s in latest.items():
        vals[pair].append(s)
    units = {k: v for k, v in vals.items() if len(v) >= 2}
    if len(units) < 2:
        return None, len(units)
    levels = sorted(scale)
    # ordinal distance: squared difference of cumulative ranks
    def dist(a, b):
        ia, ib = levels.index(a), levels.index(b)
        return float((ia - ib) ** 2)
    # observed disagreement
    Do_num, Do_den = 0.0, 0.0
    all_vals = []
    for v in units.values():
        m = len(v)
        all_vals.extend(v)
        for i in range(m):
            for j in range(m):
                if i != j:
                    Do_num += dist(v[i], v[j])
        Do_den += m * (m - 1)
    if Do_den == 0:
        return None, len(units)
    Do = Do_num / Do_den
    # expected disagreement from pooled values
    n = len(all_vals)
    De_num = sum(dist(a, b) for a in all_vals for b in all_vals if a is not b) \
        if False else sum(dist(all_vals[i], all_vals[j])
                          for i in range(n) for j in range(n) if i != j)
    De = De_num / (n * (n - 1)) if n > 1 else 0.0
    if De == 0:
        return None, len(units)
    return 1.0 - Do / De, len(units)
