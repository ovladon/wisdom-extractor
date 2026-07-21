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
    """constraints: iterable of {a_id, b_id, label ('must'|'cannot'), user}.

    Returns (pairs, annotators):
      pairs: list of {a_id, b_id, label, n, votes_must, votes_cannot,
                      agreement, confidence, disputed}
      annotators: {user: {"n": votes_cast, "reliability": r}}
    """
    votes = defaultdict(list)   # pair -> [(label, user), ...]
    for c in constraints:
        label = c.get("label")
        if label not in ("must", "cannot"):
            continue
        votes[_pair_key(int(c["a_id"]), int(c["b_id"]))].append((label, c.get("user") or "(anon)"))

    import math
    reliability = defaultdict(lambda: PRIOR_CORRECT)
    majority = {}
    for _ in range(max(1, iterations)):
        # E-step: log-odds weighted consensus (Dawid-Skene): an annotator whose
        # reliability is below 0.5 has their vote count AGAINST their choice.
        for pair, vs in votes.items():
            z = 0.0
            for label, user in vs:
                r = min(0.95, max(0.05, reliability[user]))
                z += math.log(r / (1 - r)) * (1 if label == "must" else -1)
            p_must = 1.0 / (1.0 + math.exp(-z))
            if abs(p_must - 0.5) < 1e-9:
                majority[pair] = (None, 0.5)
            else:
                lab = "must" if p_must > 0.5 else "cannot"
                majority[pair] = (lab, max(p_must, 1 - p_must))
        # M-step: annotator reliability = smoothed agreement with majorities
        agree = defaultdict(float)
        count = defaultdict(float)
        for pair, vs in votes.items():
            lab, _ = majority[pair]
            if lab is None:
                continue
            for label, user in vs:
                count[user] += 1.0
                if label == lab:
                    agree[user] += 1.0
        for user in count:
            reliability[user] = ((agree[user] + PRIOR_CORRECT * PRIOR_WEIGHT)
                                 / (count[user] + PRIOR_WEIGHT))

    pairs = []
    for pair, vs in sorted(votes.items()):
        lab, conf = majority[pair]
        n_must = sum(1 for l, _ in vs if l == "must")
        n_cannot = len(vs) - n_must
        raw_agreement = max(n_must, n_cannot) / len(vs)
        pairs.append({
            "a_id": pair[0], "b_id": pair[1],
            "label": lab, "n": len(vs),
            "votes_must": n_must, "votes_cannot": n_cannot,
            "agreement": round(raw_agreement, 4),
            "confidence": round(conf, 4),
            "disputed": lab is None or conf < DISPUTED_BELOW,
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
    vals = defaultdict(list)
    for c in constraints:
        s = c.get("score")
        if s is None:
            s = {"must": 4, "cannot": 0}.get(c.get("label"))
        if s is None:
            continue
        vals[_pair_key(int(c["a_id"]), int(c["b_id"]))].append(int(s))
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
