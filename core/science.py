"""Research-status statistics with uncertainty and plain-language interpretation.

Everything the admin screen needs to answer three questions honestly:
  1. What do the numbers actually say right now?
  2. How certain are they? (point estimates alone are not evidence)
  3. What, and how much more, is needed to reach a publishable standard?

Design notes:
  * Every headline statistic carries a confidence interval. An alpha of 0.85 on
    40 units and on 400 units are very different claims.
  * Sample-size targets are derived from the current estimate, so they answer
    "how many more of THESE do I need", not a textbook abstraction.
  * Thresholds follow the conventions reviewers apply: Krippendorff's alpha
    >= 0.80 for firm conclusions, >= 0.667 for tentative ones (Krippendorff
    2004); AUC interpreted with its interval, never a bare point estimate.
"""
import math
import random
from collections import Counter, defaultdict

import numpy as np

from .annotation_quality import krippendorff_alpha_ordinal, aggregate_constraints

ALPHA_FIRM = 0.80        # Krippendorff: firm conclusions
ALPHA_TENTATIVE = 0.667  # Krippendorff: tentative conclusions only
TARGET_UNITS = 150       # double-rated pairs we consider a comfortable IAA base
TARGET_PAIRS = 500       # Pelican revision plan: 500-1000 consensus pairs


# --------------------------------------------------------------- reliability
def _latest_per_user(constraints):
    """One score per (pair, annotator): the latest. Mirrors the consensus engine."""
    latest = {}
    for c in constraints:
        s = c.get("score")
        if s is None:
            s = {"must": 4, "cannot": 0}.get(c.get("label"))
        if s is None:
            continue
        key = (tuple(sorted((int(c["a_id"]), int(c["b_id"])))), c.get("user") or "(anon)")
        latest[key] = int(s)
    return latest


def alpha_with_ci(constraints, B=400, seed=0):
    """Krippendorff's alpha (ordinal) with a bootstrap CI resampled over UNITS.

    Returns dict: alpha, lo, hi, n_units, verdict, meaning, needed_units.
    """
    alpha, n_units = krippendorff_alpha_ordinal(constraints)
    out = {"alpha": alpha, "n_units": n_units, "lo": None, "hi": None,
           "needed_units": None}
    if alpha is None or n_units < 2:
        out.update(verdict="not computable",
                   meaning="Not enough pairs rated by two or more people. "
                           "Agreement cannot be measured until annotators overlap.")
        return out

    # group the raw records by unit so we can resample whole units
    by_unit = defaultdict(list)
    for c in constraints:
        by_unit[tuple(sorted((int(c["a_id"]), int(c["b_id"])))) ].append(c)
    units = [u for u, recs in by_unit.items() if len({r.get("user") for r in recs}) >= 2]

    rng = random.Random(seed)
    boots = []
    for _ in range(B):
        pick = [units[rng.randrange(len(units))] for _ in range(len(units))]
        recs = []
        for k, u in enumerate(pick):
            # re-key each resampled unit so duplicates count as separate units
            for r in by_unit[u]:
                r2 = dict(r)
                r2["a_id"], r2["b_id"] = 10**9 + k, 10**9 + k + 1
                r2["_orig"] = u
                recs.append(r2)
        a, _ = krippendorff_alpha_ordinal(recs)
        if a is not None:
            boots.append(a)
    if boots:
        out["lo"], out["hi"] = (round(float(np.percentile(boots, 2.5)), 3),
                                round(float(np.percentile(boots, 97.5)), 3))

    # how many double-rated units would put the LOWER bound above 0.80?
    # CI half-width shrinks ~ 1/sqrt(n), so n_needed = n * (half / target_half)^2
    if out["lo"] is not None and alpha > ALPHA_FIRM:
        half = (out["hi"] - out["lo"]) / 2
        room = alpha - ALPHA_FIRM
        if half > room > 0:
            out["needed_units"] = int(math.ceil(n_units * (half / room) ** 2))

    if alpha >= ALPHA_FIRM:
        if out["lo"] is not None and out["lo"] >= ALPHA_FIRM:
            v = "firm"
            m = (f"Agreement is solid: alpha = {alpha:.3f}, and even the pessimistic end "
                 f"of the interval ({out['lo']:.3f}) clears the 0.80 line reviewers use "
                 f"for firm conclusions. This supports publication as it stands.")
        else:
            v = "good but wide"
            m = (f"The estimate is good (alpha = {alpha:.3f}, above the 0.80 line) but "
                 f"uncertain: the interval reaches down to {out['lo']:.3f}, because it "
                 f"rests on only {n_units} pairs rated by two or more people. Reporting "
                 f"alpha with this interval is normal practice and defensible. Note the "
                 f"arithmetic: since alpha sits only {alpha - ALPHA_FIRM:.3f} above 0.80, "
                 f"pushing the whole interval above the line needs a very tight estimate, "
                 f"so aim first at {TARGET_UNITS} double-rated pairs for a materially "
                 f"narrower interval. More overlap, not more annotators, is what tightens it.")
    elif alpha >= ALPHA_TENTATIVE:
        v = "tentative"
        m = (f"alpha = {alpha:.3f} sits between 0.667 and 0.80: enough for tentative "
             f"conclusions, not for firm claims. Reviewers will notice. Increase the "
             f"number of pairs judged by two or more people.")
    else:
        v = "insufficient"
        m = (f"alpha = {alpha:.3f} is below 0.667. Annotators are not agreeing enough "
             f"for the labels to carry weight. Before collecting more, check whether the "
             f"guidelines are clear and whether one rater is out of step.")
    out.update(verdict=v, meaning=m)
    return out


# --------------------------------------------------------------- separation
def auc_with_ci(sims, labels):
    """AUC with a Hanley-McNeil standard error (1982), plus interpretation.

    sims: similarity per pair; labels: 1 = humans said same idea, 0 = different.
    """
    sims, labels = np.asarray(sims, float), np.asarray(labels, int)
    n1, n0 = int((labels == 1).sum()), int((labels == 0).sum())
    out = {"auc": None, "lo": None, "hi": None, "n_pos": n1, "n_neg": n0,
           "needed_pairs": None}
    if n1 < 2 or n0 < 2:
        out.update(verdict="not computable",
                   meaning="Needs at least a couple of pairs on each side.")
        return out
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(sims[labels == 1], sims[labels == 0], alternative="greater")
    A = float(u / (n1 * n0))
    q1, q2 = A / (2 - A), 2 * A * A / (1 + A)
    se = math.sqrt(max(0.0, (A * (1 - A) + (n1 - 1) * (q1 - A * A)
                            + (n0 - 1) * (q2 - A * A)) / (n1 * n0)))
    lo, hi = max(0.0, A - 1.96 * se), min(1.0, A + 1.96 * se)
    out.update(auc=round(A, 4), lo=round(lo, 3), hi=round(hi, 3),
               se=round(se, 4), p=float(p))

    # pairs needed for the lower bound to clear 0.80 (SE shrinks ~1/sqrt(n))
    if A > 0.80 and lo < 0.80:
        room = (A - 0.80) / 1.96
        if se > room > 0:
            factor = (se / room) ** 2
            out["needed_pairs"] = int(math.ceil((n1 + n0) * factor))

    if lo > 0.5:
        strength = ("strong" if A >= 0.85 else "moderate" if A >= 0.75 else "weak but real")
        out["verdict"] = strength
        out["meaning"] = (
            f"AUC = {A:.3f} (95% CI {lo:.3f} to {hi:.3f}), p = {p:.1e}. Read it as: given "
            f"one pair humans called the same idea and one they called different, the "
            f"method ranks them correctly about {A*100:.0f}% of the time. The interval "
            f"stays above 0.5, so the effect is real, not noise. Evidence is {strength}.")
    else:
        out["verdict"] = "inconclusive"
        out["meaning"] = (
            f"AUC = {A:.3f}, but the interval ({lo:.3f} to {hi:.3f}) touches 0.5, which is "
            f"chance. With {n1 + n0} pairs there is not yet enough evidence that the "
            f"method separates same from different.")
    return out


# --------------------------------------------------------------- annotators
def annotator_profile(constraints):
    """Per-annotator volume and how concentrated the dataset is.

    Concentration matters: if one person supplies most judgments, 'consensus'
    largely means that person, and independent agreement is an illusion.
    """
    _pairs, annot = aggregate_constraints(constraints)
    total = sum(a["n"] for a in annot.values()) or 1
    rows = [{"uid": u, "judgments": a["n"], "share": a["n"] / total,
             "reliability": round(a["reliability"], 3)}
            for u, a in sorted(annot.items(), key=lambda kv: -kv[1]["n"])]
    top_share = rows[0]["share"] if rows else 0.0
    if top_share > 0.6:
        v, m = ("concentrated",
                f"One annotator provides {top_share*100:.0f}% of all judgments. Consensus "
                f"largely reflects that single person, which reviewers treat as a validity "
                f"threat. Recruit or rebalance so no one exceeds roughly half.")
    elif top_share > 0.4:
        v, m = ("acceptable",
                f"The largest contributor holds {top_share*100:.0f}% of judgments. Workable, "
                f"but more balance would strengthen the independence claim.")
    else:
        v, m = ("balanced",
                f"No annotator dominates (largest share {top_share*100:.0f}%), so agreement "
                f"reflects genuinely independent judgments.")
    # anyone systematically out of step?
    outliers = [r for r in rows if r["judgments"] >= 20 and r["reliability"] < 0.5]
    return {"rows": rows, "n_annotators": len(rows), "top_share": top_share,
            "verdict": v, "meaning": m, "outliers": outliers}


def overlap_stats(constraints):
    """How much of the data is double-rated: the fuel for every agreement statistic."""
    latest = _latest_per_user(constraints)
    per_pair = Counter(pair for pair, _u in latest)
    n_pairs = len(per_pair)
    multi = sum(1 for c in per_pair.values() if c >= 2)
    rate = multi / n_pairs if n_pairs else 0.0
    if multi >= TARGET_UNITS:
        v, m = ("sufficient",
                f"{multi} pairs carry two or more independent judgments. That is a "
                f"comfortable base for reporting agreement.")
    else:
        v, m = ("needs more",
                f"Only {multi} of {n_pairs} pairs ({rate*100:.0f}%) have been judged by two "
                f"or more people. Agreement statistics rest entirely on these. Serving "
                f"already-judged pairs more often is the cheapest way to raise alpha's "
                f"precision, and needs roughly {max(0, TARGET_UNITS - multi)} more.")
    return {"n_pairs": n_pairs, "multi_rated": multi, "overlap_rate": rate,
            "verdict": v, "meaning": m}


def stratum_balance(pairs_meta):
    """Balance across sampling strata (family x region x similarity band).

    pairs_meta: iterable of stratum labels, one per judged pair.
    Uses normalised entropy: 1.0 = perfectly even, low = concentrated in a few cells.
    """
    counts = Counter(s for s in pairs_meta if s)
    if len(counts) < 2:
        return {"n_strata": len(counts), "evenness": None, "verdict": "n/a",
                "meaning": "Not enough stratum information recorded yet."}
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    H = -sum(p * math.log(p) for p in probs) / math.log(len(counts))
    if H >= 0.85:
        v, m = ("balanced", f"Judgments are spread evenly across {len(counts)} strata "
                            f"(evenness {H:.2f}). Coverage claims are defensible.")
    elif H >= 0.6:
        v, m = ("uneven", f"Coverage across {len(counts)} strata is somewhat uneven "
                          f"(evenness {H:.2f}). Some culture and similarity combinations "
                          f"are thin; results may not generalise equally.")
    else:
        v, m = ("skewed", f"Judgments concentrate in a few of {len(counts)} strata "
                          f"(evenness {H:.2f}). Conclusions mostly describe those cells.")
    return {"n_strata": len(counts), "evenness": round(H, 2), "verdict": v, "meaning": m,
            "counts": dict(counts.most_common())}


# --------------------------------------------------------------- readiness
def readiness(alpha_res, auc_res, overlap_res, annot_res, n_consensus_pairs):
    """Traffic-light checklist against what reviewers will actually look for."""
    def light(ok, warn):
        return "green" if ok else ("amber" if warn else "red")

    checks = []
    a = alpha_res.get("alpha")
    checks.append({
        "check": "Inter-annotator agreement (Krippendorff's alpha)",
        "value": "n/a" if a is None else f"{a:.3f}"
                 + (f" [{alpha_res['lo']}, {alpha_res['hi']}]" if alpha_res.get("lo") is not None else ""),
        "target": ">= 0.80 including the lower bound",
        "status": light(a is not None and alpha_res.get("lo") is not None
                        and alpha_res["lo"] >= ALPHA_FIRM,
                        a is not None and a >= ALPHA_TENTATIVE),
        "action": (alpha_res.get("meaning") or "")})
    checks.append({
        "check": "Double-rated pairs (the base for agreement)",
        "value": f"{overlap_res['multi_rated']}",
        "target": f">= {TARGET_UNITS}",
        "status": light(overlap_res["multi_rated"] >= TARGET_UNITS,
                        overlap_res["multi_rated"] >= TARGET_UNITS // 3),
        "action": overlap_res["meaning"]})
    checks.append({
        "check": "Consensus pairs (dataset size)",
        "value": f"{n_consensus_pairs}",
        "target": f">= {TARGET_PAIRS}",
        "status": light(n_consensus_pairs >= TARGET_PAIRS,
                        n_consensus_pairs >= TARGET_PAIRS // 2),
        "action": (f"{max(0, TARGET_PAIRS - n_consensus_pairs)} more consensus pairs to reach "
                   f"the {TARGET_PAIRS} agreed in the revision plan.")})
    auc = auc_res.get("auc")
    checks.append({
        "check": "Method separates same from different (AUC)",
        "value": "n/a" if auc is None else f"{auc:.3f}"
                 + (f" [{auc_res['lo']}, {auc_res['hi']}]" if auc_res.get("lo") is not None else ""),
        "target": "interval entirely above 0.5",
        "status": light(auc is not None and auc_res.get("lo", 0) > 0.6,
                        auc is not None and auc_res.get("lo", 0) > 0.5),
        "action": auc_res.get("meaning", "")})
    checks.append({
        "check": "Independence of annotators",
        "value": f"{annot_res['n_annotators']} people, largest share "
                 f"{annot_res['top_share']*100:.0f}%",
        "target": "no one above ~50%, at least 3 active",
        "status": light(annot_res["n_annotators"] >= 3 and annot_res["top_share"] <= 0.5,
                        annot_res["n_annotators"] >= 2),
        "action": annot_res["meaning"]})

    greens = sum(1 for c in checks if c["status"] == "green")
    reds = sum(1 for c in checks if c["status"] == "red")
    if reds == 0 and greens >= 4:
        overall = ("Ready to write up. The evidence base meets the standards reviewers "
                   "apply; remaining work is presentational.")
    elif reds == 0:
        overall = ("Nearly there. Nothing is broken, but some numbers are thinner than "
                   "they should be. The amber rows below say exactly what to add.")
    else:
        overall = ("Not yet publishable as a confirmatory claim. The red rows are the "
                   "blockers; each one names what it needs.")
    return {"checks": checks, "overall": overall,
            "score": f"{greens}/{len(checks)} green"}
