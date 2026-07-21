#!/usr/bin/env python3
"""Empirical validation of the reliability-weighted consensus model (Pelican revision).

Synthetic ground-truth experiment: pairs with true binary labels are annotated by a mix
of honest annotators (accuracy p) and adversaries (random or inverting). We compare
consensus accuracy of the reliability-weighted model against unweighted majority vote,
and check that estimated reliabilities separate honest from adversarial annotators.

Usage: python scripts/reliability_validation.py [--out DIR]
"""
import argparse, csv, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.annotation_quality import aggregate_constraints

random.seed(42)


def simulate(n_pairs, votes_per_pair, honest, adversaries, adv_kind):
    truth = {i: random.choice(["must", "cannot"]) for i in range(n_pairs)}
    annotators = [(f"h{i}", p, "honest") for i, p in enumerate(honest)] + \
                 [(f"a{i}", None, adv_kind) for i in range(adversaries)]
    cons = []
    for pid, t in truth.items():
        for (name, p, kind) in random.sample(annotators, min(votes_per_pair, len(annotators))):
            if kind == "honest":
                lab = t if random.random() < p else ("cannot" if t == "must" else "must")
            elif kind == "random":
                lab = random.choice(["must", "cannot"])
            else:  # inverter
                lab = "cannot" if t == "must" else "must"
            cons.append({"a_id": pid, "b_id": pid + 100000, "label": lab, "user": name})
    return truth, cons, {n: k for n, _, k in annotators}


def majority_accuracy(truth, cons):
    from collections import defaultdict, Counter
    votes = defaultdict(list)
    for c in cons:
        votes[c["a_id"]].append(c["label"])
    ok = tot = 0
    for pid, v in votes.items():
        top = Counter(v).most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            continue  # tie: undecided
        tot += 1
        ok += 1 if top[0][0] == truth[pid] else 0
    return ok / tot if tot else None, tot


def weighted_accuracy(truth, cons):
    pairs, annotators = aggregate_constraints(cons)
    ok = tot = 0
    for p in pairs:
        if p["label"] is None:
            continue
        tot += 1
        ok += 1 if p["label"] == truth[p["a_id"]] else 0
    return (ok / tot if tot else None), tot, annotators


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="."); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    scenarios = [
        ("clean: 3 honest p=.9",            [0.9, 0.9, 0.9], 0, "random"),
        ("noisy: 3 honest p=.75",           [0.75, 0.75, 0.75], 0, "random"),
        ("1 random adversary vs 2 honest",  [0.9, 0.9], 1, "random"),
        ("1 inverter vs 2 honest",          [0.9, 0.9], 1, "inverter"),
        ("2 inverters vs 3 honest",         [0.9, 0.9, 0.85], 2, "inverter"),
        ("outnumbered: 2 honest vs 3 rand", [0.9, 0.9], 3, "random"),
    ]
    rows = []
    for name, honest, adv, kind in scenarios:
        truth, cons, kinds = simulate(400, 3, honest, adv, kind)
        macc, mn = majority_accuracy(truth, cons)
        wacc, wn, annotators = weighted_accuracy(truth, cons)
        hrel = [v["reliability"] for k, v in annotators.items() if kinds[k] == "honest"]
        arel = [v["reliability"] for k, v in annotators.items() if kinds[k] != "honest"]
        rows.append({
            "scenario": name,
            "majority_acc": round(macc, 4) if macc else None,
            "weighted_acc": round(wacc, 4) if wacc else None,
            "gain": round(wacc - macc, 4) if (macc and wacc) else None,
            "mean_rel_honest": round(sum(hrel) / len(hrel), 3) if hrel else None,
            "mean_rel_adversary": round(sum(arel) / len(arel), 3) if arel else None,
        })
        print(rows[-1])
    with open(os.path.join(a.out, "reliability_validation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\n-> {os.path.join(a.out, 'reliability_validation.csv')}")


if __name__ == "__main__":
    main()
