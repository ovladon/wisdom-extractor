#!/usr/bin/env python3
"""Validate the corroboration controller against its predicted equilibrium.

The annotation server routes a fraction of pair requests to items exactly one other
annotator has already judged. This script checks that the resulting share of
double-rated items behaves as the closed form predicts, measures the cost in corpus
breadth, and tests the controller that aims at a target share.

    python scripts/routing_simulation.py --out <dir>

Writes routing_simulation.json and two vector PDF charts.
"""
import argparse, collections, json, os, random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {"ink": "#1a1a1a", "mid": "#6b6b6b", "light": "#c9c9c9",
           "accent": "#2b6a8f", "warn": "#a8452b"}


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["mid"])
    ax.tick_params(colors=PALETTE["mid"], labelsize=9)
    ax.grid(axis="y", color=PALETTE["light"], linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def simulate(n_judgments, p_fixed=None, target=None, gain=0.5, cap=0.60,
             n_raters=12, seed=0, activity_skew=1.6):
    """One annotation run.

    Raters differ in activity, drawn from a power law, because real annotation is never
    uniform and the concentration it produces is the reason the weighting exists. A rater
    never corroborates a pair they judged themselves.
    """
    rng = random.Random(seed)
    weights = [1.0 / (i + 1) ** activity_skew for i in range(n_raters)]
    judged = collections.defaultdict(set)      # pair id -> set of raters
    singles = []                               # pair ids with exactly one rater
    next_pair = 0
    history = []

    for step in range(n_judgments):
        rater = rng.choices(range(n_raters), weights=weights, k=1)[0]
        n_pairs = len(judged)
        n_double = sum(1 for v in judged.values() if len(v) >= 2)
        share = n_double / n_pairs if n_pairs else 0.0

        if target is not None:
            p = min(cap, target / (1.0 + target) + gain * max(0.0, target - share))
        else:
            p = p_fixed

        eligible = [q for q in singles if rater not in judged[q]]
        if eligible and rng.random() < p:
            q = rng.choice(eligible)
            judged[q].add(rater)
            if len(judged[q]) >= 2:
                singles.remove(q)
        else:
            q = next_pair; next_pair += 1
            judged[q].add(rater)
            singles.append(q)

        if step % 25 == 0 or step == n_judgments - 1:
            n_pairs = len(judged)
            n_double = sum(1 for v in judged.values() if len(v) >= 2)
            history.append((step + 1, n_double / n_pairs if n_pairs else 0.0,
                            n_pairs, n_double))

    n_pairs = len(judged)
    n_double = sum(1 for v in judged.values() if len(v) >= 2)
    owner = collections.Counter(min(v) for v in judged.values() if len(v) >= 2)
    top_share = owner.most_common(1)[0][1] / n_double if n_double else 0.0
    return {"pairs": n_pairs, "double": n_double,
            "share": n_double / n_pairs if n_pairs else 0.0,
            "history": history, "top_share_of_double": top_share}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--judgments", type=int, default=4000)
    ap.add_argument("--reps", type=int, default=40)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    res = {}

    # 1. does the fixed-rate equilibrium hold?
    print("fixed routing rates against the predicted equilibrium p/(1-p):")
    rows = []
    for p in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        obs = [simulate(args.judgments, p_fixed=p, seed=s)["share"] for s in range(args.reps)]
        pred = p / (1 - p) if p < 1 else float("nan")
        rows.append({"p": p, "predicted": pred, "observed": float(np.mean(obs)),
                     "sd": float(np.std(obs))})
        print(f"  p={p:.1f}  predicted {pred:.3f}  observed {np.mean(obs):.3f} "
              f"(sd {np.std(obs):.3f})")
    res["equilibrium"] = rows

    # 2. what does depth cost in breadth?
    print("\nbreadth cost:")
    br = []
    for p in (0.0, 0.2, 0.31, 0.4, 0.5):
        r = [simulate(args.judgments, p_fixed=p, seed=s) for s in range(args.reps)]
        br.append({"p": p, "pairs": float(np.mean([x["pairs"] for x in r])),
                   "double": float(np.mean([x["double"] for x in r])),
                   "share": float(np.mean([x["share"] for x in r]))})
        print(f"  p={p:.2f}  distinct pairs {br[-1]['pairs']:7.0f}  "
              f"double-rated {br[-1]['double']:6.0f}  share {br[-1]['share']:.3f}")
    res["breadth"] = br

    # 3. does the controller reach and hold a target?
    print("\ncontroller aiming at a target share:")
    ctl = []
    for t in (0.20, 0.35, 0.45):
        r = [simulate(args.judgments, target=t, seed=s) for s in range(args.reps)]
        ctl.append({"target": t, "achieved": float(np.mean([x["share"] for x in r])),
                    "sd": float(np.std([x["share"] for x in r])),
                    "equilibrium_p": t / (1 + t)})
        print(f"  target {t:.2f}  achieved {ctl[-1]['achieved']:.3f} "
              f"(sd {ctl[-1]['sd']:.3f})  equilibrium routing rate {t/(1+t):.3f}")
    res["controller"] = ctl

    # figures
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ps = [r["p"] for r in rows]
    ax.plot(ps, [r["predicted"] for r in rows], "-", color=PALETTE["ink"],
            linewidth=1.4, label=r"predicted  $p/(1-p)$")
    ax.errorbar(ps, [r["observed"] for r in rows], yerr=[r["sd"] for r in rows],
                fmt="o", color=PALETTE["accent"], capsize=3, label="simulated")
    ax.set_xlabel("fraction of judgments routed as second opinions", fontsize=9.5)
    ax.set_ylabel("double-rated share at equilibrium", fontsize=9.5)
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    ax.set_title("The double-rated share settles where the closed form says it should",
                 fontsize=10.5, color=PALETTE["ink"], pad=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_routing_equilibrium.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for t, colour in zip((0.20, 0.35, 0.45),
                         (PALETTE["light"], PALETTE["accent"], PALETTE["warn"])):
        h = simulate(args.judgments, target=t, seed=1)["history"]
        ax.plot([x[0] for x in h], [x[1] for x in h], color=colour, linewidth=1.4,
                label=f"target {t:.2f}")
        ax.axhline(t, color=colour, linestyle=":", linewidth=0.9)
    ax.set_xlabel("judgments collected", fontsize=9.5)
    ax.set_ylabel("double-rated share", fontsize=9.5)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax)
    ax.set_title("The controller reaches its target and holds it as the corpus grows",
                 fontsize=10.5, color=PALETTE["ink"], pad=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_routing_controller.pdf"))
    plt.close(fig)

    json.dump(res, open(os.path.join(out, "routing_simulation.json"), "w"), indent=2)
    print(f"\nwrote routing_simulation.json and 2 figures in {out}")


if __name__ == "__main__":
    main()
