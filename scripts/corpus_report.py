#!/usr/bin/env python3
"""Full analytical report for a frozen corpus snapshot.

Takes one immutable database and computes the complete set of corpus and annotation
statistics: composition, attestation coverage, agreement and its confidence interval,
intra-rater consistency, leave-one-rater-out robustness, the distribution of disagreement
across the scale, similarity controls, threshold sensitivity, and cross-cultural coverage.

Reading everything from a single frozen file is the point. A corpus under continuous
annotation gives different answers on different days, so any set of statistics quoted
together has to come from one snapshot with a recorded fingerprint.

    python scripts/corpus_report.py --db data/frozen/corpus_YYYYMMDD.db --out <dir>

Writes a macro file and a JSON summary of every statistic, plus three vector PDF charts:
disagreement by scale level, coverage distribution, and threshold sensitivity.
"""
import argparse, collections, datetime, hashlib, json, os, statistics, subprocess, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

PALETTE = {"ink": "#1a1a1a", "mid": "#6b6b6b", "light": "#c9c9c9",
           "accent": "#2b6a8f", "warn": "#a8452b"}


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


DIGIT_WORD = str.maketrans("0123456789", "\u0000" * 10)
_WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
          "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def macro_name(key):
    """LaTeX command names may contain letters only. A digit silently ends the name and
    the rest becomes body text, which in a preamble raises 'Missing \\begin{document}'
    and, under nonstopmode, still produces a PDF with wrong values in it."""
    return "".join(_WORDS.get(ch, ch) for ch in key)


def texnum(n):
    return "{,}".join(f"{int(n):,}".split(","))


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["mid"])
    ax.tick_params(colors=PALETTE["mid"], labelsize=9)
    ax.grid(axis="y", color=PALETTE["light"], linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    db = os.path.abspath(args.db)
    out = os.path.abspath(args.out)
    figdir = os.path.join(out, "figures")
    os.makedirs(figdir, exist_ok=True)

    os.environ["WISDOM_DB_PATH"] = db
    import core.persistence as pers
    pers.DB_PATH = db
    from core.persistence import list_constraints, list_proverbs, stats, connect
    from core.science import alpha_with_ci, overlap_stats, annotator_profile
    from core.clustering import vectorize, cluster_texts
    from core.diagnostics import silhouette_cosine, bootstrap_stability
    from sklearn.metrics.pairwise import cosine_similarity

    M = {}
    st = stats()
    cons = [c for c in list_constraints() if c.get("score") is not None]
    cons.sort(key=lambda c: (c.get("created_at") or 0))
    rows = {r["id"]: r for r in list_proverbs(excluded=False)}
    con = connect()
    dated = con.execute("SELECT COUNT(*) FROM proverbs WHERE excluded=0 "
                        "AND first_seen IS NOT NULL").fetchone()[0]
    withdrawn = con.execute("SELECT COUNT(*) FROM proverbs WHERE excluded=1").fetchone()[0]
    con.close()

    active = st["proverbs"]
    print(f"corpus: {active:,} proverbs, {st['peoples']} peoples, {len(cons):,} judgments")

    # ---------------------------------------------------------------- corpus
    M.update(CorpusProverbs=texnum(active), CorpusPeoples=str(st["peoples"]),
             CorpusWithdrawn=texnum(withdrawn), CorpusDated=texnum(dated),
             CorpusDatedPct=f"{dated/active*100:.0f}")

    # European share among items carrying region metadata
    reg = collections.Counter(r.get("region") for r in rows.values() if r.get("region"))
    eur = sum(v for k, v in reg.items() if k and "europe" in str(k).lower())
    M["RegionCovered"] = texnum(sum(reg.values()))
    M["EuropeanPct"] = f"{eur/sum(reg.values())*100:.0f}" if reg else "n/a"

    # ---------------------------------------------------------------- annotation
    ov = overlap_stats(cons)
    al = alpha_with_ci(cons, B=1000)
    an = annotator_profile(cons)
    M.update(NJudgments=texnum(len(cons)), NAnnotators=str(an["n_annotators"]),
             NPairs=texnum(ov["n_pairs"]), NDoubleRated=texnum(ov["multi_rated"]),
             OverlapPct=f"{ov['overlap_rate']*100:.0f}",
             TopAnnotatorShare=f"{an['top_share']*100:.0f}",
             Alpha=f"{al['alpha']:.3f}", AlphaLo=f"{al['lo']:.3f}",
             AlphaHi=f"{al['hi']:.3f}", AlphaUnits=texnum(ov["multi_rated"]))

    shares = sorted((a["share"] for a in an["rows"]), reverse=True)
    M["EffAnnotators"] = f"{1/sum(s*s for s in shares):.1f}"

    # latest score per (pair, rater)
    latest = {}
    for c in cons:
        latest[(tuple(sorted((c["a_id"], c["b_id"]))), c["user"])] = c["score"]
    byp = collections.defaultdict(list)
    for (p, _u), s in latest.items():
        byp[p].append(s)
    dbl = {p: v for p, v in byp.items() if len(v) >= 2}

    # self-repeats -> intra-rater reliability
    seq = collections.defaultdict(list)
    for c in cons:
        seq[(tuple(sorted((c["a_id"], c["b_id"]))), c["user"])].append(c["score"])
    reps = [(v[i], v[i + 1]) for v in seq.values() if len(v) > 1 for i in range(len(v) - 1)]
    M["NSelfRepeats"] = texnum(len(reps))
    M["IntraExact"] = f"{sum(1 for a, b in reps if a == b)/len(reps)*100:.1f}"
    M["IntraWithinOne"] = f"{sum(1 for a, b in reps if abs(a-b) <= 1)/len(reps)*100:.1f}"
    M["IntraMeanDiff"] = f"{statistics.mean(abs(a-b) for a, b in reps):.2f}"

    # one rater's share of the double-rated base
    part = collections.Counter()
    for p in dbl:
        for u in {u for (pp, u) in latest if pp == p}:
            part[u] += 1
    M["TopInDoubleRated"] = f"{part.most_common(1)[0][1]/len(dbl)*100:.0f}"

    # contested by median level, each pair counted once
    bylev = collections.defaultdict(list)
    for p, v in dbl.items():
        med = statistics.median(v)
        key = int(med) if float(med).is_integer() else "split"
        bylev[key].append(max(v) - min(v))
    levels = {}
    for k in [4, 3, 2, 1, 0]:
        sp = bylev.get(k, [])
        if sp:
            levels[k] = (len(sp), sum(1 for s in sp if s > 0)/len(sp)*100, statistics.mean(sp))
    sp = bylev.get("split", [])
    M["NSplitMedian"] = str(len(sp))
    M["SplitSpread"] = f"{statistics.mean(sp):.2f}" if sp else "n/a"
    cont_all = sum(1 for v in dbl.values() if max(v) != min(v))
    M["ContestedPct"] = f"{cont_all/len(dbl)*100:.0f}"
    whole = [v for p, v in dbl.items() if float(statistics.median(v)).is_integer()]
    M["ContestedPctWhole"] = f"{sum(1 for v in whole if max(v)!=min(v))/len(whole)*100:.0f}"
    for k, (n, pct, spread) in levels.items():
        M[f"LvlN{k}"], M[f"LvlPct{k}"], M[f"LvlSpread{k}"] = str(n), f"{pct:.0f}", f"{spread:.2f}"

    # leave-one-rater-out
    vol = collections.Counter(c["user"] for c in cons)
    loo = []
    for u, n in vol.most_common(4):
        sub = [c for c in cons if c["user"] != u]
        a2, o2 = alpha_with_ci(sub, B=600), overlap_stats(sub)
        loo.append((len(sub), o2["multi_rated"], a2["alpha"], a2.get("lo"), a2.get("hi")))
    for i, (nj, nu, a2, lo2, hi2) in enumerate(loo, start=1):
        M[f"LooN{i}"], M[f"LooUnits{i}"] = texnum(nj), str(nu)
        M[f"LooAlpha{i}"] = f"{a2:.3f}" if a2 is not None else "n/a"
        M[f"LooCI{i}"] = f"{lo2:.3f}--{hi2:.3f}" if lo2 is not None else "n/a"

    # cultural distance
    def crosses(p, field):
        a, b = rows.get(p[0]), rows.get(p[1])
        if not a or not b:
            return None
        x, y = a.get(field), b.get(field)
        return None if not x or not y else x != y
    for field, tag in (("family", "Fam"), ("region", "Reg")):
        same = [max(v)-min(v) for p, v in dbl.items() if crosses(p, field) is False]
        diff = [max(v)-min(v) for p, v in dbl.items() if crosses(p, field) is True]
        if len(same) > 4 and len(diff) > 4:
            _u, pv = mannwhitneyu(diff, same, alternative="greater")
            M[f"{tag}SameN"], M[f"{tag}DiffN"] = str(len(same)), str(len(diff))
            M[f"{tag}SameSpread"] = f"{statistics.mean(same):.2f}"
            M[f"{tag}DiffSpread"] = f"{statistics.mean(diff):.2f}"
            M[f"{tag}P"] = f"{pv:.2f}"

    # routing provenance and deliberation time
    allc = list_constraints()
    by_source = collections.Counter((c.get("source") or "organic").split(":")[0] for c in allc)
    M["NCorroborated"] = texnum(by_source.get("corroborate", 0))
    M["NChallenge"] = texnum(by_source.get("challenge", 0))
    M["NOrganic"] = texnum(by_source.get("organic", 0))
    timed = sorted(c["decide_ms"] for c in allc if c.get("decide_ms"))
    M["NTimed"] = texnum(len(timed))
    M["MedianDecideSec"] = f"{timed[len(timed)//2]/1000:.1f}" if timed else "n/a"

    # conditional agreement: anchor on one rater's actual score, look at the other.
    # Not classified by any aggregate, so the estimate cannot inherit the circularity of
    # assigning a pair to the median of the very scores whose spread is being measured.
    import random as _rnd
    plist = list(dbl.values())

    def _cond(sample):
        acc = collections.defaultdict(lambda: [0, 0]); tot = [0, 0]
        for v in sample:
            for i in range(len(v)):
                for j in range(len(v)):
                    if i != j:
                        acc[v[i]][0] += 1; acc[v[i]][1] += (v[j] == v[i])
                        tot[0] += 1; tot[1] += (v[j] == v[i])
        return {k: (n, m / n) for k, (n, m) in acc.items()}, tot[1] / tot[0]

    cond, inter = _cond(plist)
    rng2 = _rnd.Random(7)
    bagg = collections.defaultdict(list); binter = []
    for _ in range(2000):
        smp = [plist[rng2.randrange(len(plist))] for _ in range(len(plist))]
        r, o = _cond(smp); binter.append(o)
        for k, (n, v) in r.items():
            bagg[k].append(v)
    M["InterExact"] = f"{inter*100:.1f}"
    M["InterExactLo"] = f"{np.percentile(binter,2.5)*100:.0f}"
    M["InterExactHi"] = f"{np.percentile(binter,97.5)*100:.0f}"
    M["IntraInterGap"] = f"{(float(M['IntraExact']) - inter*100):.1f}"
    for k in (4, 3, 2, 1, 0):
        if k in cond:
            n, v = cond[k]
            M[f"CondN{k}"] = str(n)
            M[f"CondPct{k}"] = f"{v*100:.0f}"
            M[f"CondLo{k}"] = f"{np.percentile(bagg[k],2.5)*100:.0f}"
            M[f"CondHi{k}"] = f"{np.percentile(bagg[k],97.5)*100:.0f}"
    cond_plot = {k: (cond[k][1]*100, np.percentile(bagg[k],2.5)*100,
                     np.percentile(bagg[k],97.5)*100, cond[k][0]) for k in cond}

    # ---------------------------------------------------------------- representation
    print("vectorising corpus ...")
    withclaims = {r["id"]: r for r in list_proverbs(with_claims_only=True)}
    ids = list(withclaims)
    X, _ = vectorize([str(withclaims[i]["claim"]) for i in ids])
    pos = {p: k for k, p in enumerate(ids)}

    from core.annotation_quality import aggregate_constraints
    agg, _ = aggregate_constraints(list_constraints())
    use = [(x["a_id"], x["b_id"], 1 if x["label"] == "must" else 0) for x in agg
           if x["label"] in ("must", "cannot") and x["a_id"] in pos and x["b_id"] in pos]
    sims = np.array([float(cosine_similarity(X[pos[a]], X[pos[b]])[0, 0]) for a, b, _l in use])
    labs = np.array([l for _a, _b, l in use])

    def auc(s, y):
        p_, n_ = s[y == 1], s[y == 0]
        U, pv = mannwhitneyu(p_, n_, alternative="two-sided")
        return U/(len(p_)*len(n_)), pv, len(p_), len(n_)

    a_all, p_all, npos, nneg = auc(sims, labs)
    band = (sims >= 0.30) & (sims <= 0.85)
    a_band, p_band, nbp, nbn = auc(sims[band], labs[band])
    M.update(AUCAll=f"{a_all:.3f}", AUCAllPos=str(npos), AUCAllNeg=str(nneg),
             AUCBand=f"{a_band:.3f}", AUCBandPos=str(nbp), AUCBandNeg=str(nbn),
             AUCAllP=f"{p_all:.1e}".replace("e-", "e{-}"),
             AUCBandP=f"{p_band:.1e}".replace("e-", "e{-}"))
    below = sims < 0.30
    M["NegBelowPct"] = f"{(labs[below]==0).sum()/(labs==0).sum()*100:.0f}"
    M["PosBelowPct"] = f"{(labs[below]==1).sum()/(labs==1).sum()*100:.0f}"

    # similarity control on double-rated pairs
    ds = [(float(cosine_similarity(X[pos[p[0]]], X[pos[p[1]]])[0, 0]), max(v)-min(v),
           statistics.median(v))
          for p, v in dbl.items() if p[0] in pos and p[1] in pos]
    s_ = np.array([d[0] for d in ds]); sp_ = np.array([d[1] for d in ds])
    md_ = np.array([d[2] for d in ds])
    rho, prho = spearmanr(s_, sp_)
    inb = (s_ >= 0.30) & (s_ <= 0.85)
    _u, pmw = mannwhitneyu(sp_[inb], sp_[~inb], alternative="greater")
    M.update(SimN=str(len(ds)), SimRho=f"{rho:+.3f}", SimRhoP=f"{prho:.2f}",
             SimInBandSpread=f"{sp_[inb].mean():.2f}",
             SimOutBandSpread=f"{sp_[~inb].mean():.2f}", SimBandP=f"{pmw:.3f}")

    # ---------------------------------------------------------------- clustering
    print("clustering (threshold sweep) ...")
    rng = np.random.default_rng(42)
    claims = [str(withclaims[i]["claim"]) for i in ids]
    samp_idx = rng.choice(len(claims), size=min(3000, len(claims)), replace=False)
    samp = [claims[i] for i in samp_idx]
    samp_ids = [ids[i] for i in samp_idx]
    sweep = []
    for tau in (0.25, 0.30, 0.35, 0.40, 0.45):
        lab, _method = cluster_texts(samp, samp_ids, tau=tau)
        sil = silhouette_cosine(samp, lab)
        ari = bootstrap_stability(samp, samp_ids, tau=tau, iterations=3, frac=0.8)
        sweep.append((tau, len(set(lab)), sil, ari))
        print(f"  tau={tau}: {len(set(lab))} clusters, silhouette {sil:.3f}, ARI {ari:.3f}")
    for tau, nc, sil, ari in sweep:
        t = f"{int(round(tau*100)):02d}"
        M[f"SweepK{t}"], M[f"SweepSil{t}"], M[f"SweepARI{t}"] = texnum(nc), f"{sil:.3f}", f"{ari:.3f}"

    print("clustering (full corpus) ...")
    full, _m = cluster_texts(claims, ids, tau=0.35)
    bycl = collections.defaultdict(set)
    for i, lab in zip(ids, full):
        pe = withclaims[i].get("people")
        if pe:
            bycl[lab].add(pe)
    cov = [len(v) for v in bycl.values()]
    M.update(NClusters=texnum(len(set(full))), MaxCoverage=str(max(cov)),
             MeanCoverage=f"{statistics.mean(cov):.2f}",
             CoverageP99=str(int(np.percentile(cov, 99))))

    # ---------------------------------------------------------------- figures
    print("figures ...")
    # Fig 1: conditional agreement, the central finding
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ks = [4, 3, 2, 1, 0]
    xs = np.arange(len(ks))
    vals = [cond_plot[k][0] for k in ks]
    lo = [vals[i] - cond_plot[k][1] for i, k in enumerate(ks)]
    hi = [cond_plot[k][2] - vals[i] for i, k in enumerate(ks)]
    bars = ax.bar(xs, vals, yerr=[lo, hi], capsize=4,
                  color=[PALETTE["accent"] if k in (4, 0) else PALETTE["warn"] for k in ks],
                  width=0.6, error_kw={"ecolor": PALETTE["mid"], "elinewidth": 1})
    for x, k in zip(xs, ks):
        ax.text(x, 3, f"n={cond_plot[k][3]}", ha="center", fontsize=7.5, color="white")
    ax.axhline(float(M["IntraExact"]), color=PALETTE["ink"], linestyle=":", linewidth=1.2)
    ax.text(len(ks) - 0.45, float(M["IntraExact"]) + 2,
            f"same reader, twice: {M['IntraExact']}%", fontsize=8,
            color=PALETTE["ink"], ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels(["4\nsame rule", "3\nsame advice", "2\nsame theme",
                        "1\nrelated,\ndifferent lesson", "0\nunrelated"], fontsize=8.5)
    ax.set_ylabel("second reader gave the same score (%)", fontsize=9.5)
    ax.set_ylim(0, 100)
    style(ax)
    ax.set_title("Agreement is high at the ends of the scale and collapses in its middle",
                 fontsize=10.5, color=PALETTE["ink"], pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig1_disagreement_by_level.pdf"))
    plt.close(fig)

    # Fig 2: coverage distribution
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    counts = collections.Counter(cov)
    xs = sorted(counts)
    ax.bar(xs, [counts[x] for x in xs], color=PALETTE["accent"], width=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("distinct cultures in a cluster", fontsize=9.5)
    ax.set_ylabel("clusters (log scale)", fontsize=9.5)
    style(ax)
    ax.set_title("Cross-cultural coverage is heavy-tailed", fontsize=10.5,
                 color=PALETTE["ink"], pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig2_coverage.pdf"))
    plt.close(fig)

    # Fig 3: threshold sensitivity
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    taus = [s[0] for s in sweep]
    ax.plot(taus, [s[2] for s in sweep], "o-", color=PALETTE["accent"], label="silhouette")
    ax.set_xlabel(r"distance threshold $\tau$", fontsize=9.5)
    ax.set_ylabel("silhouette (cosine)", fontsize=9.5, color=PALETTE["accent"])
    style(ax)
    ax2 = ax.twinx()
    ax2.plot(taus, [s[3] for s in sweep], "s--", color=PALETTE["warn"], label="bootstrap ARI")
    ax2.set_ylabel("bootstrap ARI", fontsize=9.5, color=PALETTE["warn"])
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(colors=PALETTE["mid"], labelsize=9)
    ax.set_title("Partitions stay stable while separation stays low", fontsize=10.5,
                 color=PALETTE["ink"], pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig3_sensitivity.pdf"))
    plt.close(fig)

    # ---------------------------------------------------------------- write outputs
    try:
        commit = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        commit = "unknown"
    M.update(NumbersGenerated=datetime.date.today().isoformat(),
             NumbersCommit=commit, NumbersDBFingerprint=fingerprint(db),
             FrozenCorpusFile=os.path.basename(db).replace("_", r"\_"))

    path = os.path.join(out, "numbers.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("% Generated by scripts/paper_analysis.py from ONE frozen corpus.\n")
        f.write(f"% {os.path.basename(db)}  sha256:{M['NumbersDBFingerprint']}  "
                f"code {commit}  {M['NumbersGenerated']}\n\n")
        for k, v in sorted(M.items()):
            f.write(f"\\newcommand{{\\{macro_name(k)}}}{{{v}}}\n")
    print(f"\nwrote {path} ({len(M)} macros) and 3 figures in {figdir}")
    json.dump(M, open(os.path.join(out, "numbers.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
