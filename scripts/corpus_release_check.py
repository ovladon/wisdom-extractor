#!/usr/bin/env python3
"""Evaluate the pre-registered corpus release criteria (CORPUS_RELEASE_CRITERIA.md).

Detects and prepares. Never publishes: Zenodo publication is permanent and public,
so a human confirms it. Triggers are size-based, never result-based, so the version
history cannot be accused of being timed to flatter the statistics.

  python scripts/corpus_release_check.py             # status
  python scripts/corpus_release_check.py --prepare   # also write the export + manifest
"""
import argparse, json, os, sys, csv, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.persistence import init_db, list_proverbs, list_constraints, stats
from core.annotation_quality import aggregate_constraints
from core.science import alpha_with_ci, overlap_stats, annotator_profile

STATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "last_corpus_release.json")

GROWTH_PROVERBS = 0.25      # +25%
GROWTH_JUDGMENTS = 1.00     # +100%
GROWTH_DOUBLE = 150         # +150 double-rated pairs
STALE_DAYS = 365


def current_state():
    st = stats()
    cons = list_constraints()
    agg, _ = aggregate_constraints(cons)
    ov = overlap_stats(cons)
    an = annotator_profile(cons)
    al = alpha_with_ci(cons, B=200)
    return {
        "generated": datetime.date.today().isoformat(),
        "proverbs": st["proverbs"], "peoples": st["peoples"],
        "judgments": len(cons), "consensus_pairs": len(agg),
        "double_rated": ov["multi_rated"], "annotators": an["n_annotators"],
        # recorded for transparency, never used as a trigger
        "alpha": al["alpha"], "alpha_ci": [al.get("lo"), al.get("hi")],
    }


def last_release():
    if os.path.exists(STATE):
        return json.load(open(STATE, encoding="utf-8"))
    return None


def evaluate(cur, last):
    if not last:
        return [{"trigger": "no version released yet",
                 "met": True,
                 "detail": "Nothing has been published from this corpus."}], True
    fired = []
    dp = (cur["proverbs"] - last["proverbs"]) / max(1, last["proverbs"])
    fired.append({"trigger": "+25% proverbs", "met": dp >= GROWTH_PROVERBS,
                  "detail": f"{dp*100:+.1f}% ({last['proverbs']:,} -> {cur['proverbs']:,})"})
    dj = (cur["judgments"] - last["judgments"]) / max(1, last["judgments"])
    fired.append({"trigger": "+100% judgments", "met": dj >= GROWTH_JUDGMENTS,
                  "detail": f"{dj*100:+.1f}% ({last['judgments']:,} -> {cur['judgments']:,})"})
    dd = cur["double_rated"] - last.get("double_rated", 0)
    fired.append({"trigger": "+150 double-rated pairs", "met": dd >= GROWTH_DOUBLE,
                  "detail": f"{dd:+d} ({last.get('double_rated', 0)} -> {cur['double_rated']})"})
    try:
        age = (datetime.date.fromisoformat(cur["generated"])
               - datetime.date.fromisoformat(last["generated"])).days
    except Exception:
        age = 0
    fired.append({"trigger": "12 months since last release", "met": age >= STALE_DAYS,
                  "detail": f"{age} days"})
    return fired, any(f["met"] for f in fired)


def prepare_export(outdir):
    """Write the CSVs a release would contain, plus a manifest. Nothing is uploaded."""
    os.makedirs(outdir, exist_ok=True)
    rows = list_proverbs(excluded=False)
    with open(os.path.join(outdir, "proverbs.csv"), "w", newline="", encoding="utf-8") as f:
        cols = ["id", "text", "people", "language", "family", "region", "original",
                "claim", "gloss", "first_seen", "cluster_id", "url", "sensitive"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    cons = list_constraints()
    with open(os.path.join(outdir, "annotations.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["a_id", "b_id", "score", "label", "user"],
                           extrasaction="ignore")
        w.writeheader()
        for c in cons:
            w.writerow(c)          # `user` is already a pseudonym uid
    agg, _ = aggregate_constraints(cons)
    with open(os.path.join(outdir, "consensus_pairs.csv"), "w", newline="", encoding="utf-8") as f:
        keys = ["a_id", "b_id", "consensus_score", "label", "n", "confidence", "disputed"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for p in agg:
            w.writerow(p)
    return outdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true",
                    help="write the export files (still does not publish)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--record-release", action="store_true",
                    help="mark the CURRENT state as released (run after publishing)")
    args = ap.parse_args()

    init_db()
    cur = current_state()
    last = last_release()
    checks, ready = evaluate(cur, last)

    print("Corpus now: "
          f"{cur['proverbs']:,} proverbs · {cur['peoples']} peoples · "
          f"{cur['judgments']:,} judgments · {cur['double_rated']} double-rated · "
          f"{cur['annotators']} annotators")
    if last:
        print(f"Last released version: {last.get('version','?')} on {last.get('generated','?')}")
    print("\nPre-registered triggers (size-based only, never result-based):")
    for c in checks:
        print(f"  [{'MET ' if c['met'] else '    '}] {c['trigger']:32} {c['detail']}")
    print("\n=> " + ("READY: a new corpus version is due. Prepare, inspect, then publish "
                     "by hand." if ready else
                     "Not yet due. Keep annotating; nothing to do."))
    print("   (alpha is recorded in the manifest for transparency but never triggers a "
          "release: publishing only on good numbers would bias the version history.)")

    if args.prepare:
        out = args.outdir or os.path.join(os.path.dirname(STATE), "corpus_export_pending")
        prepare_export(out)
        json.dump(cur, open(os.path.join(out, "manifest.json"), "w"), indent=2)
        print(f"\nExport prepared in {out} (proverbs.csv, annotations.csv, "
              f"consensus_pairs.csv, manifest.json). Nothing has been uploaded.")
        print("Publish with journal_submission/zenodo_deposit/publish_corpus.sh after "
              "checking the files, then re-run with --record-release.")

    if args.record_release:
        cur["version"] = input("version label just published (e.g. v2.0): ").strip() or "?"
        json.dump(cur, open(STATE, "w"), indent=2)
        print("Recorded. Future checks compare against this state.")


if __name__ == "__main__":
    main()
