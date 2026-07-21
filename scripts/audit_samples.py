#!/usr/bin/env python3
"""Human-audit instruments (Pelican revision): canonicalisation & cluster quality.

generate mode: produces two CSVs with empty rating columns for human judges:
  canon_audit_100.csv    100 structurally-rewritten items; rate meaning preservation
                         (yes / partial / no)
  cluster_audit_50.csv   50 sampled multi-member clusters; rate coherence
                         (pure / mostly / mixed / junk) and count intruder items

summarize mode: reads the filled CSVs back and prints the paper-ready statistics.

Usage:
  WISDOM_DB_PATH=... python scripts/audit_samples.py generate --out DIR
  python scripts/audit_samples.py summarize --dir DIR
"""
import argparse, csv, os, random, re, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

random.seed(42)


def norm(t):
    t = re.sub(r"\s+", " ", str(t)).strip().lower()
    return t[:-1] if t and t[-1] == "." else t


def generate(outdir):
    from core.persistence import list_proverbs
    rows = list_proverbs(with_claims_only=True)

    rewritten = [r for r in rows if norm(r["claim"]) != norm(r["text"])]
    sample = random.sample(rewritten, min(100, len(rewritten)))
    with open(os.path.join(outdir, "canon_audit_100.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "original_text", "canonical_claim", "meaning_preserved(yes/partial/no)", "notes"])
        for r in sample:
            w.writerow([r["id"], r["text"], r["claim"], "", ""])

    by_cluster = {}
    for r in rows:
        if r["cluster_id"] is not None:
            by_cluster.setdefault(r["cluster_id"], []).append(r)
    multi = {c: m for c, m in by_cluster.items() if len(m) >= 3}
    chosen = random.sample(sorted(multi), min(50, len(multi)))
    with open(os.path.join(outdir, "cluster_audit_50.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "size", "members(gloss_or_text; ||-separated)",
                    "coherence(pure/mostly/mixed/junk)", "n_intruders", "notes"])
        for c in chosen:
            members = " || ".join((m.get("gloss") or m["text"])[:90] for m in multi[c][:12])
            w.writerow([c, len(multi[c]), members, "", "", ""])
    print(f"generated: canon_audit_100.csv ({len(sample)} items), "
          f"cluster_audit_50.csv ({len(chosen)} clusters) in {outdir}")
    print("Hand these to two raters; then run: audit_samples.py summarize --dir", outdir)


def summarize(d):
    p1 = os.path.join(d, "canon_audit_100.csv")
    if os.path.exists(p1):
        with open(p1, newline="", encoding="utf-8") as f:
            vals = [row[3].strip().lower() for row in list(csv.reader(f))[1:] if len(row) > 3]
        filled = [v for v in vals if v]
        c = Counter(filled)
        if filled:
            print(f"canonicalisation audit: n={len(filled)} rated | " +
                  ", ".join(f"{k}: {v} ({v/len(filled):.0%})" for k, v in c.most_common()))
        else:
            print("canonicalisation audit: not yet rated")
    p2 = os.path.join(d, "cluster_audit_50.csv")
    if os.path.exists(p2):
        with open(p2, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))[1:]
        rated = [r for r in rows if len(r) > 3 and r[3].strip()]
        c = Counter(r[3].strip().lower() for r in rated)
        intr = [int(r[4]) for r in rated if len(r) > 4 and r[4].strip().isdigit()]
        sizes = [int(r[1]) for r in rated if r[1].strip().isdigit()]
        if rated:
            print(f"cluster audit: n={len(rated)} rated | " +
                  ", ".join(f"{k}: {v} ({v/len(rated):.0%})" for k, v in c.most_common()))
            if intr and sizes:
                purity = 1 - sum(intr) / max(1, sum(sizes))
                print(f"  estimated member-level purity: {purity:.1%}")
        else:
            print("cluster audit: not yet rated")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["generate", "summarize"])
    ap.add_argument("--out", "--dir", dest="dir", default=".")
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    generate(a.dir) if a.mode == "generate" else summarize(a.dir)
