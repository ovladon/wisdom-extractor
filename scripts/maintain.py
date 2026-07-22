#!/usr/bin/env python3
"""Self-maintenance for the Wisdom Extractor (v19.7). Run periodically (cron).

One run = the corpus digests everything that happened since the last run:
  1. optionally scrape the next N catalog sources (round-robin cursor, polite);
  2. backfill culture labels, family/region, attestation years, English glosses;
  3. canonicalize rows that lack a claim;
  4. aggregate all annotations (graded + legacy) into consensus constraints;
  5. recluster the full corpus with those constraints and save cluster ids.

New scraped sayings therefore become annotatable (glossed) and clustered without
any manual step, and every accumulated human judgment reshapes the clusters.

Usage:
  python scripts/maintain.py                     # digest only
  python scripts/maintain.py --scrape 2          # also crawl next 2 sources
  python scripts/maintain.py --scrape 1 --source omniglot   # crawl matching source(s)
Env: WISDOM_DB_PATH (the shared database).
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.persistence import (init_db, list_sources, insert_proverb, list_proverbs,
                              save_claims, save_clusters, list_constraints,
                              backfill_people_from_urls, enrich_family_region,
                              backfill_attestation_years, backfill_glosses,
                              infer_people_from_url)
from core.cleaner import keep, quality_score, strip_citations, extract_attestation_year
from core.canonicalize import canonicalize
from core.clustering import cluster_texts
from core.annotation_quality import aggregate_constraints, constraint_pairs_for_clustering
from core.persistence import (merge_reported_duplicates, fix_ocr_artifacts,
                              apply_corrections, dedup_normalized)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def ensure_catalog():
    """Load the bundled sources catalog into the DB if it's missing (idempotent)."""
    from core.persistence import upsert_source
    catalog_path = os.path.join(DATA_DIR, "sources_catalog.json")
    if len(list_sources()) < 5 and os.path.exists(catalog_path):
        catalog = json.load(open(catalog_path, encoding="utf-8"))
        for s in catalog.get("sources", []):
            upsert_source(s.get("name", s.get("url", "(no name)")), s["url"], ",".join(s.get("tags", [])))


def scrape_next(n, source_filter=None, respect_robots=True, workers=6):
    from scraper.basic_scraper import crawl_source
    ensure_catalog()
    cursor_path = os.environ.get("WISDOM_DB_PATH", "wisdom.db") + ".cursor"
    cursor = 0
    if os.path.exists(cursor_path):
        try:
            cursor = json.load(open(cursor_path)).get("cursor", 0)
        except Exception:
            cursor = 0
    sources = [s for s in list_sources(enabled_only=True)
               if not source_filter or source_filter.lower() in (s["name"] + s["url"]).lower()]
    if not sources:
        return {"scraped_sources": 0, "new": 0, "skipped_noise": 0}
    new, noise, done = 0, 0, []
    for i in range(min(n, len(sources))):
        s = sources[(cursor + i) % len(sources)]
        done.append(s["name"])
        try:
            pages, items = crawl_source(s["url"], respect_robots=respect_robots, workers=workers)
        except Exception as e:
            print(f"  scrape failed {s['url']}: {e}")
            continue
        for it in items:
            year = extract_attestation_year(it["text"])
            text = strip_citations(it["text"])
            if not keep(text):
                noise += 1
                continue
            people = infer_people_from_url(it["url"], s["name"])
            if insert_proverb(s["id"], text, it["url"], people=people, first_seen=year):
                new += 1
    if not source_filter:
        json.dump({"cursor": (cursor + n) % len(sources), "updated": time.time()}, open(cursor_path, "w"))
    return {"scraped_sources": len(done), "sources": done, "new": new, "skipped_noise": noise}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrape", type=int, default=0, help="crawl next N catalog sources")
    ap.add_argument("--source", default=None, help="only sources matching this substring")
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--no-cluster", action="store_true", help="skip reclustering")
    args = ap.parse_args()

    t0 = time.time()
    summary = {}
    init_db()

    if args.scrape > 0:
        summary["scrape"] = scrape_next(args.scrape, args.source)
        print("scrape:", summary["scrape"])

    summary["ocr_fixed"] = fix_ocr_artifacts()
    summary["corrections_applied"] = apply_corrections()
    summary["dedup_excluded"] = dedup_normalized()

    summary["people_backfilled"] = backfill_people_from_urls()
    summary["family_region_enriched"] = enrich_family_region(os.path.join(DATA_DIR, "people_metadata.csv"))
    cit, src = backfill_attestation_years(os.path.join(DATA_DIR, "source_years.json"))
    summary["years_backfilled"] = cit + src
    glossed, unglossable = backfill_glosses()
    summary["glossed"] = glossed

    rows = list_proverbs(excluded=False)
    missing = [(r["id"], canonicalize(r["text"]), quality_score(r["text"]))
               for r in rows if not r["claim"]]
    if missing:
        save_claims(missing)
    summary["canonicalized_new"] = len(missing)

    cons = list_constraints()
    summary["duplicates_merged"] = merge_reported_duplicates()

    agg_pairs, annotators = aggregate_constraints(cons)
    must, cannot = constraint_pairs_for_clustering(agg_pairs, min_confidence=args.min_conf)
    summary["annotations"] = {"raw": len(cons), "consensus_pairs": len(agg_pairs),
                              "must": len(must), "cannot": len(cannot),
                              "annotators": len(annotators)}

    if not args.no_cluster:
        rows = list_proverbs(with_claims_only=True)
        texts = [r["claim"] for r in rows]
        ids = [r["id"] for r in rows]
        labels, method = cluster_texts(texts, ids, tau=args.tau,
                                       must_pairs=must, cannot_pairs=cannot, agglo_limit=4000)
        save_clusters(list(zip(ids, labels)))
        summary["clustering"] = {"items": len(ids), "clusters": int(len(set(labels.tolist()))),
                                 "method": method, "tau": args.tau}

    summary["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
