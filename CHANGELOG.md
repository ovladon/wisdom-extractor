# Changelog — The Wisdom Extractor

All notable transformations of this project. Versions before v19 predate this git
repository and are imported as dated archive commits (tags `v7.7`–`v18`); their
snapshots were recovered from the project's version folders and `v8-18.zip`.

## v19.3 — 2026-07-17
- **World map integrated into the app** (`core/mapview.py`, Results tab): motif view
  (attestation points + arcs, per-culture proverb tooltips), shared-wisdom network
  (120 strongest culture pairs), Europe zoom, and a **time slider** — "attested by
  year N" — driven by the first_seen bounds, with undated attestations shown faded.
  Fully self-contained HTML rendered via components.html; no external requests.
- New data assets: `data/world_map_paths.svg` (compact public-domain world outline),
  `data/people_coords.csv` (67 cultural centroids, extendable).

## v19.2 — 2026-07-17
- **Attestation years / historical timeline.** `extract_attestation_year()` harvests
  "attested no later than" years from citation tails before stripping;
  `backfill_attestation_years()` dates existing rows from citations or dated sources
  (`data/source_years.json`). On the current corpus: 5,738 rows dated (31%), earliest
  bounds 1611–1758. Scrape/seed capture years automatically; tab 2 gains a 🕰️ backfill
  button; tab 5 shows a per-cluster attestation timeline (year × culture).
- New boilerplate rules drop source-book headers ("Nathan Bailey (1721). …") that
  previously survived as fake proverbs.
- Third-party warning noise (TensorFlow plugin registration, sklearn FutureWarning,
  numba/TBB) suppressed at app start.

## v19.1 — 2026-07-17
- **Annotation confidence hierarchy** (`core/annotation_quality.py`): reliability-weighted
  consensus per annotated pair (Dawid–Skene-lite; annotator reliability learned from
  agreement with consensus, prior 0.7). Clustering enforces only consensus constraints
  above a confidence slider; tied/low-confidence pairs are excluded and queued.
- Annotate tab: new "Verify disputed / low-confidence" strategy serving the pairs whose
  confidence benefits most from another vote.
- Diagnostics: annotation consensus stats, per-annotator reliability table, review queue.
- Fix: empty-database crash in Diagnostics (`cached_proverbs` now returns explicit columns).

## v19.0 — 2026-07-16 — the unification
Merged the two development lineages and fixed the defects of both.
- From the paper pipeline (v10): canonicalization cascade, lexical normalization,
  char 3–5-gram TF-IDF, agglomerative clustering (τ cut), wisdom score
  (coverage + 0.3·support), 2D projection, τ-sweep/silhouette/bootstrap-ARI diagnostics,
  permutation triangulation, themed interpretation, ecological correlations,
  21,378-row culture-labelled seed dataset.
- From Wisdom Lab (v18): depth-1 concurrent scraper (robots.txt, MediaWiki API
  fallbacks), SQLite persistence with WAL + hash dedup, annotation game with batch
  writes and leaderboard.
- New in the merge: **constraint-aware clustering** (must/cannot-links enforced via
  constrained union-find); **constraint-agreement evaluation** (annotations double as
  evaluation set); **scalable sparse clustering** (radius-neighbour graph; 18k rows in
  ~14 s vs the 2.7 GB dense matrix that v18 required); culture inference from URLs +
  backfill + family/region enrichment; citation stripping; in-place schema migration
  for v15–v18 databases; empty-catalog fix (v18 shipped a 19-byte sources_catalog.json).

## v18 — 2025-10-08 (archive)
Wisdom Lab "fast annotate": batch/instant write modes, autosave, cached pairs, WAL
tuning, richer scraper (stop button, uncapped depth-1). Analysis layer unchanged since
v15; culture labels still not captured (defect fixed in v19).

## v16 — 2025-10-01 (archive)
Wisdom Lab: scraper gains MediaWiki API fallback and XML parser auto-detect; DB exports.

## v15 — 2025-10-01 (archive)
First "Wisdom Lab" pivot: Streamlit app with SQLite persistence, sources catalog,
basic scraper, TF-IDF paraphrase graph + connected components, annotation game
(must/cannot/not-a-saying), survival score, leaderboard. Analysis simpler than the
paper lineage (word n-grams, no canonicalization, no diagnostics).

## v10 — 2025-09-30 (archive)
Final paper-lineage version; produced the numbers in the ConsILR-2025 submission
(τ sensitivity table, cophenetic 0.805, bootstrap ARI, permutation triangulation,
canonicalization/back-translation audit templates). Enhanced canonicalization
(~20 rule families), lexical normalization, adaptive thresholds, trust-score
diagnostics, simple validation tab.

## v7.7 — 2025-08-21 (archive)
The version described in the ConsILR-2025 paper text: five-tab Streamlit pipeline
(Dataset Builder, Wisdom Extractor, Results & Visualisation, Interpretation,
Diagnostics), regex canonicalization, char n-gram clustering, deterministic + optional
local-LLM interpretation.

## Earlier (v1–v6, v8–v9, v11–v14, v17) — not imported
Intermediate experiments; folders exist in the project archive but are not part of
this repository's history.
