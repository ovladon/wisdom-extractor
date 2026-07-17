# Wisdom Extractor — Unified v19: Full Explanation

*Prepared 2026-07-16. This document explains what was merged, what was broken and fixed,
what is genuinely new, how the result was verified, and how it relates to the ConsILR-2025 paper.*

---

## 1. Where this comes from: the two development lines

The project split into two lineages after the paper submission:

| Lineage | Versions | What it was good at | What it lacked |
|---|---|---|---|
| **Paper pipeline** | v7_7 → v8 → v9 → **v10_working** | Canonicalization (~20 rule families), char 3–5-gram TF-IDF + average-linkage clustering (cross-script, no training data), wisdom score, 2D maps, τ-sensitivity sweep, bootstrap ARI, permutation triangulation, deterministic interpretation, ecological correlations | CSV-file workflow (no database), simple fetcher, no deduplication, no human validation loop |
| **Wisdom Lab** | v15 → v16 → **v18** | Robust depth-1 concurrent crawler (robots.txt, MediaWiki API fallback), SQLite persistence with hash dedup, fast annotation game (must-link / cannot-link / not-a-saying) with batch writes and leaderboard | Essentially no analysis: word-n-gram similarity only (monolingual), no canonicalization, no diagnostics, no interpretation, and several outright defects (below) |

The paper's published numbers (sensitivity table, cophenetic 0.805, ARI 0.9995, permutation
baseline) were produced by **v10_working** — its result files match the paper digit for digit.

## 2. Defects found in v18 (and how v19 fixes each)

1. **Culture was never captured.** The scraper stored no `people`/`language`; 18,395 of 18,453
   DB rows came from a CSV import that also carried none. Since *coverage across cultures* is the
   project's central quantity, the Candidates tab degenerated to score 0 for everything.
   → v19 infers the culture from the page URL at scrape time (`Romanian_proverbs` → Romanian),
   ships a **backfill** button for existing rows, enriches family/region from
   `data/people_metadata.csv`, and seeds from the paper dataset which carries 77 peoples.

2. **Annotations were collected but never used.** Must/cannot-links went into the DB and stopped
   there. → v19 wires them into clustering (constrained union-find: must-links force merges,
   cannot-links block them, processed in ascending distance order) **and** scores every clustering
   run against them ("constraint agreement" in Diagnostics) — the annotation game is now both
   training signal and evaluation set.

3. **O(n²) dense similarity matrices.** v18's `build_edges`/`nearest_pairs` materialized an
   18,430×18,430 dense matrix (~2.7 GB) and looped over ~170M pairs in Python. → v19 uses sparse
   radius/kNN queries (scikit-learn brute cosine on sparse TF-IDF, chunked); the full 18k-row
   clustering runs in ~14 s within a few hundred MB.

4. **Word n-grams can't align across languages.** v18 clustered raw text with word 1–2-gram
   TF-IDF — cross-lingual matches are impossible. → v19 restores the paper's char 3–5-gram
   `char_wb` vectorizer over canonicalized claims (script-agnostic).

5. **The proposition extractor was vestigial.** Six generic frames with unfilled slots, unused
   downstream. → replaced by the paper's full canonicalization cascade plus lexical
   normalization for similarity.

6. **Boilerplate ingestion.** The v16–v18 DB literally contains "TRANSCRIBER'S NOTES" as a
   proverb. → v19 filters at ingestion with the paper's keep() rule plus new patterns learned
   from the actual failures, and **strips trailing bibliographic citations** (the
   "von Düringsfield… pp. 358-359" tails that polluted top clusters).

## 3. What was kept from each lineage

**From v10_working (paper):** `canonicalize()` rule cascade; `preprocess_for_similarity()`
lexical normalization; char 3–5-gram TF-IDF; average-linkage agglomerative clustering with cut
threshold τ (exact paper method, used up to a configurable size limit so paper results remain
reproducible); wisdom score S = coverage + 0.3·support; 2D projection (SVD→UMAP/PCA);
silhouette / bootstrap-ARI / τ-sweep diagnostics; permutation triangulation vs random mixing;
themed deterministic interpretation; ecological Spearman correlations against
`people_metadata.csv`; the cleaned 21,378-row dataset as one-click seed.

**From v18 (Wisdom Lab):** the entire scraper (depth-1 concurrent crawl, robots.txt,
MediaWiki API + `action=render` fallbacks, category expansion); SQLite persistence with WAL and
hash dedup; the annotation play mode with Batch/Instant writes, autosave, leaderboard and JSON
export; the `db_version` caching pattern in the UI.

**New in v19 (in neither parent):** constrained clustering (must/cannot enforced);
constraint-agreement evaluation; scalable sparse clustering path for large corpora;
culture inference from URLs + backfill + metadata enrichment; citation stripping;
in-place schema migration so an existing v15–v18 `wisdom.db` upgrades losslessly
(`WISDOM_DB_PATH=/path/to/old/wisdom.db streamlit run app.py`).

## 4. Architecture

```
app.py                     Streamlit UI, 9 tabs
core/persistence.py        SQLite schema + migration, bulk ops, backfill, constraints
core/cleaner.py            keep()/quality_score()/strip_citations()
core/canonicalize.py       canonicalize() + preprocess_for_similarity()
core/clustering.py         vectorize, agglomerative & sparse-graph clustering,
                           constrained union-find, nearest_pairs, summarize_clusters
core/diagnostics.py        silhouette, bootstrap ARI, τ sweep, coverage bins,
                           constraint agreement, permutation triangulation
core/interpret.py          themes, themed report, productive tensions, ecological correlations
core/projection.py         2D semantic map (SVD → UMAP/PCA)
scraper/basic_scraper.py   v18 crawler (unchanged)
data/seed_proverbs.csv     paper dataset: 21,378 rows, 77 peoples, full provenance
data/people_metadata.csv   ecological/institutional proxies per people
data/sources_catalog.json  46 curated sources
```

Pipeline: **scrape/seed → clean (keep + strip citations) → canonicalize (claim per proverb) →
vectorize (char 3–5-gram) → cluster (τ; constraints) → score (coverage + 0.3·support) →
interpret / diagnose → annotate → re-cluster**. Every artifact lives in one SQLite file;
every step is re-runnable and exportable.

Method note: above the "agglomerative limit" (default 4,000 items) clustering switches from the
paper's average linkage to a radius-graph single-linkage equivalent — necessary for memory, and
slightly more merge-happy at equal τ. For paper-faithful numbers, run on a ≤4,000 stratified
sample (Diagnostics does this automatically for the τ sweep).

## 5. Verification

A 12-stage end-to-end test was run against the real seed dataset (test DB, not your files):

- Seeded 18,188 clean proverbs (of 21,378 raw; 555+ filtered as noise/citations), 76 peoples.
- Canonicalization rule checks pass (e.g. "Too many cooks spoil the broth" → "Excess of cooks harms broth.").
- Agglomerative path: 1,465 clusters on a 1,500 sample. Graph path: 15,448 clusters on 18,188 rows in ~14 s.
- Top clusters after citation-stripping are clean cross-cultural families, e.g.
  *"If there is smoke, there is fire."* — **17 cultures**; *"Don't put off until tomorrow…"* — 15;
  *"As ye sow shall ye reap."* — 13; *"An apple does not fall far from the tree."* — 12.
- Must-link and cannot-link constraints verified enforced; constraint-agreement metric verified.
- Silhouette, permutation triangulation (top cluster: 17 cultures, 5 families, 7 regions,
  92nd percentile vs random on regions), themed report, productive tensions (8 pairs),
  ecological correlations (40 theme×proxy pairs), 2D projection, URL backfill — all pass.
- The Streamlit app was launched and exercised in a browser (Cluster and Diagnostics tabs render
  with live data).

Coverage now reaches 17 cultures on the full dataset (the paper's sample-based run maxed at 6),
still short of the ≥20 "universal" bin — an honest number, and a concrete target for corpus growth.

## 6. Relation to the paper — and to the original idea

**What the paper's protocol looks like here.** Tab 7 reproduces the paper's evaluation:
τ sensitivity sweep on a stratified sample, silhouette, bootstrap ARI, coverage bins,
permutation triangulation. Run them after any corpus change to keep claims calibrated.

**Did the original idea succeed?** Partially — and the honest picture is:

*What holds up:* recurrence across cultures is real and measurable. Kinship-inheritance,
consequence ("as ye sow…"), timing, and evidence ("no smoke without fire") motifs recur across
10–17 unrelated-ish cultures in this corpus, and permutation tests show several top clusters are
more family/region-diverse than random mixing — that is genuine signal, not artifact.
The infrastructure contribution (transparent, offline, reproducible, now with a human-in-the-loop)
is solid and citable.

*What is not yet demonstrated:* "universal wisdom" in the strong sense. No cluster reaches 20+
cultures; European sources dominate; shared Indo-European heritage and translation-mediated
diffusion are not yet separable from independent discovery; and textual presence says nothing
about behavioral use. These are the same caveats the paper itself states — v19 does not erase
them, it gives you the tooling (annotation, triangulation, corpus growth) to attack them.

## 7. Suggested roadmap

1. **Grow non-European coverage** (the catalog already lists African, Māori, Irish archives) —
   the single highest-leverage move for the universality question.
2. **Annotate ~500 pairs** in Tab 6; report constraint agreement as your headline human-validation
   metric; re-cluster with constraints on.
3. **Multilingual embeddings as an optional second vectorizer** (e.g. LaBSE / distiluse) behind
   the same interface, keeping char n-grams as the offline default — would lift the semantic
   ceiling that limits silhouette.
4. **Diachronic layer**: attach earliest-attestation dates where sources give them
   (Gutenberg 1857/1867 collections are dated) to begin separating inheritance from convergence.
5. **Behavioral validation study**: use top clusters as stimuli (vignette experiments) — the
   paper's stated missing link.

## 8. v19.1 addition: annotation confidence hierarchy (2026-07-17)

`core/annotation_quality.py` implements reliability-weighted consensus over annotations:
majority vote per pair, iterated with per-annotator reliability (Dawid–Skene-lite, prior 0.7),
yielding per-pair confidence. Clustering (tab 4) now enforces only consensus constraints above a
confidence slider (disputed/tied pairs excluded); Diagnostics (tab 7) shows annotator reliability
and the review queue; the Annotate tab (tab 6) gained a "Verify disputed / low-confidence"
strategy that serves exactly the pairs whose confidence would benefit most from another vote.
Result: the more people annotate, the better the constraints, the evaluation set, and the
clustering — monotonically, with bad-faith annotators automatically down-weighted.

## 9. v19.2 addition: attestation years & historical timeline (2026-07-17)

- `core/cleaner.py: extract_attestation_year()` harvests the earliest year from citation
  contexts ("(1875)", trailing ", 1857") BEFORE citations are stripped — an "attested no
  later than" bound, never an origin date.
- `core/persistence.py: backfill_attestation_years()` dates existing rows from citations,
  else from dated sources matched by URL via `data/source_years.json` (1857/1867
  compilations, KJV 1611, etc.). On the current corpus: 800 rows dated from citations +
  4,938 from dated sources = 5,738 rows (31%), earliest bounds 1611–1758.
- Scrape and seed paths now capture years automatically; tab 2 has a 🕰️ backfill button;
  tab 5 shows a per-cluster attestation timeline (year × culture scatter).
- New boilerplate rules drop source-book headers ("Nathan Bailey (1721). Divers
  Proverbs…") that previously survived as fake proverbs.
- App header now suppresses third-party warning noise (TensorFlow plugin registration,
  sklearn FutureWarning, numba/TBB) when run outside a venv.
