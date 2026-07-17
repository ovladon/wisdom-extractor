# Wisdom Extractor — Unified (v19)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21413838.svg)](https://doi.org/10.5281/zenodo.21413838)

One app that merges the two development lines of the Wisdom Extractor project:

- the **ConsILR-2025 paper pipeline** (v7 → v10_working): canonicalization, char n-gram
  clustering, wisdom scoring, 2D maps, diagnostics, deterministic interpretation;
- the **Wisdom Lab platform** (v15 → v18): robust depth-1 scraping, SQLite persistence,
  and the human annotation game (must-link / cannot-link / not-a-saying).

See `EXPLANATION.md` for the full account of what was merged, what was broken, and what is new.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The database is `wisdom.db` in the working directory (override with `WISDOM_DB_PATH`).
Pointing it at an existing v15–v18 `wisdom.db` upgrades the schema in place, non-destructively.

## Typical workflow

1. **Tab 2 — Import & Seed**: click "Seed from paper dataset" (21,378 proverbs, 77 peoples,
   full provenance) and/or import your own CSV.
2. **Tab 1 — Sources & Scrape**: crawl more sources; culture labels are inferred from URLs
   automatically and enriched with language family / region metadata.
3. **Tab 3 — Clean & Canonicalize**: filter noise, compute canonical claims.
4. **Tab 4 — Cluster**: char 3–5-gram TF-IDF clustering with your annotation constraints applied.
5. **Tab 6 — Annotate**: play the annotation game; your judgments improve the next clustering run
   and are scored as an evaluation set.
6. **Tabs 5/7/8 — Results, Diagnostics, Interpretation**: browse clusters, validate, interpret.
7. **Tab 9 — Export**: CSV/JSON exports of everything.

## Data files

- `data/seed_proverbs.csv` — the paper's cleaned dataset (v10_working, 21,378 rows, `people` labels)
- `data/people_metadata.csv` — per-people ecological/institutional proxies for correlations
- `data/sources_catalog.json` — 46 curated proverb sources (Wikiquote, Wiktionary, Gutenberg, archives)

## Versioning workflow

This folder is a git repository whose history includes the project's full lineage
(tags `v7.7` through `v19.3` — see `CHANGELOG.md`). Do not make
per-version copies or zips by hand anymore. Instead:

1. Make changes; describe them in `CHANGELOG.md`.
2. Run `./scripts/release.sh <version> "one-line summary"` — commits, tags, and builds
   `dist/wisdom-extractor-v<version>.zip`.
3. `git push origin main --tags` (GitHub Releases can then serve the zips).

Browse any historical version with `git checkout v10` (return with `git checkout main`),
or compare versions: `git diff v18 v19.2 --stat`.

## Citing

Software (always resolves to the latest version): DOI [10.5281/zenodo.21413838](https://doi.org/10.5281/zenodo.21413838).
Method and results: Belciug, V. & Pelican, E. (2025). *The Wisdom Extractor: Mining
Cross-Cultural Proverbs to Elicit Time-Tested Heuristics.* ConsILR-2025, Bucharest.
