# Wisdom Extractor

Streamlit app for building a cross‑cultural dataset of proverbs, clustering similar *claims*, and producing interpretable reports and diagnostics.

## Features
- Scrape/merge proverb sources and clean them robustly
- Canonicalize and cluster semantically similar claims
- Visualize clusters and inspect examples per culture
- Lightweight human validation UI
- Deterministic (non‑LLM) interpretation and optional local LLM summary
- Practical diagnostics with a composite **trust score**

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Defaults expect these files in the repo root:
- `proverbs_clean_v2.csv` (sample dataset included)
- `people_metadata_v2.csv` (sample metadata included)
- `sources.yaml` (optional scraping targets)

Outputs (written to repo root or `runs/`):
- `wisdom_clusters.json`, `clusters.csv`, `clusters_coords.csv`
- `interpretation_report.txt`

### Optional local LLM
`ai_filters.py` tries to auto‑download a tiny GGUF (TinyLlama) into `./models/` for an optional filter/summary. You can also place another `.gguf` there.

## Repository layout
```
wisdom-extractor/
├─ app.py
├─ extractor.py
├─ dataset_builder.py
├─ diagnostics.py
├─ interpret_v2.py
├─ interpret_llm.py
├─ ai_filters.py
├─ proverbs_cleaner.py
├─ requirements.txt
├─ people_metadata_v2.csv
├─ proverbs_clean_v2.csv
├─ sources.yaml
├─ runs/            # artifacts (git-ignored)
├─ models/          # local LLMs (git-ignored)
├─ docs/QUICKSTART.md
├─ .github/         # issue & PR templates
├─ .gitignore
├─ LICENSE
├─ CHANGELOG.md
├─ CONTRIBUTING.md
└─ CODE_OF_CONDUCT.md
```

## Run locally
```bash
streamlit run app.py
```

## License
MIT © 2025 ovladon
