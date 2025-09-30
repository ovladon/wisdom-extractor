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

## How to publish on GitHub (browser workflow)
1. Sign in and open **New repository** (Owner: **ovladon**).
2. **Repository name**: `wisdom-extractor`. Visibility: Public.  
   **Important**: *Do not* tick “Initialize with a README/ .gitignore/ license” to avoid merge conflicts.
3. Click **Create repository**.
4. On the next page, click **“uploading an existing file”**.  
   From your computer, unzip the archive from this chat and **drag & drop the *contents* of the `wisdom-extractor/` folder** (not the folder itself) into the upload area. This preserves the folder tree (including `.github/`).
5. Scroll down, add a commit message (e.g., "Initial commit"), and click **Commit changes**.
6. Open the repo → **Settings → Pages** if you later want to enable docs (optional).

## Run locally
```bash
streamlit run app.py
```

## License
MIT © 2025 ovladon
