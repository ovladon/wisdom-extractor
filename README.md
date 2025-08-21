
# Wisdom Extractor v7.7 (multi-source, offline-capable)

**Run**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

**Notes**
- Deterministic interpretation works without any model or internet.
- Optional local LLM (llama.cpp) is offline; use 'auto' or upload a `.gguf` in the UI.
- Dataset Builder now ships with an expanded `sources.yaml` (~60+ peoples across Wikiquote/Wiktionary + two classic collections).
- Each fetch is saved separately under `./runs/*.csv`. Merge & Clean concatenates everything with uploads.
- Diagnostics tab: silhouette, compactness, stability, and theme×proxy Spearman correlations.
