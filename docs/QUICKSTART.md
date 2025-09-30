# Quickstart

1. Create a Python 3.10+ virtual env and install deps:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. In the UI:
   - **Dataset Builder**: Upload CSV(s) or fetch from `sources.yaml`.
   - **Improved Extractor**: Cluster wisdom claims.
   - **Results & Analysis**: Inspect clusters.
   - **Quick Validation**: Rate 10 clusters.
   - **Interpretation**: Generate cross‑cultural summaries.
   - **Practical Diagnostics**: See trust score and actions.

> Tip: If `llama-cpp-python` can't find a model, `ai_filters.py` will try a tiny default under `./models/` if possible.
