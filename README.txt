# Wisdom Lab — Full Plus (Seeded v18)

## New in v18
- **Fast annotation UX**:
  - "Batch (faster)" mode queues operations and applies them with one click (or autosaves after N actions).
  - "Instant" mode still writes to DB immediately.
  - UI updates immediately without recomputing heavy similarity.
- **Heavy compute cached**:
  - Active proverbs list cached with a `db_version` invalidation flag.
  - Nearest-pairs computation cached and recomputed only on explicit "Refresh pairs" or after a DB change.
- **DB tuned**: WAL journal + NORMAL sync for snappy writes.
- **Still includes**: MediaWiki API fallback, lxml/XML parser auto-detect, depth‑1 crawling, stop button, exports.

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
