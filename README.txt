# Wisdom Lab — Full Plus (Seeded v16)

Fixes & improvements
- Bugfix: `core/persistence.py::stats()` no longer assigns to sqlite connection attributes.
- Scraper: MediaWiki-aware, robust selectors, `?action=render` fallback, automatic parser choice using lxml.
- Seeding: Built-in Wikiquote catalog loads automatically on first run.
- UI: Progress, live stats, and exports.

Run
```
pip install -r requirements.txt
streamlit run app.py
```
