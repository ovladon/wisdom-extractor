# Contributing

Thanks for your interest!

- Use issues for bugs and feature requests.
- When adding code, follow standard Python style (PEP 8).
- Keep functions small and testable.
- Add or update documentation as needed.

## Dev quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- `ai_filters.py` can optionally download a TinyLlama GGUF into `./models/`.
- All run artifacts are written to `./runs/` (ignored by git).
