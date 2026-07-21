#!/usr/bin/env python3
"""Ingest a public-domain plain-text book (e.g. Internet Archive OCR) as proverbs.

OCR text is noisy; this applies conservative line filters, then the standard cleaning
(citation stripping, keep()) before insertion. Whatever noise survives is caught later
by glossing (non-English lines aren't served to annotators) and by annotators'
"not a saying" reports. Source year flows into the attestation timeline.

Usage:
  WISDOM_DB_PATH=... python scripts/ingest_text.py \
      --url https://archive.org/download/ID/ID_djvu.txt \
      --name "Archive – Swahili proverbs (Taylor, 1891)" \
      --people Swahili --year 1891
"""
import argparse, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
from core.persistence import init_db, upsert_source, bulk_insert_proverbs
from core.cleaner import keep, quality_score, strip_citations

ROMAN_RX = re.compile(r"^[IVXLC]+\.?$")
SCHOLARLY_RX = re.compile(r"\b(cp\.|op\. ?cit|ibid|viz\.|sc\.|q\.v\.|supra|infra|"
                          r"properly|literally|lit\.|the subject is|see note|footnote|"
                          r"chap\.|vol\.|§ ?\d|\bpage \d)", re.I)
LEAD_NUM_RX = re.compile(r"^\s*[\(\[]?\d{1,4}[\)\].:]?\s*")


def clean_line(line):
    t = line.strip()
    t = LEAD_NUM_RX.sub("", t)                       # entry numbers
    t = re.sub(r"\s+", " ", t).strip(" -–—*•|")
    return t


def plausible(t):
    if not t:
        return False
    words = t.split()
    if not (3 <= len(words) <= 40):
        return False
    letters = sum(ch.isalpha() for ch in t)
    if letters / max(1, len(t)) < 0.65:              # OCR junk / tables
        return False
    if t.upper() == t:                                # page headers
        return False
    if ROMAN_RX.match(t):
        return False
    caps = sum(1 for w in words if w[:1].isupper())
    if caps / len(words) > 0.6 and len(words) > 4:    # title-case headings
        return False
    # OCR commentary defence: demand a complete sentence shape
    if not t[0].isupper():
        return False
    if t[-1] not in ".!?":
        return False
    if len(words) > 25:
        return False
    if SCHOLARLY_RX.search(t):
        return False
    if t.count("(") != t.count(")"):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--people", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true", help="print sample, insert nothing")
    a = ap.parse_args()

    init_db()
    r = requests.get(a.url, timeout=120, headers={"User-Agent": "WisdomExtractor/1.0 (research; polite)"})
    r.raise_for_status()
    text = r.text

    rows, seen = [], set()
    for raw in text.splitlines():
        t = clean_line(raw)
        if not plausible(t):
            continue
        t = strip_citations(t)
        if not keep(t) or t.lower() in seen:
            continue
        seen.add(t.lower())
        rows.append({"text": t, "people": a.people, "url": a.url,
                     "first_seen": a.year, "quality_score": quality_score(t)})

    if a.dry_run:
        import random
        random.seed(1)
        print(f"candidates: {len(rows)} — sample:")
        for s in random.sample(rows, min(12, len(rows))):
            print("  -", s["text"][:100])
        return

    sid = upsert_source(a.name, a.url, f"archive,dated,{a.people.lower()},non-european")
    for row in rows:
        row["source_id"] = sid
    n = bulk_insert_proverbs(rows)
    print(f"{a.name}: {n} inserted ({len(rows) - n} duplicates skipped) of {len(rows)} candidates")


if __name__ == "__main__":
    main()
