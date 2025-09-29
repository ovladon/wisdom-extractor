
import time, re, os, pandas as pd, yaml, requests
from bs4 import BeautifulSoup
import datetime as dt
from proverbs_cleaner import clean_dataframe

HEADERS = {"User-Agent":"WisdomExtractor/0.5 (research; polite; contact: you@example.com)"}
STOP_IDS = {"References","External_links","See_also","External_links_and_references"}

def _is_in_stopped_section(node):
    h = node.find_previous(["h2","h3"])
    if h:
        span = h.find("span", {"class":"mw-headline"})
        if span and span.get("id") in STOP_IDS:
            return True
    return False

def scrape_wikiquote(url, people, sleep=1.0):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    content = soup.select_one("#mw-content-text .mw-parser-output") or soup
    items, seen = [], set()
    for li in content.select("li"):
        try:
            txt = li.get_text(" ", strip=True)
        except Exception:
            continue
        if not txt or txt in seen: 
            continue
        if _is_in_stopped_section(li):
            continue
        if re.search(r"(?i)\b(What links here|Navigation menu|Text is available under|Creative Commons|Terms of Use|Privacy Policy|Retrieved from|Category:)\b", txt):
            continue
        if re.search(r"(?i)\b(ISBN|ISSN)\b", txt) or re.search(r"(?i)\bp\.\s*\d+\b", txt):
            continue
        if len(txt.split()) < 3:
            continue
        items.append(txt); seen.add(txt)
    rows = []
    now = time.strftime("%Y-%m-%d")
    for it in items:
        rows.append({"people": people,"original": it,"saying": it,"english_equivalent": "",
                     "source_title": f"Wikiquote: {people}","source_url": url,"accessed": now})
    return pd.DataFrame(rows)

def scrape_wiktionary_category(url, people, sleep=1.0):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    items = []
    for li in soup.select("#mw-pages li a[title]"):
        title = li.get_text(" ", strip=True)
        if not title: continue
        items.append(title)
    rows = []
    now = time.strftime("%Y-%m-%d")
    for it in items:
        rows.append({"people": people,"original": it,"saying": it,"english_equivalent": "",
                     "source_title": f"Wiktionary Category: {people}","source_url": url,"accessed": now})
    return pd.DataFrame(rows)

def scrape_wiktionary_appendix(url, people, sleep=1.0):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    items = []
    for li in soup.select("#mw-content-text .mw-parser-output li"):
        txt = li.get_text(" ", strip=True)
        if not txt: continue
        if re.search(r"(?i)\b(edit|what links here|permanent link|download as pdf)\b", txt):
            continue
        if len(txt.split()) < 3: continue
        items.append(txt)
    rows = []
    now = time.strftime("%Y-%m-%d")
    for it in items:
        rows.append({"people": people,"original": it,"saying": it,"english_equivalent": "",
                     "source_title": f"Wiktionary Appendix: {people}","source_url": url,"accessed": now})
    return pd.DataFrame(rows)

def scrape_gutenberg_html(url, people, sleep=1.0):
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    text = soup.get_text("\n", strip=True)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    items = []
    for ln in lines:
        if len(ln.split()) < 3 or len(ln.split()) > 20: continue
        if re.search(r"(?i)\b(Contents|Index|CHAPTER|ADVERTISEMENT|Publisher|London:|Copyright|Project Gutenberg)\b", ln):
            continue
        if re.search(r"[.!?;:]", ln):
            items.append(ln)
    rows = []
    now = time.strftime("%Y-%m-%d")
    for it in items[:5000]:
        rows.append({"people": people,"original": it,"saying": it,"english_equivalent": "",
                     "source_title": f"Gutenberg: {people}","source_url": url,"accessed": now})
    return pd.DataFrame(rows)

def scrape_internet_archive_html(url, people, sleep=1.0):
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    items = []
    for li in soup.select("li"):
        t = li.get_text(" ", strip=True)
        if t and len(t.split()) > 3 and re.search(r"[.!?;:]", t):
            items.append(t)
    rows = []
    now = time.strftime("%Y-%m-%d")
    for it in items[:5000]:
        rows.append({"people": people,"original": it,"saying": it,"english_equivalent": "",
                     "source_title": f"InternetArchive: {people}","source_url": url,"accessed": now})
    return pd.DataFrame(rows)

SCRAPERS = {
    "wikiquote": scrape_wikiquote,
    "wiktionary_category": scrape_wiktionary_category,
    "wiktionary_appendix": scrape_wiktionary_appendix,
    "gutenberg_html": scrape_gutenberg_html,
    "internet_archive_html": scrape_internet_archive_html,
}

def build_from_sources(sources_yaml_path, selected_people=None, selected_types=None, sleep=1.0, save_dir="runs"):
    srcs = yaml.safe_load(open(sources_yaml_path, "r", encoding="utf-8"))
    frames = []
    os.makedirs(save_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    for rec in srcs:
        p = rec.get("people"); u = rec.get("url"); t = rec.get("type","wikiquote")
        if selected_people and p not in selected_people: 
            continue
        if selected_types and t not in selected_types:
            continue
        fn = SCRAPERS.get(t)
        if not fn:
            print(f"[WARN] Unknown source type: {t} for {p}")
            continue
        try:
            df = fn(u, p, sleep=sleep)
            out = os.path.join(save_dir, f"{t}_{p}_{stamp}.csv").replace(" ", "_")
            df.to_csv(out, index=False, encoding="utf-8")
            print(f"[INFO] Saved {len(df)} rows → {out}")
            frames.append(df)
            time.sleep(sleep)
        except Exception as e:
            print(f"[WARN] Failed {p} ({t}): {e}")
    if frames:
        all_df = pd.concat(frames, ignore_index=True)
    else:
        all_df = pd.DataFrame(columns=["people","original","saying","english_equivalent","source_title","source_url","accessed"])
    return all_df

def merge_and_clean(uploaded_frames, scraped_df, use_ai=False, model_path='auto'):
    frames = [f for f in uploaded_frames if f is not None]
    if scraped_df is not None and not scraped_df.empty:
        frames.append(scraped_df)
    if not frames:
        return pd.DataFrame(columns=["people","original","saying","english_equivalent","source_title","source_url","accessed"]), pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    clean = clean_dataframe(raw)
    if use_ai:
        try:
            from ai_filters import filter_rows_with_ai
            basis = clean.get("basis", clean.get("saying")).astype(str).tolist()
            mask = filter_rows_with_ai(basis, model_path=model_path)
            clean = clean[pd.Series(mask, index=clean.index)]
        except Exception as e:
            print(f"[WARN] AI filter unavailable or failed: {e}")
    def qscore(row):
        import re
        t = str(row.get("basis") or row.get("saying") or "")
        score = 0
        if re.search(r"[.!?;:]", t): score += 2
        if re.search(r"(?i)\b(if|then|never|always|should|must|avoid|prefer|time|practice|friend|truth|honesty|fortune|blood|work)\b", t): score += 2
        if len(t.split()) >= 5: score += 1
        if len(t) >= 20: score += 1
        return score
    clean["quality_score"] = clean.apply(qscore, axis=1)
    clean = clean.sort_values(["quality_score","people"], ascending=[False, True])
    return raw, clean
