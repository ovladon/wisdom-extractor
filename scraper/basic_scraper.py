import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import urllib.robotparser as robotparser
from concurrent.futures import ThreadPoolExecutor, as_completed

UA = 'Mozilla/5.0 (compatible; WisdomLabBot/0.2; +https://example.org/wisdomlab)'
TIMEOUT = 12

def _can_fetch(root, url, respect_robots=True):
    if not respect_robots:
        return True
    try:
        rp = robotparser.RobotFileParser()
        base = f"{urlparse(root).scheme}://{urlparse(root).netloc}"
        rp.set_url(urljoin(base, "/robots.txt"))
        rp.read()
        return rp.can_fetch(UA, url)
    except Exception:
        return True

def fetch(url):
    r = requests.get(url, headers={'User-Agent': UA}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text

def is_plausible_list_item(text):
    t = text.strip()
    return (len(t) >= 4 and len(t) <= 300 and len(t.split()) >= 2)

def discover_links(root_url, html):
    soup = BeautifulSoup(html, 'html.parser')
    links = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        full = urljoin(root_url, href)
        if urlparse(full).netloc != urlparse(root_url).netloc:
            continue
        hl = href.lower()
        if any(k in hl for k in ['proverb','proverbs','saying','sayings','adage','aphorism','idiom']):
            links.add(full)
    return list(links)

def extract_items(page_url, html):
    soup = BeautifulSoup(html, 'html.parser')
    items = []
    for li in soup.find_all('li'):
        txt = li.get_text(' ', strip=True)
        if is_plausible_list_item(txt):
            items.append({'text': txt, 'url': page_url})
    if not items:
        for p in soup.find_all('p'):
            txt = p.get_text(' ', strip=True)
            if is_plausible_list_item(txt):
                items.append({'text': txt, 'url': page_url})
    out, seen = [], set()
    for it in items:
        t = it['text']
        if t not in seen:
            seen.add(t); out.append(it)
    return out

def crawl_source(root_url, respect_robots=True, workers=8):
    if not _can_fetch(root_url, root_url, respect_robots):
        return [], []
    html = fetch(root_url)
    root_items = extract_items(root_url, html)
    links = discover_links(root_url, html)
    pages = []
    items = list(root_items)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(fetch, link): link for link in links if _can_fetch(root_url, link, respect_robots)}
        for fut in as_completed(futs):
            link = futs[fut]
            try:
                html2 = fut.result()
                pages.append(link)
                items.extend(extract_items(link, html2))
            except Exception:
                continue
    return pages, items
