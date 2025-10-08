import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit, parse_qsl, urlencode, unquote
import urllib.robotparser as robotparser
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import time

HTTP_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)
ROBOTS_UA = "*"

TIMEOUT = 20
CONNECT_RETRIES = 2
SLOW_DOWN = 0.5

MEDIAWIKI_HOSTS = ("wikipedia.org", "wikiquote.org", "wiktionary.org")


def _is_mediawiki(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(h in host for h in MEDIAWIKI_HOSTS)


def _mw_title_from_url(url: str) -> str:
    path = urlparse(url).path
    if "/wiki/" in path:
        title = path.split("/wiki/", 1)[1]
    else:
        title = path.strip("/")
    title = unquote(title)
    title = title.replace("_", " ")
    return title or "Main Page"


def _mw_api_base(url: str) -> str:
    u = urlparse(url)
    return f"{u.scheme}://{u.netloc}/w/api.php"


def _add_render(url: str) -> str:
    parts = list(urlsplit(url))
    q = dict(parse_qsl(parts[3]))
    if "action" not in q:
        q["action"] = "render"
        parts[3] = urlencode(q, doseq=True)
    return urlunsplit(parts)


def _can_fetch(root, url, respect_robots=True):
    if not respect_robots:
        return True
    try:
        rp = robotparser.RobotFileParser()
        base = f"{urlparse(root).scheme}://{urlparse(root).netloc}"
        rp.set_url(urljoin(base, "/robots.txt"))
        rp.read()
        return rp.can_fetch(ROBOTS_UA, url)
    except Exception:
        return True


def _get(url, params=None):
    last_exc = None
    for i in range(CONNECT_RETRIES):
        try:
            r = requests.get(url, params=params, headers={"User-Agent": HTTP_UA}, timeout=TIMEOUT)
            if r.status_code == 200 and r.text:
                return r
            if r.status_code in (429, 503):
                time.sleep(SLOW_DOWN * (i + 1))
            else:
                break
        except Exception as e:
            last_exc = e
            time.sleep(SLOW_DOWN * (i + 1))
    if last_exc:
        raise last_exc
    raise requests.HTTPError(f"GET {url} failed")


def fetch(url):
    r = _get(url)
    return r.text, r.headers.get("Content-Type", "")


def _soup_for(html: str, content_type: str | None = None):
    ct = (content_type or "").lower()
    head = (html or "")[:500].lstrip()
    looks_xml = ("xml" in ct) or head.startswith("<?xml") or head.lower().startswith("<rss") or bool(re.search(r"<\w+:\w+", head))
    return BeautifulSoup(html, features=("xml" if looks_xml else "lxml"))


def _mw_api_parse(url):
    api = _mw_api_base(url)
    title = _mw_title_from_url(url)
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
        "formatversion": "2",
    }
    r = _get(api, params=params)
    j = r.json()
    html = j.get("parse", {}).get("text", "")
    return html, "text/html; charset=UTF-8"


def _mw_api_category_members(url, limit=500):
    api = _mw_api_base(url)
    title = _mw_title_from_url(url)
    if not title.lower().startswith("category:"):
        return []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": title,
        "cmlimit": str(limit),
        "format": "json",
    }
    r = _get(api, params=params)
    j = r.json()
    members = j.get("query", {}).get("categorymembers", [])
    out = []
    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}/wiki/"
    for m in members:
        t = m.get("title")
        if t:
            out.append(base + t.replace(" ", "_"))
    return out


def _clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t or "").strip()
    # strip trailing " — author" style attribution if present
    t = re.split(r"\s+[—–-]\s+[^,;:.]{2,}$", t)[0].strip()
    return t


EXCLUDE_PREFIXES = {"see also", "references", "external links", "notes", "bibliography", "further reading"}


def looks_like_proverb(t: str) -> bool:
    if not t or len(t.split()) < 2:
        return False
    if len(t) > 800:
        return False
    low = t.lower()
    if any(low.startswith(pfx) for pfx in EXCLUDE_PREFIXES):
        return False
    if re.fullmatch(r"\[\w+\]", t):
        return False
    return True


def _extract_mediawiki_items(page_url, soup):
    root = soup.select_one("#mw-content-text") or soup.select_one(".mw-parser-output") or soup
    items = []
    for li in root.select("ul > li, ol > li"):
        txt = _clean_text(li.get_text(" ", strip=True))
        if looks_like_proverb(txt):
            items.append({"text": txt, "url": page_url})
    if not items:
        for dd in root.select("dl > dd"):
            txt = _clean_text(dd.get_text(" ", strip=True))
            if looks_like_proverb(txt):
                items.append({"text": txt, "url": page_url})
    if not items:
        for p in root.select("p"):
            txt = _clean_text(p.get_text(" ", strip=True))
            if looks_like_proverb(txt):
                items.append({"text": txt, "url": page_url})
    seen = set()
    out = []
    for it in items:
        if it["text"] not in seen:
            seen.add(it["text"])
            out.append(it)
    return out


def extract_items(page_url, html, content_type=""):
    soup = _soup_for(html, content_type)
    if _is_mediawiki(page_url):
        items = _extract_mediawiki_items(page_url, soup)
        if not items:
            # try API parse, then action=render
            try:
                html2, ct2 = _mw_api_parse(page_url)
                soup2 = _soup_for(html2, ct2)
                items = _extract_mediawiki_items(page_url, soup2)
            except Exception:
                try:
                    html3, ct3 = fetch(_add_render(page_url))
                    soup3 = _soup_for(html3, ct3)
                    items = _extract_mediawiki_items(page_url, soup3)
                except Exception:
                    items = []
        return items

    # generic fallback
    items = []
    for li in soup.find_all("li"):
        txt = _clean_text(li.get_text(" ", strip=True))
        if len(txt) >= 4 and len(txt.split()) >= 2 and len(txt) <= 800:
            items.append({"text": txt, "url": page_url})
    if not items:
        for p in soup.find_all("p"):
            txt = _clean_text(p.get_text(" ", strip=True))
            if len(txt) >= 4 and len(txt.split()) >= 2 and len(txt) <= 800:
                items.append({"text": txt, "url": page_url})
    seen, out = set(), []
    for it in items:
        if it["text"] not in seen:
            seen.add(it["text"])
            out.append(it)
    return out


def discover_links(root_url, html, content_type=""):
    soup = _soup_for(html, content_type)
    links = set()
    host = urlparse(root_url).netloc
    if _is_mediawiki(root_url):
        title = _mw_title_from_url(root_url)
        if title.lower().startswith("category:"):
            try:
                return _mw_api_category_members(root_url, limit=500)
            except Exception:
                pass
        for a in soup.select(".mw-parser-output a[href]"):
            href = a.get("href", "")
            full = urljoin(root_url, href)
            u = urlparse(full)
            if u.netloc != host:
                continue
            if "/wiki/Special:" in u.path or "/wiki/Talk:" in u.path or u.path.startswith("/w/"):
                continue
            if u.path.startswith("/wiki/"):
                links.add(full)
        return list(links)

    # non-mediawiki heuristic: keep only proverb-ish links on same host
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        full = urljoin(root_url, a["href"])
        if urlparse(full).netloc != host:
            continue
        if any(k in href for k in ["proverb", "proverbs", "saying", "sayings", "adage", "aphorism", "idiom"]):
            links.add(full)
    return list(links)


def crawl_source(root_url, respect_robots=True, workers=8):
    """Depth-1 crawl: root page + all discovered links on that domain."""
    if not _can_fetch(root_url, root_url, respect_robots):
        return [], []
    try:
        html, ct = fetch(root_url)
    except Exception:
        html, ct = ("", "")
        if _is_mediawiki(root_url):
            try:
                html, ct = _mw_api_parse(root_url)
            except Exception:
                html, ct = fetch(_add_render(root_url))
    root_items = extract_items(root_url, html, ct)
    links = discover_links(root_url, html, ct)
    pages = []
    items = list(root_items)

    def _fetch_and_extract(link):
        if respect_robots and not _can_fetch(root_url, link, True):
            return None, []
        try:
            h, ct2 = fetch(link)
        except Exception:
            if _is_mediawiki(link):
                try:
                    h, ct2 = _mw_api_parse(link)
                except Exception:
                    try:
                        h, ct2 = fetch(_add_render(link))
                    except Exception:
                        return None, []
            else:
                return None, []
        its = extract_items(link, h, ct2)
        return link, its

    if links:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = {ex.submit(_fetch_and_extract, link): link for link in links}
            for fut in as_completed(futs):
                link = futs[fut]
                try:
                    page, its = fut.result()
                    if page:
                        pages.append(page)
                        items.extend(its)
                except Exception:
                    continue
    return pages, items


__all__ = ["crawl_source", "discover_links", "extract_items"]

