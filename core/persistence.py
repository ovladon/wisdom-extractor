"""SQLite persistence layer (from Wisdom Lab v18, extended).

Adds to the v18 schema: people / claim / quality_score / cluster_id columns,
automatic in-place migration of older wisdom.db files, bulk inserts, and
a culture backfill that recovers the `people` label from source URLs.
"""
import sqlite3, time, os, hashlib, re, csv

DB_PATH = os.environ.get("WISDOM_DB_PATH", "wisdom.db")

PROVERB_COLUMNS = [
    ("source_id", "INTEGER"),
    ("text", "TEXT"),
    ("people", "TEXT"),          # culture label, e.g. "Romanian" (paper's `people`)
    ("language", "TEXT"),
    ("family", "TEXT"),          # language family (Germanic, Romance, ...)
    ("region", "TEXT"),
    ("original", "TEXT"),        # mother-tongue original if different from text
    ("claim", "TEXT"),           # canonicalized proposition (paper's `claim`)
    ("gloss", "TEXT"),           # English gloss shown to annotators (None = not annotatable)
    ("quality_score", "INTEGER"),
    ("cluster_id", "INTEGER"),
    ("first_seen", "INTEGER"),
    ("last_seen", "INTEGER"),
    ("url", "TEXT"),
    ("hash", "TEXT UNIQUE"),
    ("excluded", "INTEGER DEFAULT 0"),
    ("added_at", "REAL"),
]


def connect():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def init_db():
    con = connect(); cur = con.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS sources(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT, url TEXT UNIQUE, tags TEXT, enabled INTEGER DEFAULT 1, added_at REAL
    );
    CREATE TABLE IF NOT EXISTS proverbs(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      source_id INTEGER, text TEXT, hash TEXT UNIQUE
    );
    CREATE TABLE IF NOT EXISTS constraints(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      a_id INTEGER, b_id INTEGER, label TEXT, user TEXT, created_at REAL
    );
    CREATE TABLE IF NOT EXISTS users(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT UNIQUE, created_at REAL
    );
    CREATE TABLE IF NOT EXISTS annotators(
      uid TEXT PRIMARY KEY,          -- random pseudonym used in ALL science/exports
      nickname TEXT UNIQUE,          -- display-only (leaderboard); never exported
      created_at REAL
    );
    CREATE TABLE IF NOT EXISTS duplicate_reports(
      a_id INTEGER, b_id INTEGER, user TEXT, created_at REAL,
      UNIQUE(a_id, b_id, user)
    );
    """)
    # migrate: add any missing proverb columns (upgrades v15-v18 databases in place)
    cur.execute("PRAGMA table_info(proverbs)")
    have = {r[1] for r in cur.fetchall()}
    for col, typ in PROVERB_COLUMNS:
        if col not in have:
            cur.execute(f"ALTER TABLE proverbs ADD COLUMN {col} {typ}")
    # graded semantic-equivalence score (Pelican scale: 4..0 and -1), alongside legacy label
    cur.execute("PRAGMA table_info(constraints)")
    if "score" not in {r[1] for r in cur.fetchall()}:
        cur.execute("ALTER TABLE constraints ADD COLUMN score INTEGER")
    _pseudonymize_legacy_users(cur)
    cur.executescript("""
    CREATE INDEX IF NOT EXISTS idx_proverbs_excluded ON proverbs(excluded);
    CREATE INDEX IF NOT EXISTS idx_proverbs_people ON proverbs(people);
    CREATE INDEX IF NOT EXISTS idx_proverbs_cluster ON proverbs(cluster_id);
    """)
    con.commit(); con.close()


def _pseudonymize_legacy_users(cur):
    """One-time, idempotent: convert plain nicknames in constraints.user to uids."""
    import secrets
    cur.execute(r"SELECT DISTINCT user FROM constraints WHERE user IS NOT NULL AND user NOT LIKE 'u\_%' ESCAPE '\'")
    for (nick,) in cur.fetchall():
        cur.execute("SELECT uid FROM annotators WHERE nickname=?", (nick,))
        row = cur.fetchone()
        uid = row[0] if row else "u_" + secrets.token_hex(4)
        if not row:
            cur.execute("INSERT OR IGNORE INTO annotators(uid,nickname,created_at) VALUES(?,?,?)",
                        (uid, nick, time.time()))
        cur.execute("UPDATE constraints SET user=? WHERE user=?", (uid, nick))


def annotator_uid(nickname):
    """Stable random pseudonym for a display nickname (created on first use)."""
    import secrets
    nickname = (nickname or "(anon)").strip()[:40]
    con = connect(); cur = con.cursor()
    cur.execute("SELECT uid FROM annotators WHERE nickname=?", (nickname,))
    row = cur.fetchone()
    if row:
        uid = row[0]
    else:
        uid = "u_" + secrets.token_hex(4)
        cur.execute("INSERT INTO annotators(uid,nickname,created_at) VALUES(?,?,?)",
                    (uid, nickname, time.time()))
        con.commit()
    con.close()
    return uid


def nickname_of(uid):
    con = connect(); cur = con.cursor()
    cur.execute("SELECT nickname FROM annotators WHERE uid=?", (uid,))
    row = cur.fetchone(); con.close()
    return row[0] if row else uid


def _hash_text(t):
    return hashlib.sha256(str(t).strip().lower().encode("utf-8")).hexdigest()


# ---------- sources ----------

def upsert_source(name, url, tags=""):
    con = connect(); cur = con.cursor()
    cur.execute("INSERT OR IGNORE INTO sources(name,url,tags,added_at) VALUES(?,?,?,?)",
                (name, url, tags, time.time()))
    con.commit()
    cur.execute("SELECT id FROM sources WHERE url=?", (url,))
    row = cur.fetchone()
    con.close()
    return row[0] if row else None


def list_sources(enabled_only=False):
    con = connect(); cur = con.cursor()
    cur.execute("SELECT id,name,url,tags,enabled FROM sources" + (" WHERE enabled=1" if enabled_only else ""))
    rows = cur.fetchall(); con.close()
    return [{"id": r[0], "name": r[1], "url": r[2], "tags": r[3] or "", "enabled": bool(r[4])} for r in rows]


# ---------- proverbs ----------

def insert_proverb(source_id, text, url, people=None, language=None, family=None,
                   region=None, original=None, first_seen=None, last_seen=None):
    con = connect(); cur = con.cursor()
    try:
        cur.execute("""INSERT INTO proverbs(source_id,text,people,language,family,region,original,
                                            first_seen,last_seen,url,hash,added_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (source_id, text, people, language, family, region, original,
                     first_seen, last_seen, url, _hash_text(text), time.time()))
        con.commit(); pid = cur.lastrowid
    except sqlite3.IntegrityError:
        pid = None
    con.close()
    return pid


def bulk_insert_proverbs(rows):
    """rows: list of dicts with keys matching insert_proverb args. Returns count inserted."""
    con = connect(); cur = con.cursor()
    now = time.time(); inserted = 0
    for r in rows:
        try:
            cur.execute("""INSERT INTO proverbs(source_id,text,people,language,family,region,original,
                                                first_seen,last_seen,url,hash,quality_score,added_at)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (r.get("source_id"), r["text"], r.get("people"), r.get("language"),
                         r.get("family"), r.get("region"), r.get("original"),
                         r.get("first_seen"), r.get("last_seen"), r.get("url"),
                         _hash_text(r["text"]), r.get("quality_score"), now))
            inserted += 1
        except sqlite3.IntegrityError:
            continue
    con.commit(); con.close()
    return inserted


def list_proverbs(excluded=False, with_claims_only=False):
    con = connect(); cur = con.cursor()
    q = """SELECT id,text,people,language,family,region,original,claim,gloss,quality_score,
                  cluster_id,first_seen,last_seen,url,excluded FROM proverbs"""
    conds = []
    if not excluded:
        conds.append("excluded=0")
    if with_claims_only:
        conds.append("claim IS NOT NULL AND claim != ''")
    if conds:
        q += " WHERE " + " AND ".join(conds)
    cur.execute(q)
    rows = cur.fetchall(); con.close()
    keys = ["id", "text", "people", "language", "family", "region", "original", "claim", "gloss",
            "quality_score", "cluster_id", "first_seen", "last_seen", "url", "excluded"]
    return [dict(zip(keys, r)) for r in rows]


def mark_excluded(pid, excluded=True):
    con = connect(); cur = con.cursor()
    cur.execute("UPDATE proverbs SET excluded=? WHERE id=?", (1 if excluded else 0, pid))
    con.commit(); con.close()


def bulk_mark_excluded(pids, excluded=True):
    con = connect(); cur = con.cursor()
    cur.executemany("UPDATE proverbs SET excluded=? WHERE id=?",
                    [(1 if excluded else 0, int(p)) for p in pids])
    con.commit(); con.close()


def save_claims(id_claim_quality):
    """id_claim_quality: iterable of (id, claim, quality_score)."""
    con = connect(); cur = con.cursor()
    cur.executemany("UPDATE proverbs SET claim=?, quality_score=? WHERE id=?",
                    [(c, q, i) for i, c, q in id_claim_quality])
    con.commit(); con.close()


def save_clusters(id_cluster):
    con = connect(); cur = con.cursor()
    cur.execute("UPDATE proverbs SET cluster_id=NULL")
    cur.executemany("UPDATE proverbs SET cluster_id=? WHERE id=?",
                    [(int(c), int(i)) for i, c in id_cluster])
    con.commit(); con.close()


def set_people(id_people):
    con = connect(); cur = con.cursor()
    cur.executemany("UPDATE proverbs SET people=? WHERE id=?", [(p, int(i)) for i, p in id_people])
    con.commit(); con.close()


# ---------- culture backfill ----------

_URL_PEOPLE_RX = [
    re.compile(r"/wiki/(?:Category:)?([A-Za-z%C4%81%C4%93\-_]+?)_(?:proverbs|sayings)", re.I),
    re.compile(r"/wiki/List_of_([A-Za-z\-_]+?)_proverbs", re.I),
]


def infer_people_from_url(url, source_name=""):
    """Recover the culture label from a Wikiquote/Wiktionary/Wikipedia URL or source name."""
    for rx in _URL_PEOPLE_RX:
        m = rx.search(str(url) or "")
        if m:
            return m.group(1).replace("_", " ").replace("%C4%81", "ā").replace("%C4%93", "ē").strip().title()
    m = re.search(r"(?:–|\-|:)\s*([A-Za-zāē ]+?)\s+proverbs", str(source_name), re.I)
    if m:
        return m.group(1).strip().title()
    return None


def backfill_people_from_urls():
    """Fill missing `people` labels by parsing each row's page URL. Returns count updated."""
    con = connect(); cur = con.cursor()
    cur.execute("SELECT id,url FROM proverbs WHERE (people IS NULL OR people='') AND url IS NOT NULL")
    rows = cur.fetchall()
    updates = []
    for pid, url in rows:
        p = infer_people_from_url(url)
        if p:
            updates.append((p, pid))
    cur.executemany("UPDATE proverbs SET people=? WHERE id=?", updates)
    con.commit(); con.close()
    return len(updates)


def backfill_attestation_years(source_years_json=None):
    """Fill first_seen ('attested no later than' year) where missing.

    Priority: earliest citation year found in the raw `original` or `text`
    (e.g. 'von Düringsfield … (1875)'), else the publication year of a dated
    source matched by URL substring from data/source_years.json.
    Returns (n_from_citations, n_from_sources).
    """
    from .cleaner import extract_attestation_year
    src_years = {}
    if source_years_json and os.path.exists(source_years_json):
        raw = __import__("json").load(open(source_years_json, encoding="utf-8"))
        src_years = {k: v for k, v in raw.items() if isinstance(v, int)}
    con = connect(); cur = con.cursor()
    cur.execute("SELECT id, text, original, url FROM proverbs WHERE first_seen IS NULL")
    rows = cur.fetchall()
    from_cit, from_src = [], []
    for pid, text, original, url in rows:
        year = extract_attestation_year(original or "") or extract_attestation_year(text or "")
        if year:
            from_cit.append((year, pid))
            continue
        for frag, y in src_years.items():
            if url and frag in url:
                from_src.append((y, pid))
                break
    cur.executemany("UPDATE proverbs SET first_seen=? WHERE id=?", from_cit + from_src)
    con.commit(); con.close()
    return len(from_cit), len(from_src)


def backfill_glosses():
    """Compute the English gloss for rows missing one. Returns (n_glossed, n_unglossable)."""
    from .gloss import extract_gloss
    con = connect(); cur = con.cursor()
    cur.execute("SELECT id, text FROM proverbs WHERE gloss IS NULL")
    rows = cur.fetchall()
    updates, n_none = [], 0
    for pid, text in rows:
        g = extract_gloss(text or "")
        if g:
            updates.append((g, pid))
        else:
            n_none += 1
    cur.executemany("UPDATE proverbs SET gloss=? WHERE id=?", updates)
    con.commit(); con.close()
    return len(updates), n_none


def enrich_family_region(metadata_csv):
    """Fill family/region from a people metadata CSV (people,region,language_family,...)."""
    if not os.path.exists(metadata_csv):
        return 0
    meta = {}
    with open(metadata_csv, newline="", encoding="utf-8-sig") as f:
        content = f.read().lstrip("\n\r ")
    for row in csv.DictReader(content.splitlines()):
        if row.get("people"):
            meta[row["people"].strip().lower()] = (row.get("language_family", ""), row.get("region", ""))
    con = connect(); cur = con.cursor()
    cur.execute("SELECT id,people FROM proverbs WHERE people IS NOT NULL AND (family IS NULL OR family='')")
    updates = []
    for pid, people in cur.fetchall():
        hit = meta.get(str(people).strip().lower())
        if hit:
            updates.append((hit[0], hit[1], pid))
    cur.executemany("UPDATE proverbs SET family=?, region=? WHERE id=?", updates)
    con.commit(); con.close()
    return len(updates)


# ---------- constraints / annotation ----------

# Pelican graded scale -> hard clustering link:
#   4 identity / 3 functional equivalence -> must-link
#   2 same theme                          -> no hard link (stored as 'theme')
#   1 complementary / 0 unrelated / -1 contradiction -> cannot-link
SCORE_TO_LABEL = {4: "must", 3: "must", 2: "theme", 1: "cannot", 0: "cannot", -1: "cannot"}


def add_duplicate_report(a_id, b_id, user):
    """Human-verified word-for-word duplicate: feeds dedup/attestation merging,
    kept OUTSIDE the graded scale so IAA statistics stay on the -1..4 levels."""
    import time as _t
    a, b = (a_id, b_id) if a_id < b_id else (b_id, a_id)
    con = connect()
    con.execute("INSERT OR IGNORE INTO duplicate_reports(a_id,b_id,user,created_at) "
                "VALUES(?,?,?,?)", (a, b, user, _t.time()))
    con.commit(); con.close()


def list_duplicate_reports():
    con = connect(); con.row_factory = __import__("sqlite3").Row
    rows = [dict(r) for r in con.execute(
        "SELECT a_id, b_id, COUNT(DISTINCT user) AS reporters FROM duplicate_reports "
        "GROUP BY a_id, b_id ORDER BY reporters DESC")]
    con.close(); return rows


def add_constraint(a_id, b_id, label, user, score=None):
    if score is not None and label is None:
        label = SCORE_TO_LABEL.get(int(score), "theme")
    con = connect(); cur = con.cursor()
    cur.execute("INSERT INTO constraints(a_id,b_id,label,score,user,created_at) VALUES(?,?,?,?,?,?)",
                (a_id, b_id, label, score, user, time.time()))
    con.commit(); con.close()


def bulk_apply(pending_ops):
    con = connect(); cur = con.cursor()
    for op in pending_ops:
        if op.get("op") == "exclude":
            cur.execute("UPDATE proverbs SET excluded=1 WHERE id=?", (op["pid"],))
        elif op.get("op") == "constraint":
            cur.execute("INSERT INTO constraints(a_id,b_id,label,user,created_at) VALUES(?,?,?,?,?)",
                        (op["a"], op["b"], op["label"], op.get("user"), time.time()))
    con.commit(); con.close()


def list_constraints(label=None):
    con = connect(); cur = con.cursor()
    if label:
        cur.execute("SELECT a_id,b_id,label,score,user FROM constraints WHERE label=?", (label,))
    else:
        cur.execute("SELECT a_id,b_id,label,score,user FROM constraints")
    rows = cur.fetchall(); con.close()
    return [{"a_id": r[0], "b_id": r[1], "label": r[2], "score": r[3], "user": r[4]} for r in rows]


def stats():
    con = connect(); cur = con.cursor()
    out = {}
    for key, q in [
        ("proverbs", "SELECT COUNT(*) FROM proverbs WHERE excluded=0"),
        ("excluded", "SELECT COUNT(*) FROM proverbs WHERE excluded=1"),
        ("with_people", "SELECT COUNT(*) FROM proverbs WHERE excluded=0 AND people IS NOT NULL AND people!=''"),
        ("with_claim", "SELECT COUNT(*) FROM proverbs WHERE excluded=0 AND claim IS NOT NULL AND claim!=''"),
        ("clustered", "SELECT COUNT(*) FROM proverbs WHERE excluded=0 AND cluster_id IS NOT NULL"),
        ("peoples", "SELECT COUNT(DISTINCT people) FROM proverbs WHERE excluded=0 AND people IS NOT NULL"),
        ("must", "SELECT COUNT(*) FROM constraints WHERE label='must'"),
        ("cannot", "SELECT COUNT(*) FROM constraints WHERE label='cannot'"),
        ("duplicate_reports", "SELECT COUNT(*) FROM duplicate_reports"),
    ]:
        cur.execute(q); out[key] = cur.fetchone()[0]
    con.close()
    return out


def leaderboard(top=50):
    con = connect(); cur = con.cursor()
    cur.execute("""SELECT COALESCE(a.nickname, c.user),
                          SUM(CASE WHEN c.label='must' THEN 1 ELSE 0 END),
                          SUM(CASE WHEN c.label='cannot' THEN 1 ELSE 0 END),
                          COUNT(*)
                   FROM constraints c LEFT JOIN annotators a ON a.uid = c.user
                   GROUP BY c.user ORDER BY 4 DESC LIMIT ?""", (top,))
    rows = cur.fetchall(); con.close()
    return [{"user": r[0] or "(anon)", "must": r[1], "cannot": r[2], "total": r[3]} for r in rows]


def export_annotations():
    con = connect(); cur = con.cursor()
    cur.execute("SELECT a_id,b_id,label,score,user,created_at FROM constraints")
    cons = [{"a_id": r[0], "b_id": r[1], "label": r[2], "score": r[3], "user": r[4], "created_at": r[5]}
            for r in cur.fetchall()]
    cur.execute("SELECT id FROM proverbs WHERE excluded=1")
    excl = [r[0] for r in cur.fetchall()]
    con.close()
    return {"constraints": cons, "excluded_ids": excl, "meta": {"exported_at": time.time()}}
