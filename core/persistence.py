"""SQLite persistence layer (from Wisdom Lab v18, extended).

Adds to the v18 schema: people / claim / quality_score / cluster_id columns,
automatic in-place migration of older wisdom.db files, bulk inserts, and
a culture backfill that recovers the `people` label from source URLs.
"""
import sqlite3, time, os, hashlib, re, csv, secrets

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
    ("sensitive", "INTEGER DEFAULT 0"),   # adult language: kept in the corpus, hidden from public surfaces
    ("gloss_source", "TEXT"),             # collector | extracted | trimmed | auto_unreviewed | machine
    ("added_at", "REAL"),
    # Who removed a row from the active corpus, when, and why. Exclusion is the one
    # editorial act that changes what the published corpus contains, so it is the one
    # that most needs to be attributable and reversible.
    ("excluded_by", "TEXT"),
    ("excluded_at", "REAL"),
    ("exclude_reason", "TEXT"),
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
    CREATE TABLE IF NOT EXISTS sensitive_reports(
      pid INTEGER, user TEXT, created_at REAL,
      UNIQUE(pid, user)
    );
    CREATE TABLE IF NOT EXISTS corrections(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      pid INTEGER, old_text TEXT, new_text TEXT, user TEXT,
      created_at REAL, applied INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS settings(
      key TEXT PRIMARY KEY,          -- operational knobs, editable from the Admin panel
      value TEXT,                    -- so the live server changes behaviour without a redeploy
      updated_at REAL, updated_by TEXT
    );
    """)
    # migrate: add any missing proverb columns (upgrades v15-v18 databases in place)
    cur.execute("PRAGMA table_info(proverbs)")
    have = {r[1] for r in cur.fetchall()}
    for col, typ in PROVERB_COLUMNS:
        if col not in have:
            cur.execute(f"ALTER TABLE proverbs ADD COLUMN {col} {typ}")
    cur.execute("PRAGMA table_info(annotators)")
    _acols = {r[1] for r in cur.fetchall()}
    if "key_hash" not in _acols:
        cur.execute("ALTER TABLE annotators ADD COLUMN key_hash TEXT")
    if "blocked" not in _acols:
        cur.execute("ALTER TABLE annotators ADD COLUMN blocked INTEGER DEFAULT 0")
        cur.execute("ALTER TABLE annotators ADD COLUMN block_reason TEXT")
        cur.execute("ALTER TABLE annotators ADD COLUMN blocked_at REAL")
    # graded semantic-equivalence score (Pelican scale: 4..0 and -1), alongside legacy label
    cur.execute("PRAGMA table_info(constraints)")
    _ccols = {r[1] for r in cur.fetchall()}
    if "score" not in _ccols:
        cur.execute("ALTER TABLE constraints ADD COLUMN score INTEGER")
    # How the pair reached the annotator, and how long the judgment took. Both are
    # provenance, never inputs to a score: they let agreement be recomputed with any
    # routing strategy excluded, so a change in routing can be shown not to have
    # manufactured the agreement it reports. NULL = organic, pre-instrumentation.
    if "source" not in _ccols:
        cur.execute("ALTER TABLE constraints ADD COLUMN source TEXT")
    if "decide_ms" not in _ccols:
        cur.execute("ALTER TABLE constraints ADD COLUMN decide_ms INTEGER")
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


# ---------- settings ----------
# Operational knobs live in the database, not in module constants, because the
# Admin panel runs on the laptop while the annotation server runs elsewhere: a
# shared table is the only channel between them that does not require a redeploy.
# Every default here reproduces the behaviour that shipped before settings existed,
# so an empty table and a fresh install behave identically.

SETTING_DEFAULTS = {
    "corroborate_adaptive": "0",     # 0 = legacy fixed 1-in-N, 1 = aim at a target rate
    "corroborate_target": "0.45",    # wanted share of judged pairs carrying >= 2 raters
    "corroborate_every": "5",        # legacy fixed rate, and the floor/fallback
    "rate_max_account": "90",        # requests per minute per annotator
    "rate_max_ip": "600",            # requests per minute per address (bot guard)
    "human_check_after": "0",        # judgments allowed before the human check (0 = ask first)
    "challenge_enabled": "0",        # "Judge this!" per-pair invitations
}


def get_setting(key, default=None):
    """Read a knob. Falls back to SETTING_DEFAULTS, then to the supplied default."""
    con = connect()
    row = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    con.close()
    if row is not None and row[0] is not None:
        return row[0]
    return SETTING_DEFAULTS.get(key, default)


def get_setting_float(key, default=0.0):
    try:
        return float(get_setting(key))
    except (TypeError, ValueError):
        return default


def get_setting_int(key, default=0):
    try:
        return int(float(get_setting(key)))
    except (TypeError, ValueError):
        return default


def get_setting_bool(key, default=False):
    v = str(get_setting(key) or "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


def set_setting(key, value, user="admin"):
    con = connect()
    con.execute("INSERT INTO settings(key,value,updated_at,updated_by) VALUES(?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "updated_at=excluded.updated_at, updated_by=excluded.updated_by",
                (key, str(value), time.time(), user))
    con.commit(); con.close()


def all_settings():
    """Every knob with its effective value and whether it has been overridden."""
    con = connect()
    rows = {r[0]: (r[1], r[2], r[3]) for r in
            con.execute("SELECT key,value,updated_at,updated_by FROM settings").fetchall()}
    con.close()
    out = []
    for k, dflt in SETTING_DEFAULTS.items():
        v = rows.get(k)
        out.append({"key": k, "value": v[0] if v else dflt, "default": dflt,
                    "overridden": v is not None,
                    "updated_at": v[1] if v else None, "updated_by": v[2] if v else None})
    return out


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
                  cluster_id,first_seen,last_seen,url,excluded,
                  COALESCE(sensitive,0) AS sensitive, gloss_source FROM proverbs"""
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
            "quality_score", "cluster_id", "first_seen", "last_seen", "url", "excluded",
            "sensitive", "gloss_source"]
    return [dict(zip(keys, r)) for r in rows]


def mark_excluded(pid, excluded=True, user=None, reason=None):
    """Withdraw a row from the active corpus, or restore it.

    The row is never deleted — an excluded proverb stays in the database and in the
    provenance record, it simply stops being served, counted, clustered or mapped.
    That is what makes the act reversible, and it is why the corpus can be cleaned
    without the cleaning becoming an undocumented edit to the published data.
    """
    con = connect(); cur = con.cursor()
    if excluded:
        cur.execute("UPDATE proverbs SET excluded=1, excluded_by=?, excluded_at=?, "
                    "exclude_reason=? WHERE id=?", (user, time.time(), reason, pid))
    else:
        cur.execute("UPDATE proverbs SET excluded=0, excluded_by=NULL, excluded_at=NULL, "
                    "exclude_reason=NULL WHERE id=?", (pid,))
    con.commit(); con.close()


def bulk_mark_excluded(pids, excluded=True, user=None, reason=None):
    con = connect(); cur = con.cursor()
    if excluded:
        cur.executemany("UPDATE proverbs SET excluded=1, excluded_by=?, excluded_at=?, "
                        "exclude_reason=? WHERE id=?",
                        [(user, time.time(), reason, int(p)) for p in pids])
    else:
        cur.executemany("UPDATE proverbs SET excluded=0, excluded_by=NULL, excluded_at=NULL, "
                        "exclude_reason=NULL WHERE id=?", [(int(p),) for p in pids])
    con.commit(); con.close()


def list_recent_exclusions(limit=30):
    """Rows withdrawn by a named reviewer, newest first — the undo queue."""
    con = connect()
    rows = con.execute(
        "SELECT id, text, gloss, people, excluded_by, excluded_at, exclude_reason "
        "FROM proverbs WHERE excluded=1 AND excluded_at IS NOT NULL "
        "ORDER BY excluded_at DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return [{"id": r[0], "text": r[1], "gloss": r[2], "people": r[3],
             "excluded_by": r[4], "excluded_at": r[5], "exclude_reason": r[6]} for r in rows]


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


def trim_glosses_to_first_sentence():
    """Shorten existing glosses that trail past the saying into commentary.

    Only applies when the trimmed text is confidently English and materially
    shorter, so a genuine multi-clause proverb is left alone. Returns count.
    """
    from .gloss import first_english_sentence
    con = connect(); cur = con.cursor()
    n = 0
    for pid, gloss in cur.execute(
            "SELECT id, gloss FROM proverbs WHERE gloss IS NOT NULL AND gloss != ''"
            ).fetchall():
        short = first_english_sentence(gloss)
        if short and len(short) < len(gloss) * 0.75 and len(short.split()) >= 4:
            con.execute("UPDATE proverbs SET gloss=?, gloss_source='trimmed', claim=NULL "
                        "WHERE id=?", (short, pid))
            n += 1
    con.commit(); con.close()
    return n


def recover_glosses_unreviewed():
    """Find an English sentence for rows that have no gloss.

    Automatic identification is imperfect: romanised non-English, OCR damage and
    glossary definitions can pass as English. Recovered glosses are therefore marked
    `auto_unreviewed` and are NOT served to annotators until a human accepts them.
    Returns count.
    """
    from .gloss import first_english_sentence
    con = connect(); cur = con.cursor()
    n = 0
    for pid, text in cur.execute(
            "SELECT id, text FROM proverbs WHERE (gloss IS NULL OR gloss='')").fetchall():
        g = first_english_sentence(text)
        if g:
            con.execute("UPDATE proverbs SET gloss=?, gloss_source='auto_unreviewed' "
                        "WHERE id=?", (g, pid))
            n += 1
    con.commit(); con.close()
    return n


def review_gloss(pid, accept=True):
    """Accept an auto-recovered gloss into service, or discard it."""
    con = connect()
    if accept:
        con.execute("UPDATE proverbs SET gloss_source='extracted' WHERE id=?", (pid,))
    else:
        con.execute("UPDATE proverbs SET gloss=NULL, gloss_source=NULL WHERE id=?", (pid,))
    con.commit(); con.close()


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


def claim_nickname(nickname, device_key):
    """First-device-wins nickname binding. Empty key = legacy client: allowed, no claim.
    Returns True if the nickname is usable from this device, False if it belongs to
    another device."""
    if not device_key:
        return True
    kh = hashlib.sha256(device_key.encode("utf-8")).hexdigest()
    uid = annotator_uid(nickname)          # ensures the row exists
    con = connect(); cur = con.cursor()
    cur.execute("SELECT key_hash FROM annotators WHERE uid=?", (uid,))
    stored = (cur.fetchone() or [None])[0]
    ok = True
    if stored is None:
        cur.execute("UPDATE annotators SET key_hash=? WHERE uid=?", (kh, uid))
        con.commit()
    elif stored != kh:
        ok = False
    con.close()
    return ok


def nickname_taken(nickname, device_key):
    """True if this nickname is already bound to a different device. Read-only —
    unlike claim_nickname it creates no annotator row, so it is safe to call while
    hunting for a free suggestion."""
    if not device_key:
        return False
    kh = hashlib.sha256(device_key.encode("utf-8")).hexdigest()
    con = connect()
    row = con.execute("SELECT key_hash FROM annotators WHERE nickname=?", (nickname,)).fetchone()
    con.close()
    return bool(row) and row[0] is not None and row[0] != kh


def suggest_nickname(nickname, device_key, limit=40):
    """First free variant of a taken nickname: Vlad -> Vlad2, Vlad3, ...
    Bouncing someone back to the setup screen to invent a new name loses annotators
    who were one tap from their first judgment; handing them a working name does not."""
    base = (nickname or "").strip()[:36] or "annotator"
    if not nickname_taken(base, device_key):
        return base
    for n in range(2, limit + 2):
        cand = f"{base}{n}"
        if not nickname_taken(cand, device_key):
            return cand
    return f"{base}{secrets.randbelow(9000) + 1000}"


def add_duplicate_report(a_id, b_id, user):
    """Human-verified word-for-word duplicate: feeds dedup/attestation merging,
    kept OUTSIDE the graded scale so IAA statistics stay on the -1..4 levels."""
    import time as _t
    a, b = (a_id, b_id) if a_id < b_id else (b_id, a_id)
    con = connect()
    con.execute("INSERT OR IGNORE INTO duplicate_reports(a_id,b_id,user,created_at) "
                "VALUES(?,?,?,?)", (a, b, user, _t.time()))
    con.commit(); con.close()


# Unambiguous adult vocabulary. Deliberately NARROW: words like "ass" (donkey),
# "cock" (rooster) and "bitch" (female dog) are ordinary in historical proverb
# collections, so they are NOT listed here — human reports catch what this misses.
EXPLICIT_RX = re.compile(
    r"\b(fuck\w*|cunt\w*|shit\w*|shitty|piss\w*|whore\w*|slut\w*|penis|dick|dicks|"
    r"vagina|testicl\w*|semen|masturbat\w*|copulat\w*|fornicat\w*|prostitut\w*|"
    r"brothel|harlot\w*|buttocks|anus|erection|orgasm|genital\w*|arsehole|asshole)\b",
    re.I)


# Some digitised dictionary collections carry their editorial apparatus inline:
# an entry number, then a headword, then the saying, e.g.
#   "5. Adversity. Adversity makes a man wise, not rich."
#   "28 ] 476. Edged Tools. It is ill meddling with edged tools."
# The gloss presented for comparison holds the saying alone; the source string is
# always preserved unchanged in `text`.
#
# A leading entry number is required before a headword is removed, so ordinary
# two-sentence sayings such as "Money talks. Everyone listens." are untouched.
_ENTRY_RX = re.compile(r"^\s*(?:\d{1,4}\s*\]?\s*[.)]?\s*)+")
_HEADWORD_RX = re.compile(r"^([A-Z][A-Za-z]*(?:\s+[A-Za-z]+){0,2})\.\s+(?=[A-Z\"'])")
_ATTRIB_RX = re.compile(r"^(by|from|see|cf|after)\b", re.I)


def clean_dictionary_apparatus(text):
    """Strip entry numbers, an optional headword, and trailing citation debris.
    Returns (cleaned, changed)."""
    t = (text or "").strip()
    before = t
    m = _ENTRY_RX.match(t)
    if m and m.end() > 0:
        rest = t[m.end():].strip()
        h = _HEADWORD_RX.match(rest)
        t = rest[h.end():].strip() if h else rest
    t = re.sub(r"\s*\b(ibid\.|op\.\s?cit\.?|loc\.\s?cit\.?)\s*$", "", t, flags=re.I).strip()
    return t, (t != before)


def clean_glosses_apparatus(min_words=4):
    """Apply the above to every gloss. A result that no longer looks like a saying
    is left untouched rather than silently mangled. Returns (fixed, skipped)."""
    con = connect(); cur = con.cursor()
    fixed = skipped = 0
    for pid, gloss in cur.execute(
            "SELECT id, gloss FROM proverbs WHERE gloss IS NOT NULL AND gloss != ''"
            ).fetchall():
        new, changed = clean_dictionary_apparatus(gloss)
        if not changed:
            continue
        if (len(new.split()) < min_words or not re.search(r"[A-Za-z]{3}", new)
                or _ATTRIB_RX.match(new)):
            skipped += 1
            continue
        con.execute("UPDATE proverbs SET gloss=?, claim=NULL WHERE id=?", (new, pid))
        fixed += 1
    con.commit(); con.close()
    return fixed, skipped


def flag_sensitive_auto():
    """Mark proverbs containing unambiguous adult language. They stay in the corpus
    (removing them would censor the scientific record); they are simply not served
    to annotators or drawn on the public map. Idempotent; returns newly flagged."""
    con = connect(); cur = con.cursor()
    n = 0
    for pid, text, gloss in cur.execute(
            "SELECT id, text, COALESCE(gloss,'') FROM proverbs "
            "WHERE COALESCE(sensitive,0)=0").fetchall():
        if EXPLICIT_RX.search((text or "") + " " + (gloss or "")):
            con.execute("UPDATE proverbs SET sensitive=1 WHERE id=?", (pid,))
            n += 1
    con.commit(); con.close()
    return n


def mark_sensitive(pid, user=None):
    """Human report: hide a proverb from public surfaces, and record who asked.
    Every report is attributable so a single bad actor can be reverted wholesale."""
    con = connect()
    con.execute("UPDATE proverbs SET sensitive=1 WHERE id=?", (pid,))
    con.execute("INSERT OR IGNORE INTO sensitive_reports(pid,user,created_at) VALUES(?,?,?)",
                (pid, user or "(unknown)", time.time()))
    con.commit(); con.close()


def block_annotator(uid, reason=""):
    """Stop an account from submitting. Existing judgments are left untouched:
    removing data is a separate, deliberate decision (see purge_annotator)."""
    con = connect()
    con.execute("UPDATE annotators SET blocked=1, block_reason=?, blocked_at=? WHERE uid=?",
                (reason or "", time.time(), uid))
    con.commit(); con.close()


def unblock_annotator(uid):
    con = connect()
    con.execute("UPDATE annotators SET blocked=0, block_reason=NULL, blocked_at=NULL "
                "WHERE uid=?", (uid,))
    con.commit(); con.close()


def is_blocked(uid):
    con = connect()
    row = con.execute("SELECT COALESCE(blocked,0) FROM annotators WHERE uid=?", (uid,)).fetchone()
    con.close()
    return bool(row and row[0])


def list_annotators_admin():
    """Every account with volume, block state and reason, for the moderation table."""
    con = connect(); con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        """SELECT a.uid, a.nickname, COALESCE(a.blocked,0) AS blocked,
                  COALESCE(a.block_reason,'') AS block_reason, a.created_at,
                  (SELECT COUNT(*) FROM constraints c WHERE c.user = a.uid) AS judgments
           FROM annotators a ORDER BY judgments DESC""")]
    con.close(); return rows


def purge_annotator(uid):
    """Delete every judgment by one account. Destructive and irreversible: use when
    an account's data is untrustworthy, not merely unwanted. Returns rows removed."""
    con = connect(); cur = con.cursor()
    n = cur.execute("SELECT COUNT(*) FROM constraints WHERE user=?", (uid,)).fetchone()[0]
    cur.execute("DELETE FROM constraints WHERE user=?", (uid,))
    cur.execute("DELETE FROM duplicate_reports WHERE user=?", (uid,))
    cur.execute("DELETE FROM sensitive_reports WHERE user=?", (uid,))
    con.commit(); con.close()
    return n


ADJUDICATION_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "data", "adjudications.db")


def _adj_con():
    """Expert rulings live in their own file, not in the corpus database: the live
    snapshot is replaced on every sync, which would otherwise discard them."""
    con = sqlite3.connect(ADJUDICATION_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS adjudications(
        a_id INTEGER, b_id INTEGER, verdict TEXT, note TEXT, user TEXT,
        created_at REAL, UNIQUE(a_id, b_id, user))""")
    return con


def adjudicate_pair(a_id, b_id, verdict, note="", user="expert"):
    """Record an expert ruling on a contested pair.

    verdict: 'same' | 'different' | 'ambiguous'. 'ambiguous' is a substantive
    finding, not an abstention: it records that two informed readers can legitimately
    differ, which is the quantity of interest when equivalence is not determinate.
    Adjudications sit beside the crowd judgments and never overwrite them.
    """
    a, b = (a_id, b_id) if a_id < b_id else (b_id, a_id)
    con = _adj_con()
    con.execute("INSERT OR REPLACE INTO adjudications(a_id,b_id,verdict,note,user,created_at) "
                "VALUES(?,?,?,?,?,?)", (a, b, verdict, note, user, time.time()))
    con.commit(); con.close()


def adjudication_consensus():
    """Per pair: the rulings gathered, and whether the adjudicators concur.

    A pair on which several qualified adjudicators independently return different
    verdicts is evidence that the indeterminacy lies in the material rather than in
    any one reader.
    """
    from collections import defaultdict
    per = defaultdict(list)
    for r in list_adjudications():
        per[(r["a_id"], r["b_id"])].append(r["verdict"])
    out = []
    for (a, b), vs in per.items():
        agree = len(set(vs)) == 1
        out.append({"a_id": a, "b_id": b, "n_rulings": len(vs),
                    "verdicts": "|".join(sorted(vs)),
                    "unanimous": agree,
                    "verdict": vs[0] if agree else "contested-among-adjudicators"})
    return out


def list_adjudications():
    con = _adj_con(); con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT a_id,b_id,verdict,note,user,created_at FROM adjudications "
        "ORDER BY created_at DESC")]
    con.close(); return rows


def sensitive_reports_by_user():
    """Who has hidden how much, most recent first. For the admin review screen."""
    con = connect(); con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT user, COUNT(*) AS n, MAX(created_at) AS last_at "
        "FROM sensitive_reports GROUP BY user ORDER BY n DESC")]
    con.close(); return rows


def unflag_by_user(user):
    """Revert every hide requested by one annotator. Proverbs flagged by the
    automatic word list, or also reported by somebody else, stay hidden."""
    con = connect(); cur = con.cursor()
    pids = [r[0] for r in cur.execute(
        "SELECT pid FROM sensitive_reports WHERE user=?", (user,)).fetchall()]
    reverted = 0
    for pid in pids:
        others = cur.execute("SELECT COUNT(*) FROM sensitive_reports "
                             "WHERE pid=? AND user!=?", (pid, user)).fetchone()[0]
        row = cur.execute("SELECT text, COALESCE(gloss,'') FROM proverbs WHERE id=?",
                          (pid,)).fetchone()
        auto = bool(row) and bool(EXPLICIT_RX.search((row[0] or "") + " " + (row[1] or "")))
        if not others and not auto:
            cur.execute("UPDATE proverbs SET sensitive=0 WHERE id=?", (pid,))
            reverted += 1
    cur.execute("DELETE FROM sensitive_reports WHERE user=?", (user,))
    con.commit(); con.close()
    return reverted


def add_correction(pid, new_text, user):
    con = connect(); cur = con.cursor()
    cur.execute("SELECT text FROM proverbs WHERE id=?", (pid,))
    row = cur.fetchone()
    if row:
        cur.execute("INSERT INTO corrections(pid,old_text,new_text,user,created_at) "
                    "VALUES(?,?,?,?,?)", (pid, row[0], new_text, user, time.time()))
        con.commit()
    con.close()
    return bool(row)


def apply_corrections(sim_threshold=0.85):
    """Apply pending annotator corrections that are typo-sized (difflib >= threshold
    vs the CURRENT text). Bigger rewrites stay pending for manual review. The fixed
    row gets claim/gloss cleared so the pipeline re-derives them. If the corrected
    text collides with an existing row's hash, the row is a duplicate: excluded."""
    from difflib import SequenceMatcher
    con = connect(); cur = con.cursor()
    applied = 0
    for cid, pid, new_text in cur.execute(
            "SELECT id, pid, new_text FROM corrections WHERE applied=0").fetchall():
        row = con.execute("SELECT text, excluded FROM proverbs WHERE id=?", (pid,)).fetchone()
        if not row or row[1]:
            con.execute("UPDATE corrections SET applied=-1 WHERE id=?", (cid,)); continue
        if SequenceMatcher(None, row[0].lower(), new_text.lower()).ratio() < sim_threshold:
            continue
        try:
            con.execute("UPDATE proverbs SET text=?, hash=?, claim=NULL, gloss=NULL WHERE id=?",
                        (new_text, _hash_text(new_text), pid))
        except sqlite3.IntegrityError:
            con.execute("UPDATE proverbs SET excluded=1 WHERE id=?", (pid,))
        con.execute("UPDATE corrections SET applied=1 WHERE id=?", (cid,))
        applied += 1
    con.commit(); con.close()
    return applied


def fix_ocr_artifacts(min_good=10, max_bad=3):
    """Repair the classic OCR w->'iv' confusion ('ivants' -> 'wants', 'betiveen' ->
    'between') using the corpus as its own dictionary: a word is rewritten only when
    it is rare (<= max_bad occurrences) and the 'iv'->'w' variant is common
    (>= min_good). Fixed rows get claim/gloss cleared for re-derivation; a fix that
    collides with an existing row is a duplicate and gets excluded."""
    from collections import Counter
    con = connect(); cur = con.cursor()
    rows = cur.execute("SELECT id, text FROM proverbs WHERE excluded=0").fetchall()
    freq = Counter(w for _, t in rows for w in re.findall(r"[a-z]+", t.lower()))
    mapping = {}
    for w, c in freq.items():
        if "iv" in w and len(w) > 3 and c <= max_bad:
            cand = w.replace("iv", "w")
            if freq.get(cand, 0) >= min_good:
                mapping[w] = cand
    if not mapping:
        con.close(); return 0
    word_rx = re.compile(r"[A-Za-z]+")
    def repl(m):
        w = m.group(0); nw = mapping.get(w.lower())
        if not nw:
            return w
        return nw.capitalize() if w[0].isupper() else nw
    fixed = 0
    for pid, text in rows:
        new = word_rx.sub(repl, text)
        if new == text:
            continue
        try:
            con.execute("UPDATE proverbs SET text=?, hash=?, claim=NULL, gloss=NULL WHERE id=?",
                        (new, _hash_text(new), pid))
        except sqlite3.IntegrityError:
            con.execute("UPDATE proverbs SET excluded=1 WHERE id=?", (pid,))
        fixed += 1
    con.commit(); con.close()
    return fixed


def _norm_dedup(t):
    t = re.sub(r"[^a-z0-9\u00c0-\u024f\u0370-\u03ff\u0400-\u04ff ]+", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def dedup_normalized():
    """Auto-exclude twins whose texts are identical after case/punctuation
    normalization, within the same people only (cross-people twins are data).
    Keeper = earliest first_seen then lowest id; inherits the earliest year."""
    con = connect(); cur = con.cursor()
    groups = {}
    for pid, text, people, year in cur.execute(
            "SELECT id, text, people, first_seen FROM proverbs WHERE excluded=0").fetchall():
        groups.setdefault((people or "", _norm_dedup(text)), []).append((year or 9999, pid, year))
    excluded = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort()
        year = min((y for _, _, y in members if y), default=None)
        keeper = members[0][1]
        con.execute("UPDATE proverbs SET first_seen=COALESCE(?, first_seen) WHERE id=?",
                    (year, keeper))
        for _, pid, _ in members[1:]:
            con.execute("UPDATE proverbs SET excluded=1 WHERE id=?", (pid,))
            excluded += 1
    con.commit(); con.close()
    return excluded


def merge_reported_duplicates(min_reporters=2, sim_threshold=0.85):
    """Merge human-reported exact duplicates as attestations of one saying.

    Guards: only same-people pairs (identical text from two DIFFERENT peoples is a
    cross-cultural datum, never merged); a single reporter is trusted only when the
    texts really are near-identical (difflib >= sim_threshold), two+ reporters always.
    The keeper inherits the earliest attestation year (first_seen); the twin is excluded.
    Idempotent; returns number of merges performed.
    """
    from difflib import SequenceMatcher
    merged = 0
    con = connect(); con.row_factory = __import__("sqlite3").Row
    for rep in list_duplicate_reports():
        rows = {r["id"]: dict(r) for r in con.execute(
            "SELECT id, text, people, first_seen, excluded FROM proverbs WHERE id IN (?,?)",
            (rep["a_id"], rep["b_id"]))}
        if len(rows) != 2:
            continue
        a, b = rows[rep["a_id"]], rows[rep["b_id"]]
        if a["excluded"] or b["excluded"]:
            continue
        pa, pb = (a.get("people") or ""), (b.get("people") or "")
        if pa and pb and pa != pb:
            continue
        sim = SequenceMatcher(None, a["text"].lower(), b["text"].lower()).ratio()
        if rep["reporters"] < min_reporters and sim < sim_threshold:
            continue
        ya, yb = a.get("first_seen"), b.get("first_seen")
        keeper, twin = (a, b) if (ya or 9999, a["id"]) <= (yb or 9999, b["id"]) else (b, a)
        year = min([y for y in (ya, yb) if y], default=None)
        con.execute("UPDATE proverbs SET first_seen=COALESCE(?, first_seen) WHERE id=?",
                    (year, keeper["id"]))
        con.execute("UPDATE proverbs SET excluded=1 WHERE id=?", (twin["id"],))
        merged += 1
    con.commit(); con.close()
    return merged


def list_duplicate_reports():
    con = connect(); con.row_factory = __import__("sqlite3").Row
    rows = [dict(r) for r in con.execute(
        "SELECT a_id, b_id, COUNT(DISTINCT user) AS reporters FROM duplicate_reports "
        "GROUP BY a_id, b_id ORDER BY reporters DESC")]
    con.close(); return rows


def add_constraint(a_id, b_id, label, user, score=None, source=None, decide_ms=None):
    if score is not None and label is None:
        label = SCORE_TO_LABEL.get(int(score), "theme")
    con = connect(); cur = con.cursor()
    cur.execute("INSERT INTO constraints(a_id,b_id,label,score,user,created_at,source,decide_ms) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (a_id, b_id, label, score, user, time.time(), source, decide_ms))
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
        cur.execute("SELECT a_id,b_id,label,score,user,source,decide_ms,created_at "
                    "FROM constraints WHERE label=?", (label,))
    else:
        cur.execute("SELECT a_id,b_id,label,score,user,source,decide_ms,created_at FROM constraints")
    rows = cur.fetchall(); con.close()
    return [{"a_id": r[0], "b_id": r[1], "label": r[2], "score": r[3], "user": r[4],
             "source": r[5], "decide_ms": r[6], "created_at": r[7]} for r in rows]


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
    cur.execute("SELECT a_id,b_id,label,score,user,created_at,source,decide_ms FROM constraints")
    cons = [{"a_id": r[0], "b_id": r[1], "label": r[2], "score": r[3], "user": r[4],
             "created_at": r[5], "source": r[6], "decide_ms": r[7]}
            for r in cur.fetchall()]
    cur.execute("SELECT id FROM proverbs WHERE excluded=1")
    excl = [r[0] for r in cur.fetchall()]
    con.close()
    return {"constraints": cons, "excluded_ids": excl, "meta": {"exported_at": time.time()}}
