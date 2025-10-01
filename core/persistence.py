import sqlite3, time, json, os, hashlib

DB_PATH = os.environ.get("WISDOM_DB_PATH", "wisdom.db")

def connect():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    con = connect(); cur = con.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS sources(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT, url TEXT UNIQUE, tags TEXT, enabled INTEGER DEFAULT 1, added_at REAL
    );
    CREATE TABLE IF NOT EXISTS proverbs(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      source_id INTEGER, text TEXT, language TEXT, family TEXT, region TEXT,
      first_seen INTEGER, last_seen INTEGER, url TEXT, hash TEXT UNIQUE,
      excluded INTEGER DEFAULT 0, idea_formula TEXT, frame TEXT, added_at REAL
    );
    CREATE TABLE IF NOT EXISTS constraints(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      a_id INTEGER, b_id INTEGER, label TEXT, user TEXT, created_at REAL
    );
    CREATE TABLE IF NOT EXISTS users(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT UNIQUE, created_at REAL
    );
    """)
    con.commit(); con.close()

def _hash_text(t):
    return hashlib.sha256(t.strip().lower().encode("utf-8")).hexdigest()

def upsert_source(name, url, tags=""):
    con = connect(); cur = con.cursor()
    cur.execute("INSERT OR IGNORE INTO sources(name,url,tags,added_at) VALUES(?,?,?,?)",
                (name, url, tags, time.time()))
    con.commit()
    cur.execute("SELECT id FROM sources WHERE url=?", (url,))
    sid = cur.fetchone()[0]
    con.close()
    return sid

def list_sources(enabled_only=False):
    con = connect(); cur = con.cursor()
    cur.execute("SELECT id,name,url,tags,enabled FROM sources" + (" WHERE enabled=1" if enabled_only else ""))
    rows = cur.fetchall(); con.close()
    return [{"id":r[0],"name":r[1],"url":r[2],"tags":r[3] or "", "enabled":bool(r[4])} for r in rows]

def insert_proverb(source_id, text, url, language=None, family=None, region=None, first_seen=None, last_seen=None):
    h = _hash_text(text)
    con = connect(); cur = con.cursor()
    try:
        cur.execute("""INSERT INTO proverbs(source_id,text,language,family,region,first_seen,last_seen,url,hash,added_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (source_id, text, language, family, region, first_seen, last_seen, url, h, time.time()))
        con.commit()
        pid = cur.lastrowid
    except sqlite3.IntegrityError:
        pid = None
    con.close()
    return pid

def list_proverbs(excluded=False):
    con = connect(); cur = con.cursor()
    cur.execute("SELECT id,text,language,family,region,first_seen,last_seen,excluded,idea_formula,frame FROM proverbs" + (" WHERE excluded=0" if not excluded else ""))
    rows = cur.fetchall(); con.close()
    return [{
        "id":r[0], "text":r[1], "language":r[2], "family":r[3], "region":r[4],
        "first_seen":r[5], "last_seen":r[6], "excluded":bool(r[7]), "idea_formula":r[8], "frame":r[9]
    } for r in rows]

def mark_excluded(pid, excluded=True):
    con = connect(); cur = con.cursor()
    cur.execute("UPDATE proverbs SET excluded=? WHERE id=?", (1 if excluded else 0, pid))
    con.commit(); con.close()

def save_proposition(pid, idea_formula, frame):
    con = connect(); cur = con.cursor()
    cur.execute("UPDATE proverbs SET idea_formula=?, frame=? WHERE id=?", (idea_formula, frame, pid))
    con.commit(); con.close()

def add_constraint(a_id, b_id, label, user):
    con = connect(); cur = con.cursor()
    cur.execute("INSERT INTO constraints(a_id,b_id,label,user,created_at) VALUES(?,?,?,?,?)",
                (a_id, b_id, label, user, time.time()))
    con.commit(); con.close()

def stats():
    con = connect(); cur = con.cursor()
    con.execute = cur.execute  # back-compat typo guard
    cur.execute("SELECT COUNT(*) FROM proverbs WHERE excluded=0"); n_prov = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM constraints WHERE label='must'"); n_must = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM constraints WHERE label='cannot'"); n_cannot = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM proverbs WHERE excluded=1"); n_excl = cur.fetchone()[0]
    con.close()
    return {"proverbs": n_prov, "must": n_must, "cannot": n_cannot, "excluded": n_excl}

def leaderboard(top=50):
    con = connect(); cur = con.cursor()
    cur.execute("""SELECT user,
                          SUM(CASE WHEN label='must' THEN 1 ELSE 0 END) as musts,
                          SUM(CASE WHEN label='cannot' THEN 1 ELSE 0 END) as cannots,
                          COUNT(*) as total
                   FROM constraints GROUP BY user ORDER BY total DESC LIMIT ?""", (top,))
    rows = cur.fetchall(); con.close()
    return [{"user":r[0] or "(anon)", "must":r[1], "cannot":r[2], "total":r[3]} for r in rows]

def export_annotations():
    con = connect(); cur = con.cursor()
    cur.execute("SELECT a_id,b_id,label,user,created_at FROM constraints")
    cons = [{"a_id":r[0],"b_id":r[1],"label":r[2],"user":r[3],"created_at":r[4]} for r in cur.fetchall()]
    cur.execute("SELECT id FROM proverbs WHERE excluded=1")
    excl = [r[0] for r in cur.fetchall()]
    con.close()
    import time as _t
    return {"constraints": cons, "excluded_ids": excl, "meta": {"exported_at": _t.time()}}
