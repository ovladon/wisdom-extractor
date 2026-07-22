"""Wisdom Lab — mobile annotation backend (v19.5).

A tiny FastAPI service over the same SQLite database: serves the swipe UI
(mobile/index.html) and a four-endpoint JSON API. Judgments land in the same
`constraints` table the pipeline and reliability model already use.

Run:  uvicorn mobile.mobile_api:app --host 0.0.0.0 --port 8600
Env:  WISDOM_DB_PATH   shared database (same as the Streamlit apps)
      ANNOTATOR_CODE   optional access code (same semantics as the portal)
"""
import os, random, secrets, sys, threading, time
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core.persistence import (init_db, list_proverbs, add_constraint, add_duplicate_report,
                              add_correction,
                              mark_excluded,
                              list_constraints, stats, leaderboard, backfill_glosses,
                              annotator_uid)
from core.annotation_quality import aggregate_constraints, pairs_needing_review
from core.clustering import nearest_pairs

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.environ.get("ANNOTATOR_CODE", "")

# --- abuse guards ---
RATE_WINDOW = 60          # seconds
RATE_MAX = 90             # requests per window per client (fast annotators stay well under)
EXCLUDE_DAILY_MAX = 30    # "not a saying" reports per user per day
BAD_CODE_MAX = 20         # wrong access-code attempts per client per hour
_hits = defaultdict(deque)
_excludes = defaultdict(deque)
_bad_codes = defaultdict(deque)

# --- human check: finish the proverb ---
HUMAN_CHALLENGES = [
    ("An apple does not fall far from the ____.", "tree"),
    ("Where there is smoke, there is ____.", "fire"),
    ("Strike while the iron is ____.", "hot"),
    ("Better late than ____.", "never"),
    ("Actions speak louder than ____.", "words"),
    ("Don't count your chickens before they ____.", "hatch"),
    ("The early bird catches the ____.", "worm"),
    ("When in Rome, do as the Romans ____.", "do"),
    ("A bird in the hand is worth two in the ____.", "bush"),
    ("Practice makes ____.", "perfect"),
    ("Too many cooks spoil the ____.", "broth"),
    ("Look before you ____.", "leap"),
]
_pending_challenges = {}   # cid -> (answer, expiry)
_human_tokens = {}         # token -> expiry
HUMAN_TOKEN_TTL = 7 * 86400


def _client_key(request):
    # behind Caddy/any proxy the peer is the proxy; trust the first X-Forwarded-For hop
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "?"


def _prune(d):
    now = time.time()
    for k in [k for k, exp in d.items() if exp < now]:
        d.pop(k, None)


def _rate_check(key):
    now = time.time()
    q = _hits[key]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_MAX:
        raise HTTPException(429, "slow down a little — try again in a minute")
    q.append(now)


def _exclude_check(user):
    now = time.time()
    q = _excludes[user]
    while q and now - q[0] > 86400:
        q.popleft()
    if len(q) >= EXCLUDE_DAILY_MAX:
        raise HTTPException(429, "daily 'not a saying' limit reached — thank you, that's plenty")
    q.append(now)

app = FastAPI(title="Wisdom Lab mobile", docs_url=None, redoc_url=None)
init_db()


@app.middleware("http")
async def security_headers(request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path == "/api/pubstats":
        resp.headers["Access-Control-Allow-Origin"] = "*"
    if request.url.path == "/map":   # the public map may be embedded on our own sites
        resp.headers["Content-Security-Policy"] = (
            "frame-ancestors 'self' https://wisdomextractor.com https://*.wisdomextractor.com "
            "https://*.netlify.app")
    else:
        resp.headers["X-Frame-Options"] = "DENY"
    return resp

_pool = {"pairs": [], "by_id": {}, "built": 0.0, "strata": {}, "cycle": 0}
_lock = threading.Lock()
POOL_TTL = 1800  # rebuild candidate pool every 30 min


def _stratum(a, b, sim):
    """Annotation stratum (Pelican protocol): family x region x similarity band."""
    fa, fb = a.get("family"), b.get("family")
    ra, rb = a.get("region"), b.get("region")
    fam = "fam?" if not (fa and fb) else ("fam=" if fa == fb else "fam!")
    reg = "reg?" if not (ra and rb) else ("reg=" if ra == rb else "reg!")
    band = "hi" if sim >= 0.5 else "lo"
    return f"{fam}|{reg}|{band}"


def _check_code(code, request=None):
    supplied = code or (request.headers.get("x-access-code", "") if request is not None else "")
    if CODE and supplied != CODE:
        if request is not None:
            key = _client_key(request)
            now = time.time()
            q = _bad_codes[key]
            while q and now - q[0] > 3600:
                q.popleft()
            q.append(now)
            if len(q) > BAD_CODE_MAX:
                raise HTTPException(429, "too many wrong codes — try again later")
        raise HTTPException(401, "wrong access code")


def _check_human(request):
    tok = request.headers.get("x-human", "")
    _prune(_human_tokens)
    if tok not in _human_tokens:
        raise HTTPException(403, "human check required")


def _ensure_pool():
    with _lock:
        if _pool["pairs"] and time.time() - _pool["built"] < POOL_TTL:
            return
        backfill_glosses()          # ensure new rows are glossed
        rows = [r for r in list_proverbs(excluded=False) if r.get("gloss")]
        by_id = {r["id"]: r for r in rows}
        sample = random.sample(rows, min(2500, len(rows)))
        pos, neg = nearest_pairs([r["text"] for r in sample],
                                 [r["id"] for r in sample], k=6, hi=0.85, lo=0.30)
        strata = {}
        for pa, pb, sim in pos + neg:
            a, b = by_id.get(pa), by_id.get(pb)
            if a and b:
                strata.setdefault(_stratum(a, b, sim), []).append((pa, pb, sim))
        _pool.update({"pairs": pos + neg, "by_id": by_id, "built": time.time(),
                      "strata": strata, "cycle": 0})


def _item(pid):
    r = _pool["by_id"].get(pid)
    if not r:
        return None
    gloss = r.get("gloss") or r["text"]
    original = r["text"] if r["text"].strip() != gloss.strip() else None
    return {"id": r["id"], "gloss": gloss, "original": original,
            "people": r.get("people") or "culture unknown"}


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "index.html"))


@app.get("/manifest.json")
def manifest():
    return FileResponse(os.path.join(HERE, "manifest.json"))


@app.get("/privacy")
def privacy():
    return FileResponse(os.path.join(HERE, "privacy.html"))


_map_cache = {"html": None, "built": 0.0}
MAP_TTL = 6 * 3600


@app.get("/map")
def public_map(request: Request):
    """Public living map — no code required; regenerated from the database every 6h."""
    _rate_check(_client_key(request))
    if _map_cache["html"] is None or time.time() - _map_cache["built"] > MAP_TTL:
        import datetime
        import pandas as pd
        from fastapi.concurrency import run_in_threadpool  # noqa: F401 (sync route)
        from core.mapview import build_map_html
        rows = list_proverbs(with_claims_only=True)
        df = pd.DataFrame(rows)
        s = stats()
        _map_cache["html"] = build_map_html(df, meta={
            "proverbs": s["proverbs"], "peoples": s["peoples"],
            "judgments": s["must"] + s["cannot"],
            "generated": datetime.date.today().isoformat()})
        _map_cache["built"] = time.time()
    from fastapi.responses import HTMLResponse
    return HTMLResponse(_map_cache["html"])


@app.get("/api/config")
def config(request: Request):
    _rate_check(_client_key(request))
    s = stats()
    return {"code_required": bool(CODE),
            "corpus": {"proverbs": s["proverbs"], "peoples": s["peoples"],
                       "judgments": s["must"] + s["cannot"]}}


@app.get("/api/human")
def human_challenge(request: Request):
    _rate_check(_client_key(request))
    cid = secrets.token_urlsafe(8)
    prompt, answer = random.choice(HUMAN_CHALLENGES)
    _pending_challenges[cid] = (answer, time.time() + 600)
    if len(_pending_challenges) > 5000:
        _prune({k: v[1] for k, v in _pending_challenges.items()})
    return {"challenge_id": cid, "prompt": prompt}


class HumanAnswer(BaseModel):
    challenge_id: str
    answer: str


@app.post("/api/human")
def human_verify(h: HumanAnswer, request: Request):
    _rate_check(_client_key(request))
    entry = _pending_challenges.pop(h.challenge_id, None)
    if not entry or entry[1] < time.time():
        raise HTTPException(400, "challenge expired — try again")
    if h.answer.strip().lower() != entry[0]:
        raise HTTPException(400, "not quite — try another one")
    tok = secrets.token_urlsafe(24)
    _human_tokens[tok] = time.time() + HUMAN_TOKEN_TTL
    return {"token": tok}


@app.get("/api/pair")
def get_pair(request: Request, strategy: str = "uncertain", code: str = ""):
    _check_code(code, request)
    _rate_check(_client_key(request))
    _ensure_pool()
    if strategy == "disputed":
        agg, _ = aggregate_constraints(list_constraints())
        review = [p for p in pairs_needing_review(agg)
                  if p["a_id"] in _pool["by_id"] and p["b_id"] in _pool["by_id"]]
        if review:
            p = random.choice(review[:15])
            a, b = _item(p["a_id"]), _item(p["b_id"])
            if a and b:
                return {"a": a, "b": b, "strategy": "disputed"}
    if strategy in ("uncertain", "disputed") and _pool["strata"]:
        keys = sorted(_pool["strata"])
        for _try in range(len(keys)):
            key = keys[_pool["cycle"] % len(keys)]
            _pool["cycle"] += 1
            bucket = _pool["strata"].get(key) or []
            if bucket:
                pa, pb, _sim = random.choice(bucket)
                a, b = _item(pa), _item(pb)
                if a and b:
                    return {"a": a, "b": b, "strategy": f"stratified:{key}"}
    ids = random.sample(list(_pool["by_id"]), 2)
    return {"a": _item(ids[0]), "b": _item(ids[1]), "strategy": "random"}


class Judgment(BaseModel):
    a_id: int
    b_id: int
    label: str = ""   # exclude_a | exclude_b | (legacy: must | cannot)
    score: int | None = None   # Pelican scale 4..-1; 5 = exact duplicate (stored as 4 + report)
    user: str
    code: str = ""


@app.post("/api/judge")
def judge(j: Judgment, request: Request):
    _check_code(j.code, request)
    _rate_check(_client_key(request))
    _check_human(request)
    nick = (j.user or "").strip()[:40]
    if len(nick) < 2:
        raise HTTPException(400, "please set a name (2+ characters)")
    user = annotator_uid(nick)   # science sees only the pseudonym
    if j.score is not None:
        if j.score not in (5, 4, 3, 2, 1, 0, -1):
            raise HTTPException(400, "bad score")
        if j.score == 5:   # exact duplicate: strongest same-signal + dedup evidence,
            add_duplicate_report(j.a_id, j.b_id, user)   # without widening the IAA scale
            add_constraint(j.a_id, j.b_id, None, user, score=4)
        else:
            add_constraint(j.a_id, j.b_id, None, user, score=j.score)
    elif j.label in ("must", "cannot"):
        add_constraint(j.a_id, j.b_id, j.label, user)
    elif j.label == "exclude_a":
        _exclude_check(user)
        mark_excluded(j.a_id, True)
        _pool["by_id"].pop(j.a_id, None)
    elif j.label == "exclude_b":
        _exclude_check(user)
        mark_excluded(j.b_id, True)
        _pool["by_id"].pop(j.b_id, None)
    else:
        raise HTTPException(400, "bad label")
    return {"ok": True}


class Fix(BaseModel):
    pid: int
    text: str
    user: str = ""
    code: str = ""


@app.post("/api/fix")
def suggest_fix(f: Fix, request: Request):
    """Annotator-suggested spelling/OCR correction; typo-sized fixes are applied by
    the weekly pipeline, larger rewrites wait for review."""
    _rate_check(_client_key(request))
    _check_code(f.code, request)
    _check_human(request)
    name = f.user.strip()
    if len(name) < 2:
        raise HTTPException(400, "please set a nickname first")
    t = " ".join(f.text.split())
    if not (5 <= len(t) <= 300):
        raise HTTPException(400, "that doesn't look like a saying")
    import difflib
    row = next((r for r in list_proverbs(excluded=False) if r["id"] == f.pid), None)
    if not row:
        raise HTTPException(404, "unknown proverb")
    if difflib.SequenceMatcher(None, row["text"].lower(), t.lower()).ratio() < 0.5:
        raise HTTPException(400, "a fix should stay close to the original wording")
    add_correction(f.pid, t, annotator_uid(name))
    return {"ok": True}


_pub_cache = {"data": None, "built": 0.0}


@app.get("/api/pubstats")
def pubstats(request: Request):
    """Public corpus counters for the landing page (1h cache)."""
    _rate_check(_client_key(request))
    if _pub_cache["data"] is None or time.time() - _pub_cache["built"] > 3600:
        from core.persistence import connect, stats as _stats
        st = _stats()
        con = connect()
        judgments = con.execute("SELECT COUNT(*) FROM constraints").fetchone()[0]
        dated = con.execute("SELECT COUNT(*) FROM proverbs WHERE excluded=0 "
                            "AND first_seen IS NOT NULL").fetchone()[0]
        con.close()
        _pub_cache.update(data={"proverbs": st["proverbs"], "peoples": st["peoples"],
                                "judgments": judgments, "dated": dated},
                          built=time.time())
    return _pub_cache["data"]


@app.get("/api/me")
def me(request: Request, user: str, code: str = ""):
    _check_code(code, request)
    _rate_check(_client_key(request))
    nick = user.strip()[:40]
    board = leaderboard()
    rank = next((i + 1 for i, r in enumerate(board) if r["user"] == nick), None)
    mine = next((r for r in board if r["user"] == nick), None)
    _, annotators = aggregate_constraints(list_constraints())
    rel = annotators.get(annotator_uid(nick), {}).get("reliability")
    s = stats()
    return {"total": (mine or {}).get("total", 0), "rank": rank,
            "reliability": rel, "leaderboard": board[:10],
            "corpus": {"proverbs": s["proverbs"], "peoples": s["peoples"],
                       "judgments": s["must"] + s["cannot"]}}
