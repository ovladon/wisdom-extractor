"""Wisdom Lab — mobile annotation backend (v19.5).

A tiny FastAPI service over the same SQLite database: serves the swipe UI
(mobile/index.html) and a four-endpoint JSON API. Judgments land in the same
`constraints` table the pipeline and reliability model already use.

Run:  uvicorn mobile.mobile_api:app --host 0.0.0.0 --port 8600
Env:  WISDOM_DB_PATH   shared database (same as the Streamlit apps)
      ANNOTATOR_CODE   optional access code (same semantics as the portal)
"""
import os, random, sys, threading, time
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core.persistence import (init_db, list_proverbs, add_constraint, mark_excluded,
                              list_constraints, stats, leaderboard, backfill_glosses)
from core.annotation_quality import aggregate_constraints, pairs_needing_review
from core.clustering import nearest_pairs

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.environ.get("ANNOTATOR_CODE", "")

# --- abuse guards ---
RATE_WINDOW = 60          # seconds
RATE_MAX = 90             # requests per window per client (fast annotators stay well under)
EXCLUDE_DAILY_MAX = 30    # "not a saying" reports per user per day
_hits = defaultdict(deque)
_excludes = defaultdict(deque)


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
    resp.headers["X-Frame-Options"] = "DENY"
    return resp

_pool = {"pairs": [], "by_id": {}, "built": 0.0}
_lock = threading.Lock()
POOL_TTL = 1800  # rebuild candidate pool every 30 min


def _check_code(code, request=None):
    supplied = code or (request.headers.get("x-access-code", "") if request is not None else "")
    if CODE and supplied != CODE:
        raise HTTPException(401, "wrong access code")


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
        _pool.update({"pairs": pos + neg, "by_id": by_id, "built": time.time()})


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


@app.get("/api/pair")
def get_pair(request: Request, strategy: str = "uncertain", code: str = ""):
    _check_code(code, request)
    _rate_check(request.client.host if request.client else "?")
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
    if strategy in ("uncertain", "disputed") and _pool["pairs"]:
        pa, pb, _ = random.choice(_pool["pairs"])
        a, b = _item(pa), _item(pb)
        if a and b:
            return {"a": a, "b": b, "strategy": "uncertain"}
    ids = random.sample(list(_pool["by_id"]), 2)
    return {"a": _item(ids[0]), "b": _item(ids[1]), "strategy": "random"}


class Judgment(BaseModel):
    a_id: int
    b_id: int
    label: str = ""   # exclude_a | exclude_b | (legacy: must | cannot)
    score: int | None = None   # Pelican scale: 4,3,2,1,0,-1
    user: str
    code: str = ""


@app.post("/api/judge")
def judge(j: Judgment, request: Request):
    _check_code(j.code, request)
    _rate_check(request.client.host if request.client else "?")
    user = (j.user or "(anon)").strip()[:40]
    if j.score is not None:
        if j.score not in (4, 3, 2, 1, 0, -1):
            raise HTTPException(400, "bad score")
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


@app.get("/api/me")
def me(request: Request, user: str, code: str = ""):
    _check_code(code, request)
    _rate_check(request.client.host if request.client else "?")
    user = user.strip()[:40]
    board = leaderboard()
    rank = next((i + 1 for i, r in enumerate(board) if r["user"] == user), None)
    mine = next((r for r in board if r["user"] == user), None)
    _, annotators = aggregate_constraints(list_constraints())
    rel = annotators.get(user, {}).get("reliability")
    s = stats()
    return {"total": (mine or {}).get("total", 0), "rank": rank,
            "reliability": rel, "leaderboard": board[:10],
            "corpus": {"proverbs": s["proverbs"], "peoples": s["peoples"],
                       "judgments": s["must"] + s["cannot"]}}
