"""Wisdom Lab — mobile annotation backend (v19.5).

A tiny FastAPI service over the same SQLite database: serves the swipe UI
(mobile/index.html) and a four-endpoint JSON API. Judgments land in the same
`constraints` table the pipeline and reliability model already use.

Run:  uvicorn mobile.mobile_api:app --host 0.0.0.0 --port 8600
Env:  WISDOM_DB_PATH   shared database (same as the Streamlit apps)
      ANNOTATOR_CODE   optional access code (same semantics as the portal)
"""
import os, random, re, secrets, sys, threading, time
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core.persistence import (init_db, list_proverbs, add_constraint, add_duplicate_report,
                              add_correction, claim_nickname, mark_sensitive, is_blocked,
                              mark_excluded, nickname_taken, suggest_nickname,
                              list_constraints, stats, leaderboard, backfill_glosses,
                              annotator_uid, all_settings)
from core.annotation_quality import aggregate_constraints, pairs_needing_review
from core.clustering import nearest_pairs

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.environ.get("ANNOTATOR_CODE", "")

# --- live settings ---
# Operational knobs live in the shared database so the Admin panel (which runs on the
# researcher's laptop) can retune the live server without a redeploy. Cached briefly:
# the values change by hand, minutes apart at most, but they are read on every request.
_cfg_cache = {"vals": {}, "built": 0.0}
CFG_TTL = 30


def _cfg():
    if _cfg_cache["vals"] and time.time() - _cfg_cache["built"] < CFG_TTL:
        return _cfg_cache["vals"]
    try:
        vals = {s["key"]: s["value"] for s in all_settings()}
    except Exception:
        vals = dict(_cfg_cache["vals"])      # database hiccup: keep serving the last values
    _cfg_cache.update(vals=vals, built=time.time())
    return vals


def _cfg_i(key, default):
    try:
        return int(float(_cfg().get(key, default)))
    except (TypeError, ValueError):
        return default


def _cfg_f(key, default):
    try:
        return float(_cfg().get(key, default))
    except (TypeError, ValueError):
        return default


def _cfg_b(key, default=False):
    v = str(_cfg().get(key, default)).strip().lower()
    return v in ("1", "true", "yes", "on")


# --- abuse guards ---
RATE_WINDOW = 60          # seconds
RATE_MAX = 90             # per-account default; overridable via settings
RATE_MAX_IP = 600         # per-address default: a classroom shares one address
EXCLUDE_DAILY_MAX = 30    # "not a saying" reports per user per day
FLAG_DAILY_MAX = 20       # adult-language hides per user per day
BAD_CODE_MAX = 20         # wrong access-code attempts per client per hour
_hits = defaultdict(deque)
_excludes = defaultdict(deque)
_flags = defaultdict(deque)
_bad_codes = defaultdict(deque)

# --- human check: finish the proverb ---
# Offered in the annotator's own language. The English-only list was a comprehension
# test wearing a bot check's clothes: most of the people we recruit are Romanian
# speakers, and being unable to complete an English idiom said nothing about whether
# they were human or whether they could judge proverbs.
HUMAN_CHALLENGES_RO = [
    ("Buturuga mică răstoarnă carul ____.", "mare"),
    ("Cine se scoală de dimineață, departe ____.", "ajunge"),
    ("Apa trece, pietrele ____.", "raman"),
    ("Nu lăsa pe mâine ce poți face ____.", "azi"),
    ("Prietenul la nevoie se ____.", "cunoaste"),
    ("Ai carte, ai ____.", "parte"),
    ("Vorba dulce mult ____.", "aduce"),
    ("Unde-s doi puterea ____.", "creste"),
    ("Corb la corb nu-și scoate ____.", "ochii"),
    ("Graba strică ____.", "treaba"),
    ("Ce ție nu-ți place, altuia nu-i ____.", "face"),
    ("Bate fierul cât e ____.", "cald"),
    ("Meseria e brățară de ____.", "aur"),
]
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


def _rate_check(key, limit):
    now = time.time()
    q = _hits[key]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, "slow down a little — try again in a minute")
    q.append(now)


def _rate_guard(request, user=""):
    """Two tiers, because one address is not one person.

    The per-account limit is fair use: no human judges faster than this, and it keeps
    a single account from monopolising the server. The per-address limit is far higher
    and exists only to stop scripted farming — a seminar room, a university network or
    a mobile carrier presents twenty-five annotators behind one address, and the old
    single per-address limit would have throttled the whole room to roughly forty
    judgments a minute between them, which is precisely the session we most want.
    """
    _rate_check("ip:" + _client_key(request), _cfg_i("rate_max_ip", RATE_MAX_IP))
    nick = (user or "").strip().lower()[:40]
    if len(nick) >= 2:
        _rate_check("acct:" + nick, _cfg_i("rate_max_account", RATE_MAX))


def _exclude_check(user):
    now = time.time()
    q = _excludes[user]
    while q and now - q[0] > 86400:
        q.popleft()
    if len(q) >= EXCLUDE_DAILY_MAX:
        raise HTTPException(429, "daily 'not a saying' limit reached — thank you, that's plenty")
    q.append(now)

def _flag_check(user):
    now = time.time()
    q = _flags[user]
    while q and now - q[0] > 86400:
        q.popleft()
    if len(q) >= FLAG_DAILY_MAX:
        raise HTTPException(429, "daily limit for hiding pairs reached — thank you, "
                                 "that's plenty for one day")
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

_judged = {"by_pair": {}, "built": 0.0}   # (a,b) -> set of uids
JUDGED_TTL = 60
CORROBORATE_EVERY = 5     # legacy fixed rate: 1 request in 5 seeks a second opinion


def _corroborate_p(jm):
    """Probability that this request serves a second opinion instead of a fresh pair.

    Legacy mode keeps the shipped fixed rate. Adaptive mode steers the corpus toward a
    target double-rated share, and the equilibrium is worth spelling out because it is
    not the intuitive one: if a fraction p of judgments corroborate, each converts one
    single-rated pair into a double-rated pair, while each of the remaining (1-p)
    creates a new single-rated pair. The double-rated share therefore settles at
    p/(1-p), so holding a target share t needs p = t/(1+t) — for t = 0.45 that is 0.31,
    not 0.45. A proportional term clears an existing backlog faster than equilibrium
    would, and the ceiling keeps at least 40% of judgments widening the corpus: a
    corpus that only deepens stops covering new cultures, which is the whole point.

    This changes only which pairs are offered. Nobody is shown a pair twice, nobody
    sees anyone else's answer, and the judgment itself is untouched — the double-rated
    set simply stops being the accident of who happened to collide.
    """
    every = max(1, _cfg_i("corroborate_every", CORROBORATE_EVERY))
    base = 1.0 / every
    if not _cfg_b("corroborate_adaptive", False):
        return base
    target = min(0.9, max(0.0, _cfg_f("corroborate_target", 0.45)))
    if not jm:
        return base
    observed = sum(1 for users in jm.values() if len(users) >= 2) / len(jm)
    p = target / (1.0 + target) + 0.5 * max(0.0, target - observed)
    return max(base, min(0.60, p))


def _pick_corroboration(jm, me):
    """Choose a singly-judged pair for a second opinion, tempering the concentration.

    Picking uniformly over pairs would inherit the corpus's own imbalance: the most
    prolific annotator is the sole rater of about a third of the backlog, so a third of
    all new agreement would be agreement with that one person. Picking uniformly over
    annotators would overcorrect and exhaust the small contributors in an afternoon.
    Weighting each annotator by the square root of their backlog sits between the two.
    """
    by_owner = {}
    for k, users in jm.items():
        if len(users) != 1 or me in users:
            continue
        if k[0] not in _pool["by_id"] or k[1] not in _pool["by_id"]:
            continue
        by_owner.setdefault(next(iter(users)), []).append(k)
    if not by_owner:
        return None
    owners = list(by_owner)
    weights = [len(by_owner[o]) ** 0.5 for o in owners]
    owner = random.choices(owners, weights=weights, k=1)[0]
    return random.choice(by_owner[owner])


def _judged_map():
    """pair -> set of annotators who judged it. Cheap at this scale, cached briefly."""
    if _judged["by_pair"] and time.time() - _judged["built"] < JUDGED_TTL:
        return _judged["by_pair"]
    m = {}
    for c in list_constraints():
        key = (min(int(c["a_id"]), int(c["b_id"])), max(int(c["a_id"]), int(c["b_id"])))
        m.setdefault(key, set()).add(c.get("user"))
    _judged.update(by_pair=m, built=time.time())
    return m


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


def _check_nickname(name, request):
    key = request.headers.get("x-annotator-key", "")
    if not claim_nickname(name, key):
        # Hand back a name that works. Someone one tap from their first judgment,
        # told only that they are wrong, frequently just leaves.
        raise HTTPException(409, {"error": "nickname_taken",
                                  "suggestion": suggest_nickname(name, key),
                                  "message": "that nickname is already taken on another "
                                             "device — here is a free one"})
    if is_blocked(annotator_uid(name)):
        raise HTTPException(403, "this account can no longer submit judgments — "
                                 "contact the researchers if you think this is a mistake")


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
        rows = [r for r in list_proverbs(excluded=False)
                if r.get("gloss") and not r.get("sensitive")
                and r.get("gloss_source") != "auto_unreviewed"]
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
    _rate_guard(request)
    if _map_cache["html"] is None or time.time() - _map_cache["built"] > MAP_TTL:
        import datetime
        import pandas as pd
        from fastapi.concurrency import run_in_threadpool  # noqa: F401 (sync route)
        from core.mapview import build_map_html
        rows = [r for r in list_proverbs(with_claims_only=True) if not r.get("sensitive")]
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
    _rate_guard(request)
    s = stats()
    return {"code_required": bool(CODE),
            "challenge_enabled": _cfg_b("challenge_enabled", False),
            "human_check_after": max(0, _cfg_i("human_check_after", 0)),
            "corpus": {"proverbs": s["proverbs"], "peoples": s["peoples"],
                       "judgments": s["must"] + s["cannot"]}}


def _norm_answer(s):
    """Compare without diacritics: someone typing 'cunoaste' for 'cunoaște' on a phone
    keyboard has demonstrably completed the proverb, which is the whole question."""
    import unicodedata
    s = unicodedata.normalize("NFD", (s or "").strip().lower())
    return "".join(c for c in s if not unicodedata.combining(c))


@app.get("/api/human")
def human_challenge(request: Request, lang: str = "en"):
    _rate_guard(request)
    cid = secrets.token_urlsafe(8)
    pool = HUMAN_CHALLENGES_RO if (lang or "").lower().startswith("ro") else HUMAN_CHALLENGES
    prompt, answer = random.choice(pool)
    _pending_challenges[cid] = (_norm_answer(answer), time.time() + 600)
    if len(_pending_challenges) > 5000:
        _prune({k: v[1] for k, v in _pending_challenges.items()})
    return {"challenge_id": cid, "prompt": prompt}


class HumanAnswer(BaseModel):
    challenge_id: str
    answer: str


@app.post("/api/human")
def human_verify(h: HumanAnswer, request: Request):
    _rate_guard(request)
    entry = _pending_challenges.pop(h.challenge_id, None)
    if not entry or entry[1] < time.time():
        raise HTTPException(400, "challenge expired — try again")
    if _norm_answer(h.answer) != entry[0]:
        raise HTTPException(400, "not quite — try another one")
    tok = secrets.token_urlsafe(24)
    _human_tokens[tok] = time.time() + HUMAN_TOKEN_TTL
    return {"token": tok}


@app.get("/api/pair")
def get_pair(request: Request, strategy: str = "uncertain", code: str = "", user: str = "",
             challenge: str = ""):
    _check_code(code, request)
    _rate_guard(request, user)
    _ensure_pool()

    nick = (user or "").strip()[:40]
    me = annotator_uid(nick) if len(nick) >= 2 else None
    jm = _judged_map()
    mine = {k for k, users in jm.items() if me in users} if me else set()

    def _key(a, b):
        return (min(a, b), max(a, b))

    def _fresh_for_me(a, b):
        return _key(a, b) not in mine

    # Someone sent this exact pair with "Judge this!". Their answer is not carried in
    # the link and is never returned here, so the second judgment stays independent;
    # the invitation only decides WHICH pair is seen, exactly like every other strategy.
    if challenge and _cfg_b("challenge_enabled", False):
        try:
            ca, cb = (int(x) for x in challenge.split("-", 1))
        except (ValueError, TypeError):
            ca = cb = 0
        if ca and cb and ca in _pool["by_id"] and cb in _pool["by_id"] \
                and _fresh_for_me(ca, cb):        # already judged it? fall through quietly
            a, b = _item(ca), _item(cb)
            if a and b:
                return {"a": a, "b": b, "strategy": "challenge"}

    # A pair somebody ELSE judged once, that I have not seen. It is new to this
    # annotator, so it never feels like repetition, but it turns a single judgment into
    # an independent double-rating — which is what alpha is computed from.
    if me and _pool["by_id"] and random.random() < _corroborate_p(jm):
        got = _pick_corroboration(jm, me)
        if got:
            a, b = _item(got[0]), _item(got[1])
            if a and b:
                return {"a": a, "b": b, "strategy": "corroborate"}

    if strategy == "disputed":
        agg, _ = aggregate_constraints(list_constraints())
        review = [p for p in pairs_needing_review(agg)
                  if p["a_id"] in _pool["by_id"] and p["b_id"] in _pool["by_id"]
                  and _fresh_for_me(p["a_id"], p["b_id"])]
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
            bucket = [x for x in (_pool["strata"].get(key) or [])
                      if _fresh_for_me(x[0], x[1])]
            if bucket:
                pa, pb, _sim = random.choice(bucket)
                a, b = _item(pa), _item(pb)
                if a and b:
                    return {"a": a, "b": b, "strategy": f"stratified:{key}"}
    for _ in range(12):                      # random fallback, still avoiding repeats
        ids = random.sample(list(_pool["by_id"]), 2)
        if _fresh_for_me(ids[0], ids[1]):
            return {"a": _item(ids[0]), "b": _item(ids[1]), "strategy": "random"}
    ids = random.sample(list(_pool["by_id"]), 2)
    return {"a": _item(ids[0]), "b": _item(ids[1]), "strategy": "random"}


_SOURCE_OK = re.compile(r"^[a-z0-9:=!?|_-]{1,40}$")


def _clean_source(s):
    """Routing provenance as reported by our own client: which strategy served this
    pair. It exists so agreement can be recomputed later with any strategy excluded —
    the check that a change in routing did not manufacture the agreement it reports.
    It never enters a score, so a wrong value costs an audit, not a result."""
    s = (s or "").strip().lower()[:40]
    return s if s and _SOURCE_OK.match(s) else None


def _clean_ms(v):
    """Deliberation time. Judgments arriving at machine speed and machine regularity
    are the only visible trace of automated answering, which otherwise looks like an
    unusually reliable annotator. Implausible values are dropped rather than stored."""
    try:
        v = int(v)
    except (TypeError, ValueError):
        return None
    return v if 0 < v < 3_600_000 else None


class Judgment(BaseModel):
    a_id: int
    b_id: int
    label: str = ""   # exclude_a | exclude_b | (legacy: must | cannot)
    score: int | None = None   # Pelican scale 4..-1; 5 = exact duplicate (stored as 4 + report)
    user: str
    code: str = ""
    source: str = ""           # routing strategy that served the pair (provenance only)
    decide_ms: int | None = None


@app.post("/api/judge")
def judge(j: Judgment, request: Request):
    _check_code(j.code, request)
    _rate_guard(request, j.user)
    _check_human(request)
    nick = (j.user or "").strip()[:40]
    if len(nick) < 2:
        raise HTTPException(400, "please set a name (2+ characters)")
    _check_nickname(nick, request)
    user = annotator_uid(nick)   # science sees only the pseudonym
    _judged["built"] = 0.0        # a new judgment must count straight away
    if j.score is not None:
        if j.score not in (5, 4, 3, 2, 1, 0, -1):
            raise HTTPException(400, "bad score")
        if j.score == 5:   # exact duplicate: strongest same-signal + dedup evidence,
            add_duplicate_report(j.a_id, j.b_id, user)   # without widening the IAA scale
            add_constraint(j.a_id, j.b_id, None, user, score=4,
                           source=_clean_source(j.source), decide_ms=_clean_ms(j.decide_ms))
        else:
            add_constraint(j.a_id, j.b_id, None, user, score=j.score,
                           source=_clean_source(j.source), decide_ms=_clean_ms(j.decide_ms))
    elif j.label in ("must", "cannot"):
        add_constraint(j.a_id, j.b_id, j.label, user,
                       source=_clean_source(j.source), decide_ms=_clean_ms(j.decide_ms))
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


class Flag(BaseModel):
    pid: int
    user: str = ""
    code: str = ""


@app.post("/api/flag")
def flag_adult(f: Flag, request: Request):
    """Annotator reports adult content: hidden from the game and the public map.
    Attributable and capped, so one bad actor cannot empty the corpus."""
    _rate_guard(request, f.user)
    _check_code(f.code, request)
    _check_human(request)
    nick = (f.user or "").strip()[:40]
    if len(nick) < 2:
        raise HTTPException(400, "please set a name (2+ characters)")
    _check_nickname(nick, request)
    uid = annotator_uid(nick)
    _flag_check(uid)
    mark_sensitive(f.pid, uid)
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
    _rate_guard(request, f.user)
    _check_code(f.code, request)
    _check_human(request)
    name = f.user.strip()
    if len(name) < 2:
        raise HTTPException(400, "please set a nickname first")
    _check_nickname(name, request)
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
    _rate_guard(request)
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
    _rate_guard(request, user)
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
