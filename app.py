"""Wisdom Extractor — Unified (v19)

Merges the ConsILR-2025 paper pipeline (v10_working: canonicalization, char n-gram
clustering, diagnostics, interpretation) with the Wisdom Lab platform (v18: robust
scraping, SQLite persistence, human annotation game), and fixes the defects of both.
Run:  streamlit run app.py
"""
import os, sys, json, random, warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # silence TensorFlow plugin chatter if installed
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden.*")
warnings.filterwarnings("ignore", message=".*TBB threading layer.*")

import pandas as pd
import streamlit as st

from core.persistence import (
    init_db, upsert_source, list_sources, insert_proverb, bulk_insert_proverbs,
    list_proverbs, mark_excluded, bulk_mark_excluded, save_claims, save_clusters,
    add_constraint, bulk_apply, list_constraints, stats, leaderboard,
    export_annotations, backfill_people_from_urls, enrich_family_region,
    infer_people_from_url, backfill_attestation_years, annotator_uid,
)
from core.cleaner import keep, quality_score, strip_citations, extract_attestation_year
from core.canonicalize import canonicalize
from core.clustering import cluster_texts, nearest_pairs, summarize_clusters
from core.diagnostics import (
    silhouette_cosine, bootstrap_stability, sensitivity_sweep,
    coverage_distribution, constraint_agreement, permutation_triangulation,
)
from core.annotation_quality import (
    aggregate_constraints, constraint_pairs_for_clustering, pairs_needing_review,
)
from core.interpret import themed_report, ecological_correlations, productive_tensions
from core.mapview import build_map_html
from core.projection import compute_coords
from scraper.basic_scraper import crawl_source

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
METADATA_CSV = os.path.join(DATA_DIR, "people_metadata.csv")
SEED_CSV = os.path.join(DATA_DIR, "seed_proverbs.csv")
CATALOG_PATH = os.path.join(DATA_DIR, "sources_catalog.json")
SOURCE_YEARS = os.path.join(DATA_DIR, "source_years.json")

st.set_page_config(page_title="Wisdom Extractor — Unified v19", layout="wide")
st.title("Wisdom Extractor — Unified v19")
st.caption("Collect → Clean → Canonicalize → Cluster (with human constraints) → Score → Interpret → Validate")

# ---------- data source: live-first ----------
import subprocess, time as _time
import core.persistence as _pers

SNAP = os.path.join(DATA_DIR, "live_snapshot.db")
WORKSPACE_DIR = os.path.join(DATA_DIR, "workspaces")
os.makedirs(WORKSPACE_DIR, exist_ok=True)
_PULL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "pull_live_db.sh")


def _refresh_snapshot(max_age_min=10):
    """Pull the live server DB unless the local snapshot is already fresh."""
    age = (_time.time() - os.path.getmtime(SNAP)) / 60 if os.path.exists(SNAP) else 1e9
    if age < max_age_min:
        return "fresh"
    try:
        subprocess.run([_PULL, SNAP], capture_output=True, timeout=180, check=True)
        return "pulled"
    except Exception:
        return "offline" if os.path.exists(SNAP) else "none"


st.sidebar.subheader("Data source")
_workspaces = sorted(f for f in os.listdir(WORKSPACE_DIR) if f.endswith(".db"))
_choices = (["Live corpus (auto-synced from server)"] + _workspaces
            + ["Legacy local wisdom.db"])
_want = st.session_state.pop("_select_db", None) or st.session_state.get("db_choice")
_idx = _choices.index(_want) if _want in _choices else 0
_src = st.sidebar.selectbox("Database", _choices, index=_idx,
                            help="Workspaces are separate databases for new collections; "
                                 "merge into the live corpus with scripts/merge_workspace.sh")
if _src == "Live corpus (auto-synced from server)":
    if "snap_status" not in st.session_state:
        with st.spinner("Syncing live data from the server…"):
            st.session_state["snap_status"] = _refresh_snapshot()
    if st.sidebar.button("🔄 Refresh from server now"):
        with st.spinner("Pulling fresh snapshot…"):
            st.session_state["snap_status"] = _refresh_snapshot(0)
        st.cache_data.clear()
        st.session_state["db_version"] = st.session_state.get("db_version", 0) + 1
    if os.path.exists(SNAP):
        _pers.DB_PATH = SNAP
        _age = int((_time.time() - os.path.getmtime(SNAP)) / 60)
        st.sidebar.caption(f"Live snapshot, {_age} min old"
                           + (" (server unreachable — using last copy)"
                              if st.session_state["snap_status"] == "offline" else ""))
    else:
        st.sidebar.error("No snapshot and server unreachable — using legacy local DB.")
elif _src == "Legacy local wisdom.db":
    st.sidebar.caption(f"Local file: {_pers.DB_PATH}")
else:
    _pers.DB_PATH = os.path.join(WORKSPACE_DIR, _src)
    st.sidebar.caption("Workspace database — fully separate from the open corpus.")

_new_ws = st.sidebar.text_input("New workspace name", placeholder="e.g. waterloo_pilot")
if st.sidebar.button("➕ Create workspace") and _new_ws.strip():
    _fname = _new_ws.strip().replace(" ", "_") + ".db"
    _wsp = os.path.join(WORKSPACE_DIR, _fname)
    if not os.path.exists(_wsp):
        _old = _pers.DB_PATH
        _pers.DB_PATH = _wsp
        init_db()
        _pers.DB_PATH = _old
    st.session_state["_select_db"] = _fname   # auto-open the new workspace
    st.rerun()

if st.session_state.get("db_choice") != _src:
    st.session_state["db_choice"] = _src
    st.cache_data.clear()
    st.session_state["db_version"] = st.session_state.get("db_version", 0) + 1

init_db()

# active-source banner (top of main area) — always shows which database you're in
if _src not in ("Live corpus (auto-synced from server)", "Legacy local wisdom.db"):
    _ws_s = stats()
    st.warning(f"📁 **Workspace: {_src[:-3]}** — a private database, fully separate from the "
               f"open corpus. It holds **{_ws_s['proverbs']:,}** proverbs. Add data with the "
               f"**1) Sources & Scrape** or **2) Import & Seed** tabs — everything you do stays "
               f"in this workspace. Merge into the live corpus later with "
               f"`scripts/merge_workspace.sh`.")
elif _src == "Live corpus (auto-synced from server)":
    st.caption("📡 Live corpus — read-only snapshot synced from the server.")

if "db_version" not in st.session_state:
    st.session_state["db_version"] = 0
if "pending_ops" not in st.session_state:
    st.session_state["pending_ops"] = []
if "excluded_pending" not in st.session_state:
    st.session_state["excluded_pending"] = set()


def bump():
    st.session_state["db_version"] += 1


def seed_sources_if_empty():
    if not list_sources() and os.path.exists(CATALOG_PATH):
        catalog = json.load(open(CATALOG_PATH, encoding="utf-8"))
        for s in catalog.get("sources", []):
            upsert_source(s.get("name", s.get("url", "(no name)")), s["url"], ",".join(s.get("tags", [])))


seed_sources_if_empty()


PROVERB_FIELDS = ["id", "text", "people", "language", "family", "region", "original",
                  "claim", "gloss", "quality_score", "cluster_id", "first_seen", "last_seen",
                  "url", "excluded", "sensitive"]


@st.cache_data(show_spinner=False)
def cached_proverbs(db_version: int):
    # explicit columns so an empty DB still yields a well-formed frame for every tab
    return pd.DataFrame(list_proverbs(excluded=False), columns=PROVERB_FIELDS)


st.sidebar.subheader("Annotator")
user = st.sidebar.text_input("Your name (for leaderboard)", value="(anon)")
write_mode = st.sidebar.radio("Write mode", ["Batch (faster)", "Instant"], index=0)
autosave_after = st.sidebar.number_input("Auto-save after N actions (batch)", 1, 200, 20)
st.sidebar.markdown("---")
st.sidebar.json(stats())

tabs = st.tabs(["1) Sources & Scrape", "2) Import & Seed", "3) Clean & Canonicalize",
                "4) Cluster", "5) Results & Map", "6) Annotate • Play",
                "7) Diagnostics", "8) Interpretation", "9) Export", "🛡 Admin"])

# --------------------------------------------------------------- 1) Sources & Scrape
with tabs[0]:
    st.header("Sources & Scrape")
    srcs = list_sources()
    st.caption(f"Sources in DB: {len(srcs)}")
    st.dataframe(pd.DataFrame(srcs), height=220)
    respect_robots = st.checkbox("Respect robots.txt (recommended)", value=True)
    workers = st.slider("Concurrent fetch workers", 1, 16, 8)
    tag_filter = st.text_input("Filter sources by tag substring", "")
    filtered = [s for s in srcs if tag_filter.strip().lower() in (s["tags"] or "").lower()] if tag_filter else srcs
    ids = [s["id"] for s in filtered]
    labels = {s["id"]: f"{s['name']} ({s['url']})" for s in filtered}
    pick = st.multiselect("Pick sources (empty = all filtered)", options=ids, format_func=lambda i: labels.get(i, str(i)))
    c1, c2 = st.columns(2)
    if c2.button("🛑 Stop after current source"):
        st.session_state["stop_crawl"] = True
        st.info("Stop requested.")
    if c1.button("🚀 Crawl now (depth-1)"):
        st.session_state["stop_crawl"] = False
        targets = [s for s in filtered if not pick or s["id"] in pick]
        total_new, total_skip = 0, 0
        progress = st.progress(0.0)
        for i, s in enumerate(targets):
            st.write(f"**Crawling:** {s['name']} — {s['url']}")
            try:
                pages, items = crawl_source(s["url"], respect_robots=respect_robots, workers=workers)
                st.caption(f"{len(pages)} pages, {len(items)} raw items")
                for it in items:
                    year = extract_attestation_year(it["text"])
                    it["text"] = strip_citations(it["text"])
                    if not keep(it["text"]):
                        total_skip += 1
                        continue
                    people = infer_people_from_url(it["url"], s["name"])
                    if insert_proverb(s["id"], it["text"], it["url"], people=people, first_seen=year):
                        total_new += 1
            except Exception as e:
                st.warning(f"Failed {s['url']}: {e}")
            progress.progress((i + 1) / max(1, len(targets)))
            if st.session_state.get("stop_crawl"):
                st.info("Stopped as requested."); break
        enrich_family_region(METADATA_CSV)
        st.success(f"Done. New: {total_new}, filtered as noise: {total_skip}")
        bump()

# --------------------------------------------------------------- 2) Import & Seed
with tabs[1]:
    st.header("Import & Seed")
    st.subheader("One-click seed: paper dataset (21,378 proverbs, 77 peoples)")
    if os.path.exists(SEED_CSV) and st.button("🌱 Seed from paper dataset (data/seed_proverbs.csv)"):
        seed = pd.read_csv(SEED_CSV)
        sid = upsert_source("Paper dataset (v10 proverbs_clean)", "file://seed_proverbs.csv", "seed,paper")
        rows, skipped = [], 0
        for _, r in seed.iterrows():
            text = strip_citations(str(r.get("basis") or r.get("saying") or ""))
            if not text or not keep(text):
                skipped += 1
                continue
            rows.append({"source_id": sid, "text": text, "people": r.get("people"),
                         "original": r.get("original"), "url": r.get("source_url"),
                         "first_seen": extract_attestation_year(str(r.get("original") or "")) or
                                       extract_attestation_year(str(r.get("basis") or r.get("saying") or "")),
                         "quality_score": quality_score(text)})
        n = bulk_insert_proverbs(rows)
        enrich_family_region(METADATA_CSV)
        st.success(f"Seeded {n} new proverbs ({skipped} filtered as noise, duplicates skipped).")
        bump()

    st.subheader("Import your own CSV")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        cols = df.columns.tolist()

        def guess(cands, default=0):
            for c in cols:
                if any(k in c.lower() for k in cands):
                    return cols.index(c)
            return default

        text_col = st.selectbox("Text column", cols, index=guess(["basis", "text", "claim", "saying", "proverb"]))
        people_col = st.selectbox("People / culture column", ["<none>"] + cols,
                                  index=(cols.index("people") + 1) if "people" in cols else 0)
        orig_col = st.selectbox("Original (mother tongue) column", ["<none>"] + cols)
        if st.button("Import into DB"):
            sid = upsert_source("CSV Import", f"file://{up.name}", "csv")
            rows, skipped = [], 0
            for _, r in df.iterrows():
                text = strip_citations(str(r.get(text_col, "")))
                if not text or not keep(text):
                    skipped += 1
                    continue
                rows.append({
                    "source_id": sid, "text": text,
                    "people": None if people_col == "<none>" else r.get(people_col),
                    "original": None if orig_col == "<none>" else r.get(orig_col),
                    "url": f"file://{up.name}", "quality_score": quality_score(text),
                })
            n = bulk_insert_proverbs(rows)
            enrich_family_region(METADATA_CSV)
            st.success(f"Imported {n} rows ({skipped} filtered as noise).")
            bump()

    st.subheader("Repair culture labels on existing rows")
    if st.button("🔧 Backfill `people` from page URLs + enrich family/region"):
        n1 = backfill_people_from_urls()
        n2 = enrich_family_region(METADATA_CSV)
        st.success(f"Backfilled people for {n1} rows; enriched family/region for {n2} rows.")
        bump()

    st.subheader("Extract attestation years (historical timeline)")
    st.caption("Harvests 'attested no later than' years from citation tails (e.g. '(1875)') and "
               "from dated sources (1857/1867 compilations, KJV 1611). Edit data/source_years.json to add sources.")
    if st.button("🕰️ Backfill attestation years"):
        n_cit, n_src = backfill_attestation_years(SOURCE_YEARS)
        st.success(f"Dated {n_cit} rows from citation years and {n_src} rows from dated sources.")
        bump()

# --------------------------------------------------------------- 3) Clean & Canonicalize
with tabs[2]:
    st.header("Clean & Canonicalize")
    df = cached_proverbs(st.session_state["db_version"])
    st.caption(f"Active proverbs: {len(df)}")
    c1, c2 = st.columns(2)
    if c1.button("🧹 Run noise filter on existing rows (marks boilerplate as excluded)") and not df.empty:
        bad = [int(r["id"]) for _, r in df.iterrows() if not keep(r["text"])]
        bulk_mark_excluded(bad, True)
        st.success(f"Excluded {len(bad)} noise rows (reversible in DB).")
        bump()
    if c2.button("⚗️ Canonicalize all (compute claims + quality)") and not df.empty:
        updates = [(int(r["id"]), canonicalize(r["text"]), quality_score(r["text"])) for _, r in df.iterrows()]
        save_claims(updates)
        st.success(f"Canonicalized {len(updates)} proverbs.")
        bump()
    if not df.empty:
        show = df[["id", "text", "claim", "people", "quality_score"]].head(50)
        st.dataframe(show, height=350)

# --------------------------------------------------------------- 4) Cluster
with tabs[3]:
    st.header("Cluster (char 3–5-gram TF-IDF)")
    df = cached_proverbs(st.session_state["db_version"])
    if df.empty:
        st.info("No data. Scrape, seed, or import first.")
    else:
        tau = st.slider("Distance threshold τ (paper default 0.25–0.35)", 0.15, 0.60, 0.35, 0.01)
        use_constraints = st.checkbox("Apply human must/cannot-link constraints", value=True)
        min_conf = st.slider("Minimum consensus confidence to enforce a constraint",
                             0.50, 1.00, 0.60, 0.05,
                             help="Pairs are aggregated across annotators (reliability-weighted majority). "
                                  "Disputed or low-confidence pairs are excluded and queued for re-annotation.")
        agglo_limit = st.number_input("Agglomerative limit (larger datasets use sparse graph clustering)",
                                      500, 20000, 4000, 500)
        basis_col = "claim" if df["claim"].notna().any() else "text"
        if basis_col == "text":
            st.warning("No canonical claims yet — clustering raw text. Run tab 3 first for better alignment.")
        if st.button("🔗 Cluster now"):
            work = df.dropna(subset=[basis_col])
            texts = work[basis_col].astype(str).tolist()
            ids = work["id"].astype(int).tolist()
            must, cannot = [], []
            if use_constraints:
                agg_pairs, _ = aggregate_constraints(list_constraints())
                must, cannot = constraint_pairs_for_clustering(agg_pairs, min_confidence=min_conf)
                n_disputed = sum(1 for p in agg_pairs if p["disputed"])
                st.caption(f"Consensus constraints: {len(must)} must, {len(cannot)} cannot "
                           f"({n_disputed} disputed pairs excluded).")
            with st.spinner(f"Clustering {len(texts)} items..."):
                labels, method = cluster_texts(texts, ids, tau=tau, must_pairs=must,
                                               cannot_pairs=cannot, agglo_limit=int(agglo_limit))
            save_clusters(list(zip(ids, labels)))
            st.success(f"Done via {method}: {len(set(labels.tolist()))} clusters for {len(texts)} items. Saved to DB.")
            bump()

        dfc = cached_proverbs(st.session_state["db_version"])
        if dfc["cluster_id"].notna().any():
            summary = summarize_clusters(dfc)
            st.session_state["cluster_summary"] = summary
            st.subheader(f"Top clusters by wisdom score (S = coverage + 0.3·support) — {len(summary)} clusters")
            st.dataframe(summary[["cluster_id", "claim", "coverage", "support", "wisdom_score", "cultures"]].head(50),
                         height=400)

# --------------------------------------------------------------- 5) Results & Map
with tabs[4]:
    st.header("Results & Semantic Map")
    summary = st.session_state.get("cluster_summary")
    df = cached_proverbs(st.session_state["db_version"])
    if summary is None or summary.empty:
        st.info("Run clustering first (tab 4).")
    else:
        st.subheader("Cluster browser")
        top_ids = summary["cluster_id"].head(200).tolist()
        cid = st.selectbox("Cluster", top_ids,
                           format_func=lambda c: f"#{c} — {summary[summary.cluster_id == c]['claim'].iloc[0][:70]}")
        row = summary[summary["cluster_id"] == cid].iloc[0]
        st.markdown(f"**Canonical claim:** {row['claim']}")
        st.markdown(f"**Coverage:** {row['coverage']} cultures • **Support:** {row['support']} • "
                    f"**Score:** {row['wisdom_score']}")
        members = df[df["cluster_id"] == cid][["id", "people", "text", "original", "url", "first_seen"]]
        st.dataframe(members, height=260)

        dated = df[(df["cluster_id"] == cid) & df["first_seen"].notna()]
        if len(dated) >= 2:
            st.subheader("Attestation timeline (attested no later than)")
            tl = dated[["first_seen", "people"]].rename(columns={"first_seen": "year"}).copy()
            tl["year"] = tl["year"].astype(int)
            st.scatter_chart(tl, x="year", y="people")
            span = int(tl["year"].max()) - int(tl["year"].min())
            st.caption(f"{len(tl)} dated attestations spanning {span} years "
                       f"({int(tl['year'].min())}–{int(tl['year'].max())}). "
                       "Years are upper bounds from cited sources, not origins; undated rows are omitted.")
        elif df["first_seen"].notna().sum() == 0:
            st.caption("No attestation years yet — run '🕰️ Backfill attestation years' in tab 2.")

        st.subheader("🌍 World map — motifs, network, and time travel")
        if st.button("Render world map"):
            with st.spinner("Building map from database..."):
                st.session_state["map_html"] = build_map_html(df)
        if "map_html" in st.session_state:
            st.components.v1.html(st.session_state["map_html"], height=820, scrolling=True)

        st.subheader("2D semantic map (top clusters)")
        n_map = st.slider("Clusters to map", 50, 1000, 300, 50)
        if st.button("🗺️ Compute map"):
            top = summary.head(n_map)
            coords = compute_coords(top["claim"].astype(str).tolist())
            plot_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1],
                                    "coverage": top["coverage"].values,
                                    "claim": top["claim"].str.slice(0, 60).values})
            st.session_state["map_df"] = plot_df
        if "map_df" in st.session_state:
            st.scatter_chart(st.session_state["map_df"], x="x", y="y", size="coverage", color="coverage")
            st.dataframe(st.session_state["map_df"][["claim", "coverage"]].head(30), height=200)

# --------------------------------------------------------------- 6) Annotate • Play
with tabs[5]:
    st.header("Annotate — Play Mode")
    df = cached_proverbs(st.session_state["db_version"])
    if st.session_state["excluded_pending"]:
        df = df[~df["id"].isin(st.session_state["excluded_pending"])]
    if df.empty:
        st.info("No active proverbs.")
    else:
        hi = st.slider("Hi threshold", 0.50, 0.95, 0.85, 0.01)
        lo = st.slider("Lo threshold", 0.05, 0.60, 0.35, 0.01)
        k = st.slider("Neighbors per anchor (k)", 2, 20, 8)
        max_pool = st.slider("Pair pool sample size (speed)", 500, 8000, 3000, 500)

        if "pairs" not in st.session_state or st.button("🔄 Refresh pairs"):
            pool = df.sample(min(max_pool, len(df)), random_state=None)
            with st.spinner("Computing candidate pairs (sparse kNN)..."):
                pos, neg = nearest_pairs(pool["text"].astype(str).tolist(),
                                         pool["id"].astype(int).tolist(), k=k, hi=hi, lo=lo)
            st.session_state["pairs"] = {"pos": pos, "neg": neg}
            st.caption(f"{len(pos)} uncertain pairs, {len(neg)} negative pairs")

        pairs = st.session_state.get("pairs", {"pos": [], "neg": []})
        strategy = st.radio("Pair strategy",
                            ["Likely same idea", "Likely different idea",
                             "Verify disputed / low-confidence", "Surprise me"], horizontal=True)

        def pick_pair():
            if strategy == "Verify disputed / low-confidence":
                agg_pairs, _ = aggregate_constraints(list_constraints())
                known = set(df["id"].astype(int))
                review = [p for p in pairs_needing_review(agg_pairs)
                          if p["a_id"] in known and p["b_id"] in known]
                if review:
                    p = review[0] if random.random() < 0.5 else random.choice(review[:10])
                    return (p["a_id"], p["b_id"], p["confidence"])
                st.toast("No disputed pairs — all consensus is solid. Serving a fresh pair.", icon="✅")
            if strategy == "Likely same idea" and pairs["pos"]:
                return random.choice(pairs["pos"])
            if strategy == "Likely different idea" and pairs["neg"]:
                return random.choice(pairs["neg"])
            a, b = random.sample(df["id"].tolist(), 2)
            return (int(a), int(b), 0.0)

        if "pair" not in st.session_state:
            st.session_state["pair"] = pick_pair()
        a, b, s = st.session_state["pair"]
        ra = df[df["id"] == a].iloc[0] if (df["id"] == a).any() else df.sample(1).iloc[0]
        rb = df[df["id"] == b].iloc[0] if (df["id"] == b).any() else df.sample(1).iloc[0]

        def drop_from_pairs(pid):
            for key in ("pos", "neg"):
                st.session_state["pairs"][key] = [p for p in st.session_state["pairs"][key]
                                                  if p[0] != pid and p[1] != pid]

        def exclude(pid):
            if write_mode.startswith("Instant"):
                mark_excluded(pid, True); bump()
            else:
                st.session_state["pending_ops"].append({"op": "exclude", "pid": pid})
                st.session_state["excluded_pending"].add(pid)
            drop_from_pairs(pid)
            st.session_state["pair"] = pick_pair()

        def constrain_score(v):
            # graded judgments write immediately (batch queue predates the graded scheme)
            add_constraint(int(ra["id"]), int(rb["id"]), None, annotator_uid(user), score=v)
            bump()
            st.session_state["pair"] = pick_pair()

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Proverb A")
            st.write(ra["text"])
            st.caption(f"ID {int(ra['id'])} • {ra.get('people') or 'culture unknown'}")
            if st.button("❌ Not a saying (A)"):
                exclude(int(ra["id"])); st.toast("Excluded A", icon="❌")
        with c2:
            st.subheader("Proverb B")
            st.write(rb["text"])
            st.caption(f"ID {int(rb['id'])} • {rb.get('people') or 'culture unknown'}")
            if st.button("❌ Not a saying (B)"):
                exclude(int(rb["id"])); st.toast("Excluded B", icon="❌")

        st.caption("Graded scheme (Pelican): judge the lesson, not the imagery; pick the highest level that applies.")
        d1, d2, d3 = st.columns(3)
        if d1.button("🎯 Same rule (4)"):
            constrain_score(4); st.toast("Saved: same rule", icon="🎯")
        if d2.button("🤝 Same advice (3)"):
            constrain_score(3); st.toast("Saved: same advice", icon="🤝")
        if d3.button("🧩 Same theme (2)"):
            constrain_score(2); st.toast("Saved: same theme", icon="🧩")
        d4, d5, d6 = st.columns(3)
        if d4.button("🔗 Related, diff. lesson (1)"):
            constrain_score(1); st.toast("Saved: related", icon="🔗")
        if d5.button("➖ Unrelated (0)"):
            constrain_score(0); st.toast("Saved: unrelated", icon="➖")
        if d6.button("⚔️ Contradictory (−1)"):
            constrain_score(-1); st.toast("Saved: contradiction", icon="⚔️")
        if st.button("⏭️ Skip"):
            st.session_state["pair"] = pick_pair()

        if write_mode.startswith("Batch") and len(st.session_state["pending_ops"]) >= autosave_after:
            bulk_apply(st.session_state["pending_ops"])
            st.session_state["pending_ops"].clear()
            st.session_state["excluded_pending"].clear()
            bump(); st.toast("Auto-saved annotations", icon="💾")

        st.markdown("---")
        cA, cB = st.columns(2)
        if write_mode.startswith("Batch") and cA.button("💾 Save pending now"):
            bulk_apply(st.session_state["pending_ops"])
            st.session_state["pending_ops"].clear()
            st.session_state["excluded_pending"].clear()
            bump(); st.success("Saved.")
        cB.caption(f"Pending ops: {len(st.session_state['pending_ops'])}")
        st.info("Your must/cannot links feed directly into clustering (tab 4) "
                "and are scored as an evaluation set in Diagnostics (tab 7).")

# --------------------------------------------------------------- 7) Diagnostics
with tabs[6]:
    st.header("Diagnostics & Validation")
    df = cached_proverbs(st.session_state["db_version"])
    clustered = df.dropna(subset=["cluster_id"])
    if clustered.empty:
        st.info("Run clustering first.")
    else:
        summary = st.session_state.get("cluster_summary", summarize_clusters(df))

        st.subheader("Coverage distribution (operational bins)")
        st.json(coverage_distribution(summary))

        st.subheader("Human constraint agreement (annotation game as evaluation set)")
        labels_by_id = dict(zip(clustered["id"].astype(int), clustered["cluster_id"].astype(int)))
        agr = constraint_agreement(labels_by_id, list_constraints())
        st.json({k: v for k, v in agr.items() if not k.endswith("violations")})
        if agr["must_violations"] or agr["cannot_violations"]:
            st.caption(f"Violations — must: {agr['must_violations'][:10]} cannot: {agr['cannot_violations'][:10]}")

        st.subheader("Annotation consensus & annotator reliability")
        from core.annotation_quality import krippendorff_alpha_ordinal
        alpha, n_units = krippendorff_alpha_ordinal(list_constraints())
        if alpha is not None:
            st.metric("Krippendorff's α (ordinal, multi-annotated pairs)", f"{alpha:.3f}",
                      help=f"Computed over {n_units} pairs with ≥2 graded annotations.")
        else:
            st.caption(f"Krippendorff's α: needs ≥2 pairs with multiple annotations (have {n_units}).")
        agg_pairs, annotators = aggregate_constraints(list_constraints())
        if agg_pairs:
            n_disputed = sum(1 for p in agg_pairs if p["disputed"])
            multi = sum(1 for p in agg_pairs if p["n"] > 1)
            st.json({"annotated_pairs": len(agg_pairs), "with_multiple_votes": multi,
                     "disputed": n_disputed,
                     "mean_confidence": round(sum(p["confidence"] for p in agg_pairs) / len(agg_pairs), 3)})
            st.dataframe(pd.DataFrame([{"annotator": u, **v} for u, v in annotators.items()]))
            review = pairs_needing_review(agg_pairs)
            if review:
                st.caption(f"{len(review)} pairs need more votes — use "
                           f"'Verify disputed / low-confidence' in the Annotate tab.")
                st.dataframe(pd.DataFrame(review).head(20), height=220)
        else:
            st.caption("No annotations yet — play the Annotate tab first.")

        basis_col = "claim" if df["claim"].notna().any() else "text"
        work = clustered.dropna(subset=[basis_col])

        c1, c2 = st.columns(2)
        with c1:
            if st.button("📐 Silhouette (cosine, sampled)"):
                sil = silhouette_cosine(work[basis_col].astype(str).tolist(),
                                        work["cluster_id"].astype(int).to_numpy())
                st.metric("Silhouette", f"{sil:.3f}" if sil is not None else "n/a")
        with c2:
            if st.button("🎲 Bootstrap stability (mean ARI, 80% subsamples)"):
                sample = work.sample(min(3000, len(work)), random_state=0)
                ari = bootstrap_stability(sample[basis_col].astype(str).tolist(),
                                          sample["id"].astype(int).tolist(),
                                          tau=0.35, iterations=3)
                st.metric("Mean ARI", f"{ari:.4f}" if ari is not None else "n/a")

        st.subheader("τ sensitivity sweep (paper's Table 2 protocol)")
        if st.button("🔬 Run sweep (sampled, a few minutes)"):
            with st.spinner("Sweeping τ ∈ {0.25..0.45}..."):
                sweep = sensitivity_sweep(work[basis_col].astype(str).tolist(),
                                          work["id"].astype(int).tolist())
            st.dataframe(sweep)
            st.session_state["sweep"] = sweep

        st.subheader("Triangulation vs random mixing (families/regions)")
        if df["family"].notna().any():
            if st.button("🧭 Run permutation test (200 perms)"):
                tri = permutation_triangulation(clustered)
                st.dataframe(tri, height=300)
        else:
            st.caption("Needs family/region metadata — run the backfill in tab 2.")

# --------------------------------------------------------------- 8) Interpretation
with tabs[7]:
    st.header("Interpretation (deterministic, offline)")
    summary = st.session_state.get("cluster_summary")
    df = cached_proverbs(st.session_state["db_version"])
    if summary is None or summary.empty:
        st.info("Run clustering first.")
    else:
        st.subheader("Themed report")
        for theme, sub in themed_report(summary).items():
            with st.expander(f"{theme} ({len(sub)} top clusters)"):
                st.dataframe(sub, height=200)

        st.subheader("Productive tensions (contradictory high-coverage wisdom)")
        tensions = productive_tensions(summary)
        if len(tensions):
            st.dataframe(tensions)
        else:
            st.caption("No opposing high-coverage pairs found at current thresholds.")

        st.subheader("Ecological correlations (suggestive, not causal)")
        if os.path.exists(METADATA_CSV):
            corr = ecological_correlations(df, METADATA_CSV)
            if len(corr):
                st.dataframe(corr.head(30), height=300)
                st.caption("Spearman ρ across peoples between theme share and proxy. "
                           "Coarse proxies; treat as hypotheses (paper §6.3).")
            else:
                st.caption("Not enough peoples matched to metadata for correlations.")

# --------------------------------------------------------------- 9) Export
with tabs[8]:
    st.header("Export")
    df = cached_proverbs(st.session_state["db_version"])
    st.subheader("Leaderboard")
    st.caption("Nicknames are display-only and hidden by default, so this tab is safe to "
               "show on a shared screen. Exports below never contain them: judgments carry "
               "the pseudonym uid only.")
    _lb_names = st.checkbox("Show nicknames", value=False, key="_lb_names")
    _lb = pd.DataFrame(leaderboard())
    if not _lb.empty and not _lb_names:
        _first = _lb.columns[0]
        _lb = _lb.copy()
        _lb[_first] = [f"annotator {i+1}" for i in range(len(_lb))]
    st.dataframe(_lb, use_container_width=True, hide_index=True)
    st.subheader("Downloads")
    if not df.empty:
        st.download_button("proverbs.csv", df.to_csv(index=False), "proverbs.csv", "text/csv")
    summary = st.session_state.get("cluster_summary")
    if summary is not None and len(summary):
        st.download_button("clusters.csv",
                           summary.drop(columns=["examples"]).to_csv(index=False),
                           "clusters.csv", "text/csv")
        st.download_button("wisdom_clusters.json",
                           json.dumps(summary.to_dict("records"), ensure_ascii=False, indent=2),
                           "wisdom_clusters.json", "application/json")
    st.download_button("annotation_export.json",
                       json.dumps(export_annotations(), ensure_ascii=False, indent=2),
                       "annotation_export.json", "application/json")
    db_path = _pers.DB_PATH
    if os.path.exists(db_path):
        st.caption("⚠️ The raw database contains the nickname↔uid table. Share the CSV/JSON "
                   "exports instead — those are already pseudonymised.")
        with open(db_path, "rb") as f:
            st.download_button("wisdom.db (internal use)", f.read(), "wisdom.db",
                               "application/octet-stream")


# ---------------------------------------------------------------- 10) Admin / Status
with tabs[9]:
    from core.persistence import (nickname_of, sensitive_reports_by_user, unflag_by_user,
                                  block_annotator, unblock_annotator, list_annotators_admin,
                                  purge_annotator)
    from core.science import (alpha_with_ci, auc_with_ci, annotator_profile,
                              overlap_stats, readiness)

    st.subheader("Research status")
    st.caption(f"Database in use: `{_pers.DB_PATH}`")

    _cons = list_constraints()
    _agg, _annot_raw = aggregate_constraints(_cons)
    _must = sum(1 for x in _agg if x["label"] == "must")
    _cannot = sum(1 for x in _agg if x["label"] == "cannot")
    _st = stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active proverbs", f"{_st['proverbs']:,}")
    c2.metric("Peoples", _st["peoples"])
    c3.metric("Raw judgments", len(_cons))
    c4.metric("Consensus pairs", len(_agg))

    if not _cons:
        st.info("No judgments yet — statistics appear once annotation starts.")
    else:
        with st.spinner("Computing statistics with confidence intervals…"):
            _al = alpha_with_ci(_cons)
            _ov = overlap_stats(_cons)
            _an = annotator_profile(_cons)
            _auc = st.session_state.get("_auc_cache", {"auc": None, "verdict": "not run"})

        # ---------- readiness ----------
        _rd = readiness(_al, _auc, _ov, _an, len(_agg))
        st.markdown("### Publication readiness — " + _rd["score"])
        st.info(_rd["overall"])
        _icon = {"green": "🟢", "amber": "🟡", "red": "🔴"}
        for ch in _rd["checks"]:
            with st.expander(f"{_icon[ch['status']]}  {ch['check']} — {ch['value']}"):
                st.write(f"**Target:** {ch['target']}")
                st.write(ch["action"])

        st.markdown("---")
        st.markdown("### The numbers, and what they mean")

        # ---------- alpha ----------
        m1, m2 = st.columns([1, 2])
        m1.metric("Krippendorff's α (ordinal)",
                  f"{_al['alpha']:.3f}" if _al["alpha"] is not None else "n/a",
                  help="Agreement among annotators, corrected for chance, on the graded scale.")
        if _al.get("lo") is not None:
            m1.caption(f"95% CI: {_al['lo']} to {_al['hi']}  ·  {_al['n_units']} double-rated pairs")
        m2.write(_al["meaning"])
        if _al.get("needed_units"):
            m2.warning(f"To make the lower bound clear 0.80, you need roughly "
                       f"**{_al['needed_units']} double-rated pairs** in total "
                       f"({max(0, _al['needed_units'] - _al['n_units'])} more), assuming "
                       f"agreement holds at its current level.")

        # ---------- separation (on demand: it vectorises the whole corpus) ----------
        st.markdown("**Does the method match human judgment?**")
        if st.button("Compute AUC with confidence interval (~1 min)"):
            with st.spinner("Vectorising the corpus and scoring judged pairs…"):
                from sklearn.metrics.pairwise import cosine_similarity as _cos
                from core.clustering import vectorize as _vec
                _rows2 = {r["id"]: r for r in list_proverbs(with_claims_only=True)}
                _ids = list(_rows2)
                _X, _ = _vec([str(_rows2[i]["claim"]) for i in _ids])
                _pos = {pid: k for k, pid in enumerate(_ids)}
                _use = [(x["a_id"], x["b_id"], 1 if x["label"] == "must" else 0)
                        for x in _agg if x["label"] in ("must", "cannot")
                        and x["a_id"] in _pos and x["b_id"] in _pos]
                _sims = [float(_cos(_X[_pos[a]], _X[_pos[b]])[0, 0]) for a, b, _l in _use]
                _labs = [l for _a, _b, l in _use]
                st.session_state["_auc_cache"] = auc_with_ci(_sims, _labs)
            st.rerun()
        if _auc.get("auc") is not None:
            a1, a2 = st.columns([1, 2])
            a1.metric("AUC", f"{_auc['auc']:.3f}")
            a1.caption(f"95% CI: {_auc['lo']} to {_auc['hi']}  ·  "
                       f"{_auc['n_pos']} same / {_auc['n_neg']} different")
            a2.write(_auc["meaning"])
            if _auc.get("needed_pairs"):
                a2.warning(f"For the lower bound to clear 0.80, roughly "
                           f"**{_auc['needed_pairs']} judged pairs** would be needed.")
        else:
            st.caption("Not computed in this session — press the button above.")

        # ---------- overlap + independence ----------
        o1, o2 = st.columns(2)
        with o1:
            st.metric("Double-rated pairs", _ov["multi_rated"],
                      help="Pairs judged independently by two or more people. "
                           "Every agreement statistic rests on these alone.")
            st.caption(f"{_ov['overlap_rate']*100:.0f}% of {_ov['n_pairs']} judged pairs")
            st.write(_ov["meaning"])
        with o2:
            st.metric("Active annotators", _an["n_annotators"],
                      help="Independence matters: if one person supplies most judgments, "
                           "consensus mostly reflects that person.")
            st.caption(f"largest single share: {_an['top_share']*100:.0f}%")
            st.write(_an["meaning"])
        if _an["outliers"]:
            st.warning("Annotators well out of step with consensus (check before trusting "
                       "their data): " + ", ".join(f"{o['uid']} (reliability {o['reliability']})"
                                                   for o in _an["outliers"]))

        st.markdown("**Per-annotator detail** — uid is the pseudonym used in all exports.")
        _show_names = st.checkbox("Show nicknames (uncheck before screen-sharing)", value=False)
        st.dataframe(pd.DataFrame([
            {**({"nickname": nickname_of(r["uid"]) or "—"} if _show_names else {}),
             "uid": r["uid"], "judgments": r["judgments"],
             "share": f"{r['share']*100:.0f}%", "reliability": r["reliability"]}
            for r in _an["rows"]]), use_container_width=True, hide_index=True)

    # ---------- corpus release readiness ----------
    st.markdown("---")
    st.markdown("### Corpus archiving (pre-registered criteria)")
    st.caption("Versions are triggered by data volume, never by what the statistics show — "
               "publishing only on favourable numbers would bias the version history. "
               "See CORPUS_RELEASE_CRITERIA.md. Publication itself stays manual.")
    try:
        from scripts.corpus_release_check import current_state as _cs, last_release as _lr, evaluate as _ev
        _cur_s = _cs(); _last_s = _lr()
        _checks, _due = _ev(_cur_s, _last_s)
        if _last_s:
            st.caption(f"Last published: **{_last_s.get('version','?')}** "
                       f"({_last_s.get('generated','?')}) — {_last_s.get('proverbs',0):,} proverbs, "
                       f"{_last_s.get('judgments',0):,} judgments")
        st.dataframe(pd.DataFrame([
            {"trigger": c["trigger"], "met": "✅" if c["met"] else "—", "change": c["detail"]}
            for c in _checks]), use_container_width=True, hide_index=True)
        if _due:
            st.success("A new corpus version is due. Prepare the export with "
                       "`python scripts/corpus_release_check.py --prepare`, check the files, "
                       "then publish and run `--record-release`.")
        else:
            st.info("No new version due yet.")
    except Exception as _e:
        st.caption(f"Release check unavailable: {_e}")

    # ---------- moderation ----------
    st.markdown("---")
    st.markdown("### Moderation")
    st.caption("Blocking stops future submissions and leaves existing data intact. "
               "Purging deletes an account's judgments and cannot be undone.")
    _accounts = list_annotators_admin()
    if _accounts:
        st.dataframe(pd.DataFrame([
            {**({"nickname": a["nickname"] or "—"} if st.session_state.get("_mod_names") else {}),
             "uid": a["uid"], "judgments": a["judgments"],
             "status": "BLOCKED" if a["blocked"] else "active",
             "reason": a["block_reason"]} for a in _accounts]),
            use_container_width=True, hide_index=True)
        st.checkbox("Show nicknames in the moderation table", key="_mod_names")
        _pick = st.selectbox("Account", [a["uid"] for a in _accounts],
                             format_func=lambda u: f"{nickname_of(u) or u} ({u})")
        _cur = next((a for a in _accounts if a["uid"] == _pick), None)
        b1, b2, b3 = st.columns(3)
        with b1:
            _reason = st.text_input("Reason (recorded)", placeholder="e.g. random clicking")
            if st.button("🚫 Block account", disabled=bool(_cur and _cur["blocked"])):
                block_annotator(_pick, _reason)
                st.success("Blocked. They can no longer submit."); st.rerun()
        with b2:
            if st.button("✅ Unblock", disabled=not (_cur and _cur["blocked"])):
                unblock_annotator(_pick); st.success("Unblocked."); st.rerun()
        with b3:
            _confirm = st.text_input("Type DELETE to confirm purge", key="_purge_confirm")
            if st.button("🗑 Purge judgments", disabled=_confirm != "DELETE"):
                _n = purge_annotator(_pick)
                st.warning(f"Deleted {_n} judgments from {_pick}."); st.rerun()
    else:
        st.caption("No annotator accounts yet.")

    # ---------- hidden content ----------
    st.markdown("---")
    st.markdown("**Hidden proverbs (adult language)** — kept in the corpus, withheld from "
                "the game and the public map.")
    import sqlite3 as _sq
    _con = _sq.connect(_pers.DB_PATH)
    _hidden = _con.execute("SELECT COUNT(*) FROM proverbs WHERE COALESCE(sensitive,0)=1").fetchone()[0]
    _con.close()
    st.metric("Currently hidden", f"{_hidden:,}")
    _reps = sensitive_reports_by_user()
    if _reps:
        st.dataframe(pd.DataFrame([{"uid": r["user"], "hidden": r["n"]} for r in _reps]),
                     use_container_width=True, hide_index=True)
        _who = st.selectbox("Revert all hides by", [r["user"] for r in _reps])
        if st.button("↩︎ Revert this annotator's hides"):
            _n = unflag_by_user(_who)
            st.success(f"Reverted {_n} proverbs. (Word-list hides and pairs someone else "
                       f"also reported stay hidden.)")
    else:
        st.caption("No annotator reports yet — everything hidden came from the word list.")

    # ---------- canonicalization misses ----------
    st.markdown("---")
    st.markdown("**Canonicalization misses** — where human judgments disagree with the "
                "method. Each *miss* (judged the same, low similarity) is a candidate for a "
                "new rule; each *over-merge* (judged different, high similarity) flags a rule "
                "to restrain.")
    if st.button("Find canonicalization misses"):
        with st.spinner("Comparing human links against the current claims…"):
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from scripts.canon_misses import find_misses
            _misses, _over = find_misses()
        st.write(f"**{len(_misses)} misses** — judged the same idea, but claims are far apart:")
        if _misses:
            st.dataframe(pd.DataFrame(_misses), use_container_width=True, hide_index=True)
        st.write(f"**{len(_over)} over-merges** — judged different, but claims near-identical:")
        if _over:
            st.dataframe(pd.DataFrame(_over), use_container_width=True, hide_index=True)
