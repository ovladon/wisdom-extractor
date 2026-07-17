"""Wisdom Extractor — Unified (v19)

Merges the ConsILR-2025 paper pipeline (v10_working: canonicalization, char n-gram
clustering, diagnostics, interpretation) with the Wisdom Lab platform (v18: robust
scraping, SQLite persistence, human annotation game), and fixes the defects of both.
Run:  streamlit run app.py
"""
import os, json, random, warnings

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
    infer_people_from_url, backfill_attestation_years,
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

init_db()

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
                  "claim", "quality_score", "cluster_id", "first_seen", "last_seen",
                  "url", "excluded"]


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
                "7) Diagnostics", "8) Interpretation", "9) Export"])

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

        def constrain(label):
            if write_mode.startswith("Instant"):
                add_constraint(int(ra["id"]), int(rb["id"]), label, user); bump()
            else:
                st.session_state["pending_ops"].append(
                    {"op": "constraint", "a": int(ra["id"]), "b": int(rb["id"]), "label": label, "user": user})
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

        d1, d2, d3 = st.columns(3)
        if d1.button("✅ Same idea (must-link)"):
            constrain("must"); st.toast("Saved MUST link", icon="✅")
        if d2.button("🚫 Different idea (cannot-link)"):
            constrain("cannot"); st.toast("Saved CANNOT link", icon="🚫")
        if d3.button("⏭️ Skip"):
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
    st.dataframe(pd.DataFrame(leaderboard()))
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
    db_path = os.environ.get("WISDOM_DB_PATH", "wisdom.db")
    if os.path.exists(db_path):
        with open(db_path, "rb") as f:
            st.download_button("wisdom.db", f.read(), "wisdom.db", "application/octet-stream")
