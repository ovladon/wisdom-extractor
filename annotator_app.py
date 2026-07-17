"""Wisdom Extractor — Annotator Portal (v19.4)

A slim, safe entry point for annotators: the pair-judgment game, live leaderboard,
personal reliability, and the world map — nothing else. No scraping, no clustering,
no destructive admin actions. Deploy THIS for annotators; app.py stays the lab bench.

Run:  streamlit run annotator_app.py
Env:  WISDOM_DB_PATH       path to the shared database (default wisdom.db)
      ANNOTATOR_CODE       optional access code; if set, visitors must enter it once
"""
import os, random

import pandas as pd
import streamlit as st

from core.persistence import (
    init_db, list_proverbs, mark_excluded, add_constraint, list_constraints,
    stats, leaderboard,
)
from core.annotation_quality import aggregate_constraints, pairs_needing_review
from core.clustering import nearest_pairs
from core.mapview import build_map_html

st.set_page_config(page_title="Wisdom Lab — Annotate", layout="centered")
init_db()

# ---------- optional access code ----------
CODE = os.environ.get("ANNOTATOR_CODE", "")
if CODE:
    if "unlocked" not in st.session_state:
        st.session_state["unlocked"] = False
    if not st.session_state["unlocked"]:
        st.title("Wisdom Lab")
        entered = st.text_input("Access code", type="password")
        if st.button("Enter"):
            if entered == CODE:
                st.session_state["unlocked"] = True
                st.rerun()
            else:
                st.error("Wrong code — ask the study coordinator.")
        st.stop()

st.title("🧭 Wisdom Lab — help map humanity's proverbs")
st.caption("Judge pairs of sayings: do they express the same idea? "
           "Your judgments improve the clustering and are scored for research. Thank you!")

if "db_version" not in st.session_state:
    st.session_state["db_version"] = 0


@st.cache_data(show_spinner=False)
def proverbs_df(v: int):
    cols = ["id", "text", "people", "language", "family", "region", "original",
            "claim", "quality_score", "cluster_id", "first_seen", "last_seen",
            "url", "excluded"]
    return pd.DataFrame(list_proverbs(excluded=False), columns=cols)


df = proverbs_df(st.session_state["db_version"])
if df.empty:
    st.warning("The database is empty — the coordinator needs to seed it first.")
    st.stop()

name = st.text_input("Your name or nickname (for the leaderboard)", value="", max_chars=40)
if not name.strip():
    st.info("Type a name above to start annotating.")
    st.stop()
user = name.strip()

tab_play, tab_board, tab_map = st.tabs(["🎯 Annotate", "🏆 Leaderboard", "🌍 The map you're improving"])

# ---------------- Annotate ----------------
with tab_play:
    strategy = st.radio("What kind of pairs?",
                        ["Uncertain pairs (most useful)", "Disputed pairs (settle a tie)", "Random"],
                        horizontal=True)

    if "pool_pairs" not in st.session_state:
        pool = df.sample(min(2500, len(df)), random_state=None)
        with st.spinner("Preparing pairs..."):
            pos, neg = nearest_pairs(pool["text"].astype(str).tolist(),
                                     pool["id"].astype(int).tolist(), k=6, hi=0.85, lo=0.30)
        st.session_state["pool_pairs"] = pos + neg

    def pick_pair():
        if strategy.startswith("Disputed"):
            agg, _ = aggregate_constraints(list_constraints())
            known = set(df["id"].astype(int))
            review = [p for p in pairs_needing_review(agg)
                      if p["a_id"] in known and p["b_id"] in known]
            if review:
                p = random.choice(review[:15])
                return (p["a_id"], p["b_id"])
        if strategy.startswith("Uncertain") and st.session_state["pool_pairs"]:
            p = random.choice(st.session_state["pool_pairs"])
            return (p[0], p[1])
        a, b = random.sample(df["id"].tolist(), 2)
        return (int(a), int(b))

    if "pair" not in st.session_state:
        st.session_state["pair"] = pick_pair()
    a, b = st.session_state["pair"]
    ra = df[df["id"] == a].iloc[0] if (df["id"] == a).any() else df.sample(1).iloc[0]
    rb = df[df["id"] == b].iloc[0] if (df["id"] == b).any() else df.sample(1).iloc[0]

    c1, c2 = st.columns(2)
    for col, row, label in ((c1, ra, "A"), (c2, rb, "B")):
        with col:
            st.markdown(f"**Saying {label}**")
            st.markdown(f"> {row['text']}")
            st.caption(row.get("people") or "culture unknown")

    st.markdown("**Do these two sayings express the same underlying idea?**")
    b1, b2, b3 = st.columns(3)
    if b1.button("✅ Same idea", use_container_width=True):
        add_constraint(int(ra["id"]), int(rb["id"]), "must", user)
        st.session_state["pair"] = pick_pair()
        st.session_state["done"] = st.session_state.get("done", 0) + 1
        st.rerun()
    if b2.button("🚫 Different idea", use_container_width=True):
        add_constraint(int(ra["id"]), int(rb["id"]), "cannot", user)
        st.session_state["pair"] = pick_pair()
        st.session_state["done"] = st.session_state.get("done", 0) + 1
        st.rerun()
    if b3.button("⏭️ Can't tell / skip", use_container_width=True):
        st.session_state["pair"] = pick_pair()
        st.rerun()

    e1, e2 = st.columns(2)
    if e1.button("❌ A is not a real saying"):
        mark_excluded(int(ra["id"]), True)
        st.session_state["pair"] = pick_pair(); st.rerun()
    if e2.button("❌ B is not a real saying"):
        mark_excluded(int(rb["id"]), True)
        st.session_state["pair"] = pick_pair(); st.rerun()

    done = st.session_state.get("done", 0)
    if done:
        st.success(f"{done} judgments this session — every one counts. 🙌")

# ---------------- Leaderboard ----------------
with tab_board:
    st.subheader("Leaderboard")
    st.dataframe(pd.DataFrame(leaderboard()), use_container_width=True)
    agg, annotators = aggregate_constraints(list_constraints())
    me = annotators.get(user)
    if me:
        st.metric("Your consistency score", f"{me['reliability']:.2f}",
                  help="Agreement with the community consensus (0.7 is the neutral prior). "
                       "It rises as your judgments align with settled pairs.")
    s = stats()
    st.caption(f"Corpus: {s['proverbs']} sayings from {s['peoples']} peoples · "
               f"{s['must'] + s['cannot']} judgments so far")

# ---------------- Map ----------------
with tab_map:
    st.caption("Every judgment sharpens this map of shared human wisdom.")
    if st.button("🌍 Render the map"):
        with st.spinner("Drawing..."):
            st.session_state["map_html"] = build_map_html(df)
    if "map_html" in st.session_state:
        st.components.v1.html(st.session_state["map_html"], height=820, scrolling=True)
