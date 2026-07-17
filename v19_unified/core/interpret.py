"""Deterministic interpretation (from v10's interpret_v2): themes + ecological correlations.

Assigns each cluster to a wisdom theme via keyword matching on the canonical claim,
then correlates per-people theme shares with coarse ecological/institutional proxies
from data/people_metadata.csv (Spearman). Fully offline, no LLM required.
Correlations are suggestive associations, not causal claims (see paper Section 6.3).
"""
import re
from collections import Counter

import numpy as np
import pandas as pd

THEMES = [
    ("Cooperation / Social Support",
     ["cooperat", "together", "friend", "help", "union", "many hands", "community", "neighbor", "neighbour"]),
    ("Prudence / Risk & Uncertainty",
     ["avoid", "never", "caution", "careful", "haste", "risk", "danger", "guard", "beware",
      "look before", "prudence", "slow", "steady"]),
    ("Effort / Time / Persistence",
     ["work", "practice", "effort", "time", "early", "persever", "patience", "diligence", "delay"]),
    ("Trust / Honesty / Deception",
     ["honest", "truth", "deception", "lie", "false", "trust", "cheat", "thief", "wolf", "fox"]),
    ("Family / Kinship / Obligation",
     ["family", "blood", "kin", "father", "mother", "son", "daughter", "home", "house", "child", "apple", "tree"]),
    ("Fate / Fortune / Opportunity",
     ["fortune", "luck", "opportunity", "chance", "bold", "fate", "destiny", "star"]),
    ("Knowledge / Speech / Silence",
     ["word", "speak", "silence", "listen", "tongue", "knowledge", "book", "learn", "wisdom"]),
]

ORDINAL = {"low": 1, "med": 2, "medium": 2, "high": 3, "n": 0, "y": 1, "partial": 0.5}


def theme_of(text):
    low = str(text).lower()
    for theme, keys in THEMES:
        if any(k in low for k in keys):
            return theme
    return "Other"


def themed_report(cluster_summary, top_per_theme=8):
    """Group top clusters by theme; returns {theme: DataFrame} for display/export."""
    cs = cluster_summary.copy()
    cs["theme"] = cs["claim"].apply(theme_of)
    out = {}
    for theme, _ in THEMES + [("Other", [])]:
        sub = cs[cs["theme"] == theme].head(top_per_theme)
        if len(sub):
            out[theme] = sub[["cluster_id", "claim", "coverage", "support", "wisdom_score", "cultures"]]
    return out


def theme_shares_by_people(df):
    """Per-people distribution over themes (on claims; falls back to text)."""
    d = df.dropna(subset=["people"]).copy()
    if d.empty:
        return pd.DataFrame()
    textcol = "claim" if d["claim"].notna().any() else "text"
    d["theme"] = d[textcol].fillna(d["text"]).apply(theme_of)
    counts = d.pivot_table(index="people", columns="theme", values="id", aggfunc="count").fillna(0)
    return counts.div(counts.sum(axis=1), axis=0)


def _to_numeric(series):
    def conv(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if s in ORDINAL:
            return ORDINAL[s]
        try:
            return float(s)
        except ValueError:
            return np.nan
    return series.map(conv)


def ecological_correlations(df, metadata_csv, min_peoples=8):
    """Spearman rho between theme shares and metadata proxies across peoples."""
    from scipy.stats import spearmanr
    shares = theme_shares_by_people(df)
    if shares.empty:
        return pd.DataFrame()
    meta = pd.read_csv(metadata_csv)
    meta["people_key"] = meta["people"].str.strip().str.lower()
    shares.index = shares.index.astype(str).str.strip().str.lower()
    merged = meta.set_index("people_key").join(shares, how="inner")
    if len(merged) < min_peoples:
        return pd.DataFrame()
    proxy_cols = ["maritime_orientation", "trade_route_proximity", "migration_hub",
                  "urbanization_level", "individualism_proxy", "uncertainty_avoidance_proxy"]
    rows = []
    for proxy in proxy_cols:
        if proxy not in merged.columns:
            continue
        x = _to_numeric(merged[proxy])
        for theme in shares.columns:
            y = merged[theme]
            ok = x.notna() & y.notna()
            if ok.sum() < min_peoples:
                continue
            rho, p = spearmanr(x[ok], y[ok])
            rows.append({"proxy": proxy, "theme": theme, "spearman_rho": round(float(rho), 3),
                         "p_value": round(float(p), 4), "n_peoples": int(ok.sum())})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("p_value").reset_index(drop=True)
    return out


def productive_tensions(cluster_summary, min_coverage=3):
    """Find high-coverage claim pairs with opposing themes (paper's 'productive tensions')."""
    opposites = [
        ("cooperat|many hands|together|unity", "too many|excess of"),
        ("look before|caution|careful|slow", "hesitat|bold|fortune favors|he who dares"),
        ("silence|say nothing", "speak|squeaky|ask"),
        ("absence.*heart.*fonder", "out of sight"),
    ]
    top = cluster_summary[cluster_summary["coverage"] >= min_coverage]
    pairs = []
    for rx_a, rx_b in opposites:
        a = top[top["claim"].str.contains(rx_a, case=False, regex=True, na=False)]
        b = top[top["claim"].str.contains(rx_b, case=False, regex=True, na=False)]
        for _, ra in a.head(2).iterrows():
            for _, rb in b.head(2).iterrows():
                if ra["cluster_id"] != rb["cluster_id"]:
                    pairs.append({"claim_a": ra["claim"], "coverage_a": ra["coverage"],
                                  "claim_b": rb["claim"], "coverage_b": rb["coverage"]})
    return pd.DataFrame(pairs)
