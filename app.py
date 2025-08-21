import streamlit as st
import pandas as pd, json, os
from dataset_builder import build_from_sources, merge_and_clean
from extractor import run as run_extractor
from interpret_v2 import summarize as run_interpret_det
from interpret_llm import run_llm as run_interpret_llm, build_summary_from_text, ensure_default_model
import matplotlib.pyplot as plt

st.set_page_config(page_title="Wisdom Extractor v7.7", layout="wide")
st.title("🧭 Wisdom Extractor v7.7")
st.caption("Dataset builder → Wisdom extraction → Results & visualisation → Interpretation (no APIs; optional local LLM) → Diagnostics")

with st.sidebar:
    st.header("Settings")
    data_path = st.text_input("Dataset CSV path", value="proverbs_clean_v2.csv")
    meta_path = st.text_input("People metadata CSV", value="people_metadata_v2.csv")
    sources_path = st.text_input("Sources YAML", value="sources.yaml")
    st.markdown("---")
    st.markdown("**Tip:** Use the tabs to go step by step.")

tabs = st.tabs(["1) Dataset Builder", "2) Wisdom Extractor", "3) Results & Visualisation", "4) Interpretation", "5) Diagnostics"])

with tabs[0]:
    st.subheader("Build or extend a dataset")
    st.write("Upload your own CSVs and/or fetch from multiple public sources. We preserve originals and provenance, then clean and deduplicate.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Upload existing CSVs** (they will be concatenated):")
        up = st.file_uploader("Upload one or more CSVs", type=["csv"], accept_multiple_files=True, key="uploader")
        uploaded_frames = []
        if up:
            for f in up:
                df = pd.read_csv(f)
                uploaded_frames.append(df)
                st.write(f"Loaded: {f.name} → {df.shape[0]} rows")
    with c2:
        st.markdown("**Fetch from public sources** (polite; CC BY-SA/PD; provenance saved):")
        try:
            import yaml
            srcs = yaml.safe_load(open(sources_path, "r", encoding="utf-8"))
            people_list = sorted(set([s["people"] for s in srcs]))
            type_list = sorted(set([s.get("type","wikiquote") for s in srcs]))
        except Exception as e:
            st.warning(f"Could not read {sources_path}: {e}")
            people_list, type_list = [], []
        select_all_people = st.checkbox("Select all peoples", value=True)
        default_people = people_list if select_all_people else []
        sel_people = st.multiselect("Pick peoples to fetch", options=people_list, default=default_people)
        sel_types = st.multiselect("Pick source types", options=type_list, default=type_list)
        sleep = st.slider("Delay between requests (seconds)", 0.2, 5.0, 1.0, 0.2)
        use_ai = st.checkbox("Use offline AI to refine proverb detection (if local model available)", value=False)
        ai_model = st.text_input("AI model path for filtering ('.gguf' or 'auto')", value="auto")
        scraped_df = None
        if st.button("🌐 Fetch selected"):
            scraped_df = build_from_sources(sources_path, selected_people=sel_people, selected_types=sel_types, sleep=sleep, save_dir="runs")
            st.success(f"Fetched {scraped_df.shape[0]} raw rows (also saved per-source CSVs under ./runs).")
            st.dataframe(scraped_df.head(20))
            st.session_state["scraped_df"] = scraped_df.to_dict(orient="list")
        elif "scraped_df" in st.session_state:
            scraped_df = pd.DataFrame(st.session_state["scraped_df"])

    if st.button("🧹 Merge & Clean"):
        raw, clean = merge_and_clean(uploaded_frames, scraped_df, use_ai=use_ai, model_path=ai_model)
        st.session_state["clean_df"] = clean.to_dict(orient="list")
        st.write(f"Raw rows: {0 if raw is None else raw.shape[0]} → Clean rows: {0 if clean is None else clean.shape[0]}")
        if clean is not None and not clean.empty:
            st.dataframe(clean.head(50))
            st.download_button("Download cleaned CSV", data=clean.to_csv(index=False).encode("utf-8"),
                               file_name="proverbs_clean_v2.csv", mime="text/csv")
            clean.to_csv(data_path, index=False, encoding="utf-8")
            st.success(f"Wrote {data_path}")
        else:
            st.warning("No rows after cleaning. Consider relaxing filters or adding sources.")

with tabs[1]:
    st.subheader("Cluster the proverbs into candidate wisdom claims")
    st.caption("We group similar claims based on character-level similarity. The distance threshold controls how tight clusters are: lower = stricter (more, smaller clusters); higher = looser (fewer, larger clusters).")
    out_json = st.text_input("Output JSON", value="wisdom_clusters.json")
    out_csv = st.text_input("Output CSV", value="clusters.csv")
    coords_csv = st.text_input("Coords CSV (for plots)", value="clusters_coords.csv")
    dist = st.slider("Clustering distance threshold", 0.1, 0.9, 0.35, 0.01, help="Lower = stricter clustering (more clusters), Higher = looser (fewer clusters). Recommended: 0.3–0.4")
    if st.button("⚙️ Run Extractor"):
        try:
            run_extractor(data_path, out_json, out_csv, coords_csv, dist)
            st.success(f"Extraction complete → {out_json}, {out_csv}, {coords_csv}")
            st.session_state["clusters_json"] = out_json
            st.session_state["clusters_csv"] = out_csv
            st.session_state["coords_csv"] = coords_csv
        except Exception as e:
            st.error(f"Extraction failed: {e}")

with tabs[2]:
    st.subheader("Explore results")
    st.write("Top claims show their coverage (distinct peoples), support (total examples), and a composite score. The 2D map helps spot broad clusters.")
    cj = st.text_input("Clusters JSON", value=st.session_state.get("clusters_json","wisdom_clusters.json"))
    ccsv = st.text_input("Clusters CSV", value=st.session_state.get("clusters_csv","clusters.csv"))
    ccoords = st.text_input("Coords CSV", value=st.session_state.get("coords_csv","clusters_coords.csv"))
    if st.button("📥 Load Results"):
        try:
            data = json.load(open(cj, "r", encoding="utf-8"))
            df = pd.read_csv(ccsv)
            st.write(f"Clusters: {len(df)}")
            st.dataframe(df[["claim","wisdom_score","coverage","support","cultures"]].head(200))
            st.markdown("**Click a row index to inspect a cluster**")
            sel = st.number_input("Inspect cluster row", min_value=0, max_value=max(0,len(df)-1), value=0, step=1)
            row = df.iloc[int(sel)]
            import re, json as _json
            st.markdown("### Cluster card")
            st.write(f"**Claim:** {row.claim}")
            st.write(f"Coverage: {row.coverage} peoples · Support: {row.support} · Wisdom score: {row.wisdom_score}")
            try:
                cultures = row.cultures if isinstance(row.cultures,list) else _json.loads(row.cultures)
            except Exception:
                cultures = []
            st.write("Cultures:", ", ".join(cultures) if cultures else "-")
            words = [w.lower() for w in re.findall(r"[\\w\\-]{3,}", str(row.claim))]
            from collections import Counter
            st.write("Top terms:", ", ".join(w for w,_ in Counter(words).most_common(5)))
            st.info("Meaning: This cluster groups near-duplicate advice across languages. The representative claim above is the most frequent canonical form; examples per culture are available in the JSON.")
            st.download_button("Download clusters.json", data=json.dumps(data, ensure_ascii=False, indent=2), file_name="wisdom_clusters.json", mime="application/json")
            st.download_button("Download clusters.csv", data=df.to_csv(index=False).encode("utf-8"), file_name="clusters.csv", mime="text/csv")
            st.markdown("### 2D Map of claims (auto-embedding)")
            st.caption("We project claims from high-dimensional text space into 2D (TF–IDF → SVD → UMAP, fallback PCA). 'Dim 1' and 'Dim 2' summarise textual variation; bigger points = broader coverage.")
            try:
                import matplotlib.pyplot as plt
                pts = pd.read_csv(ccoords)
                fig = plt.figure(figsize=(7,5))
                plt.scatter(pts["x"], pts["y"], s=10*(1+pts["coverage"]), alpha=0.7)
                plt.xlabel("Dim 1"); plt.ylabel("Dim 2"); plt.title("Claim map (size ∝ coverage)")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not plot: {e}")
        except Exception as e:
            st.error(f"Load failed: {e}")

with tabs[3]:
    st.subheader("Interpretation (no APIs)")
    st.write("Deterministic mode builds a human-readable report with context factors (coastal/island, maritime, trade, migration, subsistence, staple, legal, urban, values). Optional LLM mode uses a small local model for a more fluent narrative—still offline.")
    cj = st.text_input("Clusters JSON", value=st.session_state.get("clusters_json","wisdom_clusters.json"), key="int_cjson")
    meta = st.text_input("People metadata CSV", value="people_metadata_v2.csv", key="int_meta")
    out = st.text_input("Output report (deterministic)", value="interpretation_report_v2.txt")
    if st.button("🧠 Generate deterministic interpretation"):
        try:
            path = run_interpret_det(cj, meta, out)
            txt = open(path, "r", encoding="utf-8").read()
            st.success(f"Wrote {path}")
            st.text_area("Report", value=txt, height=420)
            st.download_button("Download interpretation", data=txt.encode("utf-8"), file_name=os.path.basename(out), mime="text/plain")
        except Exception as e:
            st.error(f"Interpretation failed: {e}")
    st.markdown("---")
    st.markdown("**Optional local LLM (llama.cpp, no APIs)**")
    st.caption("Leave the model path as 'auto' to use ./models (attempt small download once). Or upload your own .gguf below.")
    model = st.text_input("Local model path (.gguf) — 'auto' uses ./models or downloads TinyLlama", value="auto")
    out_llm = st.text_input("Output report (LLM)", value="interpretation_llm.txt")
    st.markdown("**Model options**: Use 'auto' (search ./models and attempt a small download) or upload a .gguf below.")
    up_model = st.file_uploader("Upload local .gguf to ./models", type=["gguf"], accept_multiple_files=False)
    if up_model is not None:
        os.makedirs("models", exist_ok=True)
        dest = os.path.join("models", up_model.name)
        with open(dest, "wb") as f:
            f.write(up_model.getbuffer())
        st.success(f"Saved model to {dest}")
    if st.button("Check / fetch default model"):
        ok = ensure_default_model()
        if ok:
            st.success("Default model ready in ./models")
        else:
            st.warning("Could not locate/download a default model. You can still use deterministic mode or upload a .gguf.")
    if st.button("🧩 Generate LLM-based interpretation"):
        try:
            det_out = "__det_tmp.txt"
            path_det = run_interpret_det(cj, meta, det_out)
            txt = open(path_det, "r", encoding="utf-8").read()
            summary = build_summary_from_text(txt)
            path = run_interpret_llm(model, summary, out_llm)
            content = open(path,"r",encoding="utf-8").read()
            st.success(f"Wrote {path}")
            st.text_area("LLM Report", value=content, height=420)
            st.download_button("Download LLM interpretation", data=content.encode("utf-8"), file_name=os.path.basename(out_llm), mime="text/plain")
        except Exception as e:
            st.error(f"LLM interpretation failed: {e}")

with tabs[4]:
    st.subheader("Diagnostics: how much to trust the map")
    st.write("We report clustering quality (silhouette), compactness (mean intra-cluster cosine distance), stability under 10% subsampling, and correlations between theme frequencies and context proxies (Spearman).")
    ccsv = st.text_input("Clusters CSV", value=st.session_state.get("clusters_csv","clusters.csv"), key="diag_ccsv")
    ccoords = st.text_input("Coords CSV", value=st.session_state.get("coords_csv","clusters_coords.csv"), key="diag_ccoords")
    meta = st.text_input("People metadata CSV", value="people_metadata_v2.csv", key="diag_meta")
    if st.button("🔎 Compute diagnostics"):
        try:
            import pandas as pd, os
            from diagnostics import compute_diagnostics
            dfc = pd.read_csv(ccsv)
            import ast
            def fix(x):
                try:
                    return ast.literal_eval(x) if isinstance(x,str) and x.startswith('[') else x
                except Exception:
                    return x
            if "cultures" in dfc.columns:
                dfc["cultures"] = dfc["cultures"].apply(fix)
            dfcoords = pd.read_csv(ccoords) if os.path.exists(ccoords) else pd.DataFrame()
            meta_df = pd.read_csv(meta)
            trust, corrs = compute_diagnostics(dfc, dfcoords, meta_df)
            st.json(trust)
            if corrs:
                st.write("**Theme × Proxy Spearman correlations (rho, p)**")
                st.json(corrs)
            st.caption("Interpretation: Silhouette ~0.2–0.5 is reasonable for short, noisy strings; lower suggests mixed clusters. Lower compactness is better. Stability is the fraction of top-K claims retained after dropping 10% of entries (closer to 1 is better).")
        except Exception as e:
            st.error(f"Diagnostics failed: {e}")
