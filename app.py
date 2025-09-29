import streamlit as st
import pandas as pd, json, os
from dataset_builder import build_from_sources, merge_and_clean
from extractor import run_enhanced as run_extractor_enhanced  # Updated import
from interpret_v2 import summarize as run_interpret_det
from interpret_llm import run_llm as run_interpret_llm, build_summary_from_text, ensure_default_model
from diagnostics import compute_practical_diagnostics, compute_trust_score  # Updated import
from simple_validation import add_simple_validation_tab  # New import

st.set_page_config(page_title="Wisdom Extractor v8 - Improved", layout="wide")
st.title("Wisdom Extractor v8 - Improved")
st.caption("Realistic improvements: better preprocessing, adaptive clustering, simple validation, actionable diagnostics")

with st.sidebar:
    st.header("Settings")
    data_path = st.text_input("Dataset CSV path", value="proverbs_clean_v2.csv")
    meta_path = st.text_input("People metadata CSV", value="people_metadata_v2.csv")
    sources_path = st.text_input("Sources YAML", value="sources.yaml")
    
    st.markdown("---")
    st.markdown("**Key Improvements:**")
    st.write("• Better text normalization and canonicalization")
    st.write("• Adaptive distance threshold selection")
    st.write("• Language family diversity scoring")
    st.write("• Simple validation interface")
    st.write("• Actionable diagnostic recommendations")

tabs = st.tabs(["1) Dataset Builder", "2) Improved Extractor", "3) Results & Analysis", 
                "4) Quick Validation", "5) Interpretation", "6) Practical Diagnostics"])

with tabs[0]:
    # Dataset Builder - unchanged but with better progress feedback
    st.subheader("Build or extend a dataset")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Upload existing CSVs**")
        up = st.file_uploader("Upload one or more CSVs", type=["csv"], accept_multiple_files=True)
        uploaded_frames = []
        if up:
            for f in up:
                try:
                    df = pd.read_csv(f)
                    uploaded_frames.append(df)
                    st.success(f"Loaded: {f.name} - {df.shape[0]} rows")
                except Exception as e:
                    st.error(f"Could not read {f.name}: {e}")
    
    with c2:
        st.markdown("**Fetch from public sources**")
        try:
            import yaml
            srcs = yaml.safe_load(open(sources_path, "r", encoding="utf-8"))
            people_list = sorted(set([s.get("people","") for s in srcs]))
            type_list = sorted(set([s.get("type","wikiquote") for s in srcs]))
        except Exception as e:
            st.warning(f"Could not read {sources_path}: {e}")
            people_list, type_list = [], []
        
        # Better selection interface
        col_a, col_b = st.columns(2)
        with col_a:
            select_all = st.checkbox("Select all peoples", value=False)
            n_selected = st.slider("Number to select", 1, min(20, len(people_list)), 5)
        
        with col_b:
            if select_all:
                sel_people = people_list
            else:
                # Smart default selection - mix of major and diverse languages
                major_langs = ['English', 'Chinese', 'Spanish', 'Arabic', 'Hindi', 'French', 'Russian', 'Japanese']
                defaults = [p for p in people_list if p in major_langs][:n_selected//2]
                others = [p for p in people_list if p not in major_langs][:n_selected-len(defaults)]
                default_selection = defaults + others
                sel_people = st.multiselect("Selected peoples", people_list, default=default_selection)
        
        sel_types = st.multiselect("Source types", type_list, default=type_list[:2])
        sleep = st.slider("Delay (seconds)", 0.5, 3.0, 1.0, 0.5)
        
        if st.button("Fetch Selected Sources"):
            if not sel_people:
                st.warning("Please select at least one people group")
            else:
                with st.spinner(f"Fetching from {len(sel_people)} sources..."):
                    scraped_df = build_from_sources(sources_path, selected_people=sel_people, 
                                                  selected_types=sel_types, sleep=sleep, save_dir="runs")
                    st.success(f"Fetched {scraped_df.shape[0]} raw entries")
                    st.session_state["scraped_df"] = scraped_df.to_dict(orient="list")

    if st.button("Merge & Clean All Data"):
        scraped_df = pd.DataFrame(st.session_state.get("scraped_df", {})) if "scraped_df" in st.session_state else None
        
        with st.spinner("Processing and cleaning data..."):
            raw, clean = merge_and_clean(uploaded_frames, scraped_df, use_ai=False, model_path='auto')
            st.session_state["clean_df"] = clean.to_dict(orient="list") if clean is not None else {}
            
        if clean is not None and not clean.empty:
            st.success(f"Processed: {raw.shape[0] if raw is not None else 0} raw → {clean.shape[0]} clean rows")
            
            # Show sample of results
            st.dataframe(clean.head(20))
            
            # Save and download options
            clean.to_csv(data_path, index=False, encoding="utf-8")
            st.download_button("Download Cleaned CSV", 
                             data=clean.to_csv(index=False).encode("utf-8"),
                             file_name="proverbs_clean.csv", mime="text/csv")
        else:
            st.error("No data remained after cleaning. Check source quality or selection.")

with tabs[1]:
    # Improved Extractor
    st.subheader("Improved Clustering Engine")
    st.write("Enhanced preprocessing, adaptive parameters, and language family diversity scoring.")
    
    col1, col2 = st.columns(2)
    with col1:
        out_json = st.text_input("Output JSON", value="wisdom_clusters.json")
        out_csv = st.text_input("Output CSV", value="clusters.csv")
        coords_csv = st.text_input("Coordinates CSV", value="clusters_coords.csv")
    
    with col2:
        dist_threshold = st.slider("Base distance threshold", 0.1, 0.8, 0.35, 0.05,
                                  help="Will be automatically adjusted based on data characteristics")
        
        st.info("The system will automatically adjust the threshold based on your data size and similarity patterns.")
    
    if st.button("Run Improved Clustering"):
        if not os.path.exists(data_path):
            st.error(f"Data file not found: {data_path}")
        else:
            with st.spinner("Running enhanced clustering..."):
                try:
                    result_df = run_extractor_enhanced(data_path, out_json, out_csv, coords_csv, dist_threshold)
                    
                    st.success("Clustering complete!")
                    st.session_state.update({
                        "clusters_json": out_json,
                        "clusters_csv": out_csv, 
                        "coords_csv": coords_csv
                    })
                    
                    # Show quick summary
                    if result_df is not None and not result_df.empty:
                        st.markdown("### Quick Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Clusters", len(result_df))
                        with col2:
                            avg_coverage = result_df["coverage"].mean()
                            st.metric("Avg Coverage", f"{avg_coverage:.1f}")
                        with col3:
                            max_diversity = result_df.get("family_diversity", pd.Series([0])).max()
                            st.metric("Max Family Diversity", int(max_diversity))
                        with col4:
                            high_quality = sum(result_df["wisdom_score"] >= result_df["wisdom_score"].quantile(0.8))
                            st.metric("High Quality Clusters", high_quality)
                            
                except Exception as e:
                    st.error(f"Clustering failed: {e}")

with tabs[2]:
    # Results & Analysis with better visualizations
    st.subheader("Results Analysis")
    
    # Replace lines 169-171 in app.py with:
    cj = st.text_input("Clusters JSON", value=st.session_state.get("clusters_json", "wisdom_clusters.json"), key="results_cj")
    ccsv = st.text_input("Clusters CSV", value=st.session_state.get("clusters_csv", "clusters.csv"), key="results_ccsv")
    ccoords = st.text_input("Coordinates CSV", value=st.session_state.get("coords_csv", "clusters_coords.csv"), key="results_ccoords")
    
    if st.button("Load & Analyze Results"):
        try:
            # Load data
            data = json.load(open(cj, "r", encoding="utf-8"))
            df = pd.read_csv(ccsv)
            
            # Enhanced display
            st.markdown("### Top Clusters by Quality")
            display_cols = ["claim", "wisdom_score", "coverage", "support"]
            if "family_diversity" in df.columns:
                display_cols.append("family_diversity")
            
            top_clusters = df.head(15)[display_cols]
            st.dataframe(top_clusters)
            
            # Cluster inspection
            st.markdown("### Cluster Inspector")
            cluster_idx = st.selectbox("Select cluster to examine:", range(len(df)), format_func=lambda x: f"Cluster {x+1}: {df.iloc[x]['claim'][:50]}...")
            
            if cluster_idx is not None:
                cluster = df.iloc[cluster_idx]
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**Claim:** {cluster['claim']}")
                    st.write(f"**Quality Score:** {cluster['wisdom_score']}")
                    
                    # Show examples
                    cluster_data = data[cluster_idx]
                    examples = cluster_data.get("examples", {})
                    if examples:
                        st.markdown("**Examples by Culture:**")
                        for culture, example in list(examples.items())[:6]:
                            st.write(f"• **{culture}:** {example}")
                
                with col2:
                    st.metric("Coverage", f"{cluster['coverage']} peoples")
                    st.metric("Support", f"{cluster['support']} instances")
                    if "family_diversity" in cluster:
                        st.metric("Family Diversity", int(cluster["family_diversity"]))
                    
                    # Language families if available
                    if "language_families" in cluster_data:
                        families = cluster_data["language_families"]
                        st.write("**Language Families:**")
                        for family in families:
                            st.write(f"• {family}")
            
            # Visualization
            st.markdown("### Cluster Visualization")
            try:
                coords_df = pd.read_csv(ccoords)
                
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 1, figsize=(10, 7))
                
                # Color by family diversity if available
                if "family_diversity" in coords_df.columns:
                    scatter = ax.scatter(coords_df["x"], coords_df["y"], 
                                       s=coords_df["coverage"] * 20, 
                                       c=coords_df["family_diversity"],
                                       alpha=0.7, cmap="viridis")
                    plt.colorbar(scatter, label="Language Family Diversity")
                else:
                    ax.scatter(coords_df["x"], coords_df["y"],
                             s=coords_df["coverage"] * 20,
                             alpha=0.7, color='blue')
                
                ax.set_xlabel("Dimension 1")
                ax.set_ylabel("Dimension 2") 
                ax.set_title("Cluster Semantic Map (size = coverage)")
                
                st.pyplot(fig)
                
            except Exception as e:
                st.warning(f"Could not generate visualization: {e}")
                
        except Exception as e:
            st.error(f"Could not load results: {e}")

with tabs[3]:
    # Quick Validation
    add_simple_validation_tab()

with tabs[4]:
    # Interpretation - unchanged
    st.subheader("Interpretation")
    
    cj = st.text_input("Clusters JSON", value=st.session_state.get("clusters_json","wisdom_clusters.json"), key="int_json")
    meta = st.text_input("Metadata CSV", value=meta_path, key="int_meta")
    out_report = st.text_input("Output report", value="interpretation_report.txt")
    
    if st.button("Generate Report"):
        try:
            report_path = run_interpret_det(cj, meta, out_report)
            report_text = open(report_path, "r", encoding="utf-8").read()
            
            st.success(f"Report generated: {report_path}")
            st.text_area("Generated Report", value=report_text, height=400)
            
            st.download_button("Download Report", 
                             data=report_text.encode("utf-8"),
                             file_name=os.path.basename(out_report), 
                             mime="text/plain")
        except Exception as e:
            st.error(f"Report generation failed: {e}")

with tabs[5]:
    # Practical Diagnostics
    st.subheader("Practical Diagnostics & Recommendations")
    st.write("Actionable analysis of clustering quality with specific recommendations.")
    
    ccsv_diag = st.text_input("Clusters CSV", value=st.session_state.get("clusters_csv","clusters.csv"), key="diag_csv")
    ccoords_diag = st.text_input("Coordinates CSV", value=st.session_state.get("coords_csv","clusters_coords.csv"), key="diag_coords")
    meta_diag = st.text_input("Metadata CSV", value=meta_path, key="diag_meta")
    
    if st.button("Run Practical Diagnostics"):
        try:
            # Load data with error handling
            dfc = pd.read_csv(ccsv_diag) if os.path.exists(ccsv_diag) else pd.DataFrame()
            dfcoords = pd.read_csv(ccoords_diag) if os.path.exists(ccoords_diag) else pd.DataFrame()
            meta_df = pd.read_csv(meta_diag) if os.path.exists(meta_diag) else pd.DataFrame()
            
            # Fix cultures column if needed
            if not dfc.empty and "cultures" in dfc.columns:
                def parse_cultures(x):
                    if isinstance(x, str) and x.startswith('['):
                        try:
                            import ast
                            return ast.literal_eval(x)
                        except:
                            return []
                    return x if isinstance(x, list) else []
                
                dfc["cultures"] = dfc["cultures"].apply(parse_cultures)
            
            # Run diagnostics
            diagnostics, correlations = compute_practical_diagnostics(dfc, dfcoords, meta_df)
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### Overall Assessment")
                
                trust_score = diagnostics.get("trust_score", 0)
                trust_level = diagnostics.get("interpretation", {}).get("trust_level", "Unknown")
                
                # Color-coded trust score
                if trust_score >= 7:
                    st.success(f"Trust Score: {trust_score}/10 ({trust_level})")
                elif trust_score >= 4:
                    st.warning(f"Trust Score: {trust_score}/10 ({trust_level})")
                else:
                    st.error(f"Trust Score: {trust_score}/10 ({trust_level})")
                
                # Main issues
                main_issues = diagnostics.get("interpretation", {}).get("main_issues", [])
                if main_issues:
                    st.markdown("**Priority Issues:**")
                    for issue in main_issues:
                        st.write(f"• {issue}")
            
            with col2:
                st.markdown("### Key Metrics")
                
                metrics = {
                    "Silhouette": diagnostics.get("silhouette", "N/A"),
                    "Compactness": diagnostics.get("compactness", "N/A"),
                    "Stability": diagnostics.get("stability", "N/A")
                }
                
                for metric, value in metrics.items():
                    if isinstance(value, float) and not pd.isna(value):
                        st.metric(metric, f"{value:.3f}")
                    else:
                        st.metric(metric, "N/A")
            
            # Detailed recommendations
            recommendations = diagnostics.get("recommendations", [])
            if recommendations:
                st.markdown("### Actionable Recommendations")
                
                for rec in recommendations:
                    priority = rec.get("priority", "Medium")
                    if priority == "High":
                        st.error(f"**{rec['issue']}**")
                    elif priority == "Medium":
                        st.warning(f"**{rec['issue']}**")
                    else:
                        st.info(f"**{rec['issue']}**")
                    
                    st.write(f"Cause: {rec.get('cause', 'Unknown')}")
                    st.write(f"Action: {rec.get('action', 'No specific action suggested')}")
                    st.markdown("---")
            
            # Cluster characteristics
            cluster_chars = diagnostics.get("cluster_characteristics", {})
            if cluster_chars:
                st.markdown("### Data Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Clusters", cluster_chars.get("total_clusters", 0))
                with col2:
                    st.metric("Total Items", cluster_chars.get("total_items", 0))
                with col3:
                    st.metric("Singleton Clusters", cluster_chars.get("singleton_clusters", 0))
                with col4:
                    st.metric("Largest Cluster", cluster_chars.get("largest_cluster_size", 0))
            
        except Exception as e:
            st.error(f"Diagnostics failed: {e}")
            st.write("Please check that your data files exist and are properly formatted.")
