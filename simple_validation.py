import streamlit as st
import pandas as pd
import json
import numpy as np
from datetime import datetime

def simple_cluster_validation():
    """Lightweight validation interface that actually gets used"""
    st.subheader("Quick Cluster Quality Check")
    st.write("Rate a few clusters to get a sense of clustering quality. Takes 2-3 minutes.")
    
    # Load clusters data
    clusters_file = st.text_input("Clusters JSON file", value="wisdom_clusters.json", key="val_clusters")
    
    if not st.session_state.get("validation_data"):
        if st.button("Start Quick Validation (10 clusters)"):
            try:
                with open(clusters_file, "r", encoding="utf-8") as f:
                    all_clusters = json.load(f)
                
                # Smart sampling: mix of high-scoring and random clusters
                high_score = sorted(all_clusters, key=lambda x: x.get("wisdom_score", 0), reverse=True)[:5]
                remaining = [c for c in all_clusters if c not in high_score]
                random_sample = np.random.choice(len(remaining), min(5, len(remaining)), replace=False)
                random_clusters = [remaining[i] for i in random_sample]
                
                selected = high_score + random_clusters
                st.session_state.validation_data = {
                    "clusters": selected,
                    "current": 0,
                    "ratings": [],
                    "start_time": datetime.now()
                }
                st.success("Validation started! Rate the clusters below.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not load clusters: {e}")
    
    # Show validation interface if data exists
    if st.session_state.get("validation_data"):
        data = st.session_state.validation_data
        clusters = data["clusters"]
        current_idx = data["current"]
        
        if current_idx >= len(clusters):
            show_validation_summary()
            return
        
        cluster = clusters[current_idx]
        
        # Progress bar
        progress = current_idx / len(clusters)
        st.progress(progress)
        st.write(f"Cluster {current_idx + 1} of {len(clusters)}")
        
        # Show cluster info
        st.markdown(f"**Representative Claim:** {cluster['claim']}")
        st.write(f"**Coverage:** {cluster['coverage']} peoples | **Support:** {cluster['support']} examples")
        
        # Show 3-4 examples from different cultures
        examples = cluster.get("examples", {})
        if examples:
            st.markdown("**Examples from different cultures:**")
            for culture, example in list(examples.items())[:4]:
                st.write(f"• **{culture}:** {example}")
        
        st.markdown("---")
        
        # Simple rating interface
        col1, col2 = st.columns(2)
        
        with col1:
            coherence = st.radio(
                "Do these examples express the same core idea?",
                ["Yes, clearly", "Mostly yes", "Somewhat", "Not really", "No"],
                key=f"coherence_{current_idx}"
            )
        
        with col2:
            quality = st.radio(
                "Overall cluster quality?",
                ["Excellent", "Good", "Fair", "Poor"],
                key=f"quality_{current_idx}"
            )
        
        # Optional quick feedback
        comment = st.text_input("Quick note (optional):", key=f"comment_{current_idx}")
        
        # Navigation
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if current_idx > 0 and st.button("← Previous"):
                st.session_state.validation_data["current"] = current_idx - 1
                st.rerun()
        
        with col2:
            if st.button("Next →"):
                # Save rating
                rating = {
                    "cluster_id": cluster.get("cluster_id"),
                    "coherence": coherence,
                    "quality": quality,
                    "comment": comment,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Update or append rating
                ratings = data["ratings"]
                existing_idx = next((i for i, r in enumerate(ratings) if r["cluster_id"] == cluster.get("cluster_id")), None)
                if existing_idx is not None:
                    ratings[existing_idx] = rating
                else:
                    ratings.append(rating)
                
                st.session_state.validation_data["current"] = current_idx + 1
                st.rerun()
        
        with col3:
            if st.button("Skip"):
                st.session_state.validation_data["current"] = current_idx + 1
                st.rerun()

def show_validation_summary():
    """Show simple validation results"""
    data = st.session_state.validation_data
    ratings = data["ratings"]
    
    if not ratings:
        st.warning("No ratings recorded.")
        return
    
    st.success("Validation complete! Here's your assessment:")
    
    # Convert ratings to scores
    coherence_scores = {"Yes, clearly": 5, "Mostly yes": 4, "Somewhat": 3, "Not really": 2, "No": 1}
    quality_scores = {"Excellent": 4, "Good": 3, "Fair": 2, "Poor": 1}
    
    coherence_vals = [coherence_scores.get(r["coherence"], 3) for r in ratings]
    quality_vals = [quality_scores.get(r["quality"], 2) for r in ratings]
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_coherence = np.mean(coherence_vals)
        st.metric("Avg Coherence", f"{avg_coherence:.1f}/5")
        if avg_coherence >= 4:
            st.success("High coherence")
        elif avg_coherence >= 3:
            st.warning("Moderate coherence")
        else:
            st.error("Low coherence")
    
    with col2:
        avg_quality = np.mean(quality_vals)
        st.metric("Avg Quality", f"{avg_quality:.1f}/4")
        if avg_quality >= 3:
            st.success("Good quality")
        elif avg_quality >= 2:
            st.warning("Fair quality")
        else:
            st.error("Poor quality")
    
    with col3:
        problem_clusters = sum(1 for r in ratings if coherence_scores.get(r["coherence"], 3) <= 2)
        st.metric("Problem Clusters", f"{problem_clusters}/{len(ratings)}")
    
    # Simple interpretation
    st.markdown("### Quick Assessment")
    
    if avg_coherence >= 3.5 and avg_quality >= 2.5:
        st.success("✓ Clustering appears to be working well. Most clusters group similar wisdom coherently.")
    elif avg_coherence >= 2.5 or avg_quality >= 2:
        st.warning("⚠ Clustering has mixed results. Consider adjusting the distance threshold or improving text preprocessing.")
    else:
        st.error("✗ Clustering quality is poor. Try a lower distance threshold (more, smaller clusters) or check data quality.")
    
    # Comments
    comments = [r["comment"] for r in ratings if r["comment"].strip()]
    if comments:
        st.markdown("### Your Comments")
        for i, comment in enumerate(comments, 1):
            st.write(f"{i}. {comment}")
    
    # Export option
    if st.button("Export Validation Results"):
        export_data = {
            "validation_summary": {
                "avg_coherence": avg_coherence,
                "avg_quality": avg_quality,
                "total_clusters": len(ratings),
                "problem_clusters": problem_clusters,
                "validation_time": (datetime.now() - data["start_time"]).total_seconds()
            },
            "detailed_ratings": ratings
        }
        
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            "Download Validation Data",
            data=json_str.encode("utf-8"),
            file_name=f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )
    
    if st.button("Start New Validation"):
        del st.session_state.validation_data
        st.rerun()

def add_simple_validation_tab():
    """Function to add to main app tabs"""
    simple_cluster_validation()
