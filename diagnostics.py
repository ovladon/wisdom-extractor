import numpy as np, pandas as pd, json, warnings
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr
from collections import Counter
import re

def _safe_silhouette(X, labels):
    try:
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            return float("nan"), "Only one cluster found"
        if len(unique_labels) >= len(labels):
            return float("nan"), "All singleton clusters"
        return float(silhouette_score(X, labels, metric="cosine")), "Computed successfully"
    except Exception as e:
        return float("nan"), f"Computation failed: {str(e)}"

def _compactness(X, labels):
    from sklearn.metrics.pairwise import cosine_distances
    vals = []
    cluster_info = []
    
    for lab in set(labels):
        idx = np.where(labels==lab)[0]
        if len(idx) >= 2:
            Xi = X[idx]
            D = cosine_distances(Xi)
            triu = np.triu_indices_from(D, k=1)
            if triu[0].size > 0:
                mean_dist = D[triu].mean()
                vals.append(mean_dist)
                cluster_info.append({"cluster": int(lab), "size": len(idx), "compactness": mean_dist})
    
    overall_compactness = float(np.mean(vals)) if vals else float("nan")
    return overall_compactness, cluster_info

def _stability_simple(claims, labels, k=5, iterations=3):
    """Simplified stability test - check if top claims remain consistent"""
    from collections import Counter
    
    # Get top k most common claims
    claim_counts = Counter(claims)
    original_top = set([claim for claim, _ in claim_counts.most_common(k)])
    
    # Subsample and check consistency
    n = len(claims)
    keep_ratios = [0.8, 0.9, 0.95]  # Different subsampling rates
    
    stability_scores = []
    for ratio in keep_ratios:
        keep_n = int(n * ratio)
        retained_counts = []
        
        for _ in range(iterations):
            indices = np.random.choice(n, keep_n, replace=False)
            subset_claims = [claims[i] for i in indices]
            subset_counts = Counter(subset_claims)
            subset_top = set([claim for claim, _ in subset_counts.most_common(k)])
            
            # What fraction of original top k are retained?
            retained = len(original_top & subset_top) / len(original_top)
            retained_counts.append(retained)
        
        stability_scores.append(np.mean(retained_counts))
    
    return np.mean(stability_scores)

def analyze_cluster_characteristics(df_clusters):
    """Analyze cluster size distribution and quality patterns"""
    if df_clusters.empty:
        return {}
    
    cluster_sizes = df_clusters.groupby('cluster_id').size() if 'cluster_id' in df_clusters.columns else pd.Series([])
    
    characteristics = {
        "total_clusters": len(df_clusters) if 'cluster_id' not in df_clusters.columns else df_clusters['cluster_id'].nunique(),
        "total_items": len(df_clusters),
        "singleton_clusters": sum(cluster_sizes == 1) if len(cluster_sizes) > 0 else 0,
        "largest_cluster_size": cluster_sizes.max() if len(cluster_sizes) > 0 else 0,
        "median_cluster_size": cluster_sizes.median() if len(cluster_sizes) > 0 else 0,
        "avg_cluster_size": cluster_sizes.mean() if len(cluster_sizes) > 0 else 0
    }
    
    # Quality distribution
    if 'wisdom_score' in df_clusters.columns:
        scores = df_clusters['wisdom_score']
        characteristics.update({
            "high_quality_clusters": sum(scores >= scores.quantile(0.8)),
            "low_quality_clusters": sum(scores <= scores.quantile(0.2)),
            "score_range": [float(scores.min()), float(scores.max())],
            "avg_score": float(scores.mean())
        })
    
    return characteristics

def simple_correlation_analysis(df_clusters, meta_df):
    """Simplified correlation analysis using available metadata"""
    if df_clusters.empty or meta_df.empty:
        return {}
    
    # Build theme frequency matrix
    def get_theme(claim):
        claim_lower = str(claim).lower()
        if any(word in claim_lower for word in ['friend', 'together', 'help', 'cooperat', 'many hands']):
            return 'Cooperation'
        elif any(word in claim_lower for word in ['avoid', 'never', 'careful', 'haste', 'danger']):
            return 'Caution'
        elif any(word in claim_lower for word in ['honest', 'truth', 'lie', 'trust']):
            return 'Honesty'
        elif any(word in claim_lower for word in ['time', 'early', 'late', 'patience']):
            return 'Timing'
        elif any(word in claim_lower for word in ['work', 'practice', 'effort']):
            return 'Effort'
        else:
            return 'Other'
    
    # Extract themes and cultures
    theme_culture_pairs = []
    for _, row in df_clusters.iterrows():
        theme = get_theme(row.get('claim', ''))
        cultures = row.get('cultures', [])
        if isinstance(cultures, str):
            try:
                import ast
                cultures = ast.literal_eval(cultures)
            except:
                cultures = []
        
        for culture in cultures:
            theme_culture_pairs.append({'theme': theme, 'people': culture})
    
    if not theme_culture_pairs:
        return {}
    
    theme_df = pd.DataFrame(theme_culture_pairs)
    theme_freq = theme_df.groupby(['people', 'theme']).size().unstack(fill_value=0)
    
    # Join with metadata
    meta_indexed = meta_df.set_index('people') if 'people' in meta_df.columns else pd.DataFrame()
    if meta_indexed.empty:
        return {}
    
    combined = meta_indexed.join(theme_freq, how='inner')
    if combined.empty:
        return {"note": "No overlap between cluster cultures and metadata"}
    
    # Test correlations with available metadata columns
    correlations = {}
    theme_cols = [col for col in theme_freq.columns if col in combined.columns]
    
    for meta_col in ['maritime_orientation', 'urbanization_level', 'individualism_proxy', 'uncertainty_avoidance_proxy']:
        if meta_col not in combined.columns:
            continue
        
        # Convert categorical to numeric if needed
        series = combined[meta_col]
        if series.dtype == 'object':
            mapping = {'Low': 1, 'Med': 2, 'High': 3}
            series = series.map(mapping).dropna()
        
        for theme in theme_cols:
            if len(series) < 3:  # Need minimum samples
                continue
            
            theme_series = combined.loc[series.index, theme]
            try:
                rho, p_val = spearmanr(series, theme_series, nan_policy='omit')
                if not np.isnan(rho):
                    correlations[f"{theme}_vs_{meta_col}"] = {
                        "correlation": float(rho),
                        "p_value": float(p_val),
                        "sample_size": len(series),
                        "significant": p_val < 0.05
                    }
            except:
                continue
    
    return correlations

def generate_actionable_recommendations(trust_metrics, cluster_chars, correlations):
    """Generate specific, actionable recommendations"""
    recommendations = []
    
    # Clustering quality recommendations
    sil_score = trust_metrics.get('silhouette', float('nan'))
    if np.isnan(sil_score):
        recommendations.append({
            "issue": "Silhouette score undefined",
            "cause": "Likely too few clusters or all singletons",
            "action": "Try reducing distance threshold (0.2-0.3) to create more, smaller clusters",
            "priority": "High"
        })
    elif sil_score < 0.15:
        recommendations.append({
            "issue": "Very low clustering quality",
            "cause": "Poor separation between clusters",
            "action": "Improve text preprocessing or try different distance threshold",
            "priority": "High"
        })
    
    # Singleton cluster issue
    singleton_pct = cluster_chars.get('singleton_clusters', 0) / max(cluster_chars.get('total_clusters', 1), 1)
    if singleton_pct > 0.5:
        recommendations.append({
            "issue": f"{singleton_pct:.1%} singleton clusters",
            "cause": "Distance threshold too strict",
            "action": "Increase distance threshold (0.4-0.5) to create larger clusters",
            "priority": "Medium"
        })
    
    # Data coverage recommendations
    total_items = cluster_chars.get('total_items', 0)
    if total_items < 100:
        recommendations.append({
            "issue": "Limited data size",
            "cause": "Small dataset",
            "action": "Add more sources or languages to improve pattern detection",
            "priority": "Medium"
        })
    
    # Imbalanced clusters
    largest = cluster_chars.get('largest_cluster_size', 0)
    median = cluster_chars.get('median_cluster_size', 0)
    if largest > median * 10:
        recommendations.append({
            "issue": "Highly imbalanced cluster sizes",
            "cause": "Some clusters too dominant",
            "action": "Check for data quality issues or very generic claims",
            "priority": "Low"
        })
    
    # Correlation insights
    significant_corrs = [k for k, v in correlations.items() if v.get('significant', False)]
    if len(significant_corrs) == 0:
        recommendations.append({
            "issue": "No significant cultural correlations found",
            "cause": "Limited cultural diversity or weak patterns",
            "action": "Add more diverse cultures or check metadata quality",
            "priority": "Low"
        })
    
    return recommendations

def compute_practical_diagnostics(df_clusters, df_coords, meta_df):
    """Practical diagnostics focused on actionable insights"""
    
    # Basic setup
    from sklearn.feature_extraction.text import TfidfVectorizer
    claims = df_clusters["claim"].astype(str).tolist() if "claim" in df_clusters.columns else []
    labels = df_clusters["cluster_id"].values if "cluster_id" in df_clusters.columns else np.array([])
    
    if len(claims) < 3:
        return {
            "status": "insufficient_data",
            "message": "Need at least 3 claims for meaningful analysis",
            "recommendations": [{"action": "Add more data", "priority": "High"}]
        }, {}
    
    # Compute features for silhouette
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1, max_features=1000)
    try:
        X = vec.fit_transform(claims)
    except Exception as e:
        return {"status": "error", "message": f"Feature extraction failed: {e}"}, {}
    
    # Core metrics
    sil_score, sil_note = _safe_silhouette(X, labels)
    compactness, cluster_details = _compactness(X, labels)
    stability = _stability_simple(claims, labels, k=min(5, len(set(claims))))
    
    # Cluster characteristics analysis
    cluster_chars = analyze_cluster_characteristics(df_clusters)
    
    # Simple correlation analysis
    correlations = simple_correlation_analysis(df_clusters, meta_df)
    
    # Trust score calculation
    trust_score = 0
    score_breakdown = {}
    
    # Silhouette contribution (0-4 points)
    if not np.isnan(sil_score):
        sil_points = min(4.0, max(0.0, sil_score * 8))  # Scale 0-0.5 to 0-4
        trust_score += sil_points
        score_breakdown["silhouette"] = sil_points
    
    # Compactness contribution (0-3 points)
    if not np.isnan(compactness):
        # Lower compactness is better, so invert
        comp_points = max(0.0, min(3.0, (0.6 - compactness) * 5)) if compactness <= 0.6 else 0
        trust_score += comp_points
        score_breakdown["compactness"] = comp_points
    
    # Stability contribution (0-3 points)
    if not np.isnan(stability):
        stab_points = stability * 3
        trust_score += stab_points
        score_breakdown["stability"] = stab_points
    
    trust_score = round(trust_score, 1)
    
    # Generate recommendations
    trust_metrics = {"silhouette": sil_score, "compactness": compactness, "stability": stability}
    recommendations = generate_actionable_recommendations(trust_metrics, cluster_chars, correlations)
    
    # Final diagnostic summary
    diagnostics = {
        "trust_score": trust_score,
        "trust_score_breakdown": score_breakdown,
        "max_possible_score": 10.0,
        "silhouette": sil_score,
        "silhouette_note": sil_note,
        "compactness": compactness,
        "stability": stability,
        "cluster_characteristics": cluster_chars,
        "recommendations": recommendations,
        "interpretation": {
            "trust_level": "High" if trust_score >= 7 else "Medium" if trust_score >= 4 else "Low",
            "main_issues": [r["issue"] for r in recommendations if r["priority"] == "High"],
            "data_quality": "Good" if cluster_chars.get("total_items", 0) >= 100 else "Limited"
        }
    }
    
    return diagnostics, correlations

# Backward compatibility
def compute_diagnostics(df_clusters, df_coords, meta_df):
    """Backward compatible interface"""
    diagnostics, correlations = compute_practical_diagnostics(df_clusters, df_coords, meta_df)
    
    # Return in expected format
    trust = {
        "silhouette": diagnostics.get("silhouette"),
        "compactness_mean_cosine_distance": diagnostics.get("compactness"),
        "stability_drop10_topk_retained": diagnostics.get("stability"),
        "rule_of_thumb": f"Trust score: {diagnostics.get('trust_score')}/10"
    }
    
    return trust, correlations

def compute_trust_score(silhouette, compactness, stability, labels):
    """Enhanced trust score that's more interpretable"""
    score = 0
    breakdown = {}
    details = {}
    
    # Silhouette (0-4 points)
    if pd.isna(silhouette):
        sil_pts = 0
        details["silhouette"] = "Undefined - check cluster count and data quality"
    else:
        sil_pts = min(4.0, max(0.0, float(silhouette) * 8))
        details["silhouette"] = f"Score: {silhouette:.3f} → {sil_pts:.1f}/4 points"
    
    score += sil_pts
    breakdown["silhouette_points"] = sil_pts
    
    # Compactness (0-3 points) 
    if pd.isna(compactness):
        comp_pts = 0
        details["compactness"] = "Undefined - no multi-item clusters"
    else:
        comp_pts = max(0.0, min(3.0, (0.6 - float(compactness)) * 5)) if compactness <= 0.6 else 0
        details["compactness"] = f"Distance: {compactness:.3f} → {comp_pts:.1f}/3 points (lower is better)"
    
    score += comp_pts
    breakdown["compactness_points"] = comp_pts
    
    # Stability (0-3 points)
    if pd.isna(stability):
        stab_pts = 0
        details["stability"] = "Could not compute stability"
    else:
        stab_pts = float(stability) * 3
        details["stability"] = f"Retention: {stability:.2f} → {stab_pts:.1f}/3 points"
    
    score += stab_pts
    breakdown["stability_points"] = stab_pts
    
    score = round(score, 1)
    
    # Interpretation
    if score >= 7:
        guidance = "High confidence: Clustering appears robust and meaningful"
    elif score >= 4:
        guidance = "Moderate confidence: Some clear patterns but room for improvement"
    else:
        guidance = "Low confidence: Consider adjusting parameters or data preprocessing"
    
    return score, breakdown, guidance, details
