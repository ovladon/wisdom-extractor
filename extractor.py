import argparse, json, re, pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from packaging import version
import sklearn
from collections import Counter

def improved_canonicalize(s):
    """Improved canonicalization with better normalization and pattern recognition"""
    t = str(s).strip().strip('"''"''')
    
    # Normalize whitespace and punctuation
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r'[""''`]', '"', t)
    t = re.sub(r'[–—]', '-', t)
    
    # Remove common prefixes/suffixes that don't affect meaning
    t = re.sub(r'^(old saying:|proverb:|they say:?|it is said:?)\s*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s*(- \w+ proverb|- \w+ saying)\.?$', '', t, flags=re.IGNORECASE)
    
    # Improved structural patterns - more comprehensive
    patterns = [
        # Comparisons and preferences
        (r'(?i)^(?:it\'?s )?better (?:to )?(.+?) than (?:to )?(.+?)\.?$', r'Better \1 than \2.'),
        (r'(?i)^(.+?) is better than (.+?)\.?$', r'Better \1 than \2.'),
        (r'(?i)^prefer (.+?) (?:over|to) (.+?)\.?$', r'Better \1 than \2.'),
        
        # Prohibitions and avoidance
        (r'(?i)^(?:you should )?(?:never|don\'?t|do not|cannot|can\'?t|avoid) (.+?)\.?$', r'Avoid \1.'),
        (r'(?i)^(?:it\'?s )?(?:bad|wrong|dangerous) to (.+?)\.?$', r'Avoid \1.'),
        
        # Conditional wisdom
        (r'(?i)^(?:when|if) (.+?), (?:then )?(.+?)\.?$', r'If \1, then \2.'),
        (r'(?i)^where (?:there\'?s|you (?:have|find)) (.+?), (?:there\'?s|you (?:have|find)) (.+?)\.?$', r'If there is \1, there is \2.'),
        
        # Timing and process
        (r'(?i)^(?:the )?early (.+?) (?:gets?|catches?) (?:the )?(.+?)\.?$', r'Early action brings \2.'),
        (r'(?i)^(?:practice|repetition) makes? (?:perfect|improvement)\.?$', r'Practice improves skill.'),
        (r'(?i)^time (?:is|equals?) (?:money|wealth|value)\.?$', r'Time has value.'),
        
        # Cooperation and collective action
        (r'(?i)^(?:many|multiple|several) hands (?:make|create) (?:light work|easy work)\.?$', r'Cooperation reduces effort.'),
        (r'(?i)^(?:together|unity) (?:we|is) (?:stand|strength)(?:, (?:divided|apart) (?:we )?fall)?\.?$', r'Unity creates strength.'),
        
        # Excess and moderation
        (r'(?i)^too (?:many|much) (.+?) (?:spoils?|ruins?) (?:the )?(.+?)\.?$', r'Excess of \1 harms \2.'),
        (r'(?i)^(?:all work|only work) and no play makes? (.+?) (?:a )?(?:dull|boring) (?:boy|person)\.?$', r'Balance work and rest.'),
        
        # Patience and haste
        (r'(?i)^(?:haste|rushing|hurrying) (?:makes|creates|causes) (?:waste|mistakes|errors)\.?$', r'Haste causes problems.'),
        (r'(?i)^(?:slow and )?steady wins (?:the )?race\.?$', r'Consistency beats speed.'),
        (r'(?i)^(?:good things|patience) (?:comes?|rewards) (?:to )?(?:those who|people who) wait\.?$', r'Patience brings rewards.'),
    ]
    
    for pattern, replacement in patterns:
        if re.search(pattern, t):
            t = re.sub(pattern, replacement, t)
            break
    
    # Clean up result
    t = t.strip()
    if t and t[-1] not in '.!?':
        t += '.'
    
    return t

def improved_preprocessing(text):
    """Enhanced preprocessing for better similarity detection"""
    # Basic cleaning
    text = str(text).strip().lower()
    
    # Normalize common variations
    text = re.sub(r'\b(?:a|an|the)\b', '', text)  # Remove articles
    text = re.sub(r'\b(?:is|are|was|were|be|being)\b', 'be', text)  # Normalize "be" verbs
    text = re.sub(r'\b(?:will|shall|would|should|could|can|may|might)\b', 'will', text)  # Normalize modals
    text = re.sub(r'\b(?:do|does|did|done)\b', 'do', text)  # Normalize "do" verbs
    text = re.sub(r'\b(?:have|has|had)\b', 'have', text)  # Normalize "have" verbs
    
    # Common word substitutions for better matching
    substitutions = {
        r'\b(?:person|people|man|men|woman|women|individual|one)\b': 'person',
        r'\b(?:home|house|dwelling)\b': 'home',
        r'\b(?:money|wealth|riches|fortune|gold)\b': 'wealth',
        r'\b(?:friend|buddy|companion|ally)\b': 'friend',
        r'\b(?:enemy|foe|opponent|rival)\b': 'enemy',
        r'\b(?:work|labor|effort|toil|task)\b': 'work',
        r'\b(?:time|moment|hour|day)\b': 'time',
        r'\b(?:speak|talk|say|tell|utter)\b': 'speak',
        r'\b(?:good|fine|excellent|great|wonderful)\b': 'good',
        r'\b(?:bad|terrible|awful|horrible|poor)\b': 'bad',
    }
    
    for pattern, replacement in substitutions.items():
        text = re.sub(pattern, replacement, text)
    
    # Remove excessive punctuation and normalize spacing
    text = re.sub(r'[^\w\s\.\!\?]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def adaptive_distance_threshold(claims, initial_threshold=0.35):
    """Automatically adjust distance threshold based on data characteristics"""
    n_claims = len(claims)
    
    # For small datasets, use stricter threshold
    if n_claims < 50:
        return min(0.25, initial_threshold)
    
    # For large datasets, can be more lenient
    if n_claims > 500:
        return max(0.45, initial_threshold)
    
    # Check diversity - if claims are very similar, use stricter threshold
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1, max_features=1000)
    try:
        X = vec.fit_transform(claims)
        sim_matrix = cosine_similarity(X)
        mean_similarity = np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])
        
        # If average similarity is high, use stricter threshold
        if mean_similarity > 0.3:
            return initial_threshold * 0.8
        elif mean_similarity < 0.1:
            return initial_threshold * 1.2
    except:
        pass
    
    return initial_threshold

def quality_score_enhanced(claim, coverage, support, language_families=None):
    """Enhanced quality scoring with linguistic diversity consideration"""
    base_score = coverage + 0.3 * support
    
    # Length bonus for substantive claims
    words = len(claim.split())
    if 5 <= words <= 15:
        base_score += 0.5
    elif words > 20:
        base_score -= 0.2  # Penalty for overly long claims
    
    # Structural complexity bonus
    if any(pattern in claim.lower() for pattern in ['if ', 'when ', 'better ', 'avoid ', 'never ']):
        base_score += 0.3
    
    # Linguistic family diversity bonus
    if language_families:
        unique_families = len(set(language_families))
        if unique_families >= 3:
            base_score += 0.5 * unique_families
    
    return round(base_score, 3)

def compute_enhanced_coords(claims, n_components=2, random_state=0):
    """Improved coordinate computation with better parameters"""
    # Use mixed n-gram approach for more robust feature extraction
    vec1 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), min_df=1, max_features=800)
    vec2 = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=1, max_features=500, 
                          stop_words='english')
    
    try:
        X1 = vec1.fit_transform(claims)
        X2 = vec2.fit_transform(claims)
        
        # Combine character and word features
        from scipy.sparse import hstack
        X_combined = hstack([X1, X2])
        
        # Better dimensionality reduction
        from sklearn.decomposition import TruncatedSVD
        n_components_svd = min(100, max(10, min(X_combined.shape) - 1))
        svd = TruncatedSVD(n_components=n_components_svd, random_state=random_state)
        X_reduced = svd.fit_transform(X_combined)
        
        # Try UMAP first, fallback to PCA
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=random_state,
                              metric='cosine', min_dist=0.0, n_neighbors=min(15, len(claims)//4))
            coords = reducer.fit_transform(X_reduced)
        except ImportError:
            from sklearn.decomposition import PCA
            coords = PCA(n_components=n_components, random_state=random_state).fit_transform(X_reduced)
        
        return coords
    except Exception as e:
        print(f"[WARN] Enhanced coord computation failed: {e}, using basic approach")
        # Fallback to original method
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1)
        X = vec.fit_transform(claims)
        from sklearn.decomposition import PCA
        return PCA(n_components=n_components, random_state=random_state).fit_transform(X.toarray())

def get_language_family_mapping():
    """Simple language family mapping for diversity scoring"""
    return {
        'English': 'Germanic', 'German': 'Germanic', 'Dutch': 'Germanic', 'Swedish': 'Germanic', 'Norwegian': 'Germanic',
        'Romanian': 'Romance', 'Italian': 'Romance', 'Spanish': 'Romance', 'French': 'Romance', 'Portuguese': 'Romance',
        'Russian': 'Slavic', 'Polish': 'Slavic', 'Czech': 'Slavic', 'Ukrainian': 'Slavic', 'Bulgarian': 'Slavic',
        'Chinese': 'Sino-Tibetan', 'Japanese': 'Japonic', 'Korean': 'Koreanic', 'Thai': 'Tai-Kadai',
        'Arabic': 'Afro-Asiatic', 'Hebrew': 'Afro-Asiatic', 'Turkish': 'Turkic', 'Finnish': 'Uralic',
        'Hindi': 'Indo-Iranian', 'Persian': 'Indo-Iranian', 'Bengali': 'Indo-Iranian', 'Urdu': 'Indo-Iranian',
        'Indonesian': 'Austronesian', 'Malay': 'Austronesian', 'Filipino': 'Austronesian', 'Vietnamese': 'Austro-Asiatic',
    }

def run_enhanced(csv_path, out_json, out_csv, coords_csv, distance_threshold=0.35):
    """Enhanced extraction with realistic improvements"""
    
    print("[INFO] Loading and preprocessing data...")
    df = pd.read_csv(csv_path).dropna(subset=["people","saying"])
    
    # Improved basis selection
    df["basis"] = df.get("english_equivalent", "").fillna("")
    mask = df["basis"].astype(str).str.strip().eq("")
    df.loc[mask, "basis"] = df["saying"].astype(str)
    df = df[df["basis"].astype(str).str.strip()!=""]
    
    # Enhanced preprocessing and canonicalization
    df["processed"] = df["basis"].apply(improved_preprocessing)
    df["claim"] = df["basis"].apply(improved_canonicalize)
    
    print(f"[INFO] Processed {len(df)} proverbs from {df['people'].nunique()} cultures")
    
    # Remove very short or very similar claims
    df = df[df["claim"].str.len() >= 10]  # Minimum length
    df = df.drop_duplicates(subset=["claim"], keep='first')  # Remove exact duplicates
    
    print(f"[INFO] After deduplication: {len(df)} unique claims")
    
    # Adaptive distance threshold
    claims = df["claim"].tolist()
    adjusted_threshold = adaptive_distance_threshold(claims, distance_threshold)
    print(f"[INFO] Using distance threshold: {adjusted_threshold:.3f} (adjusted from {distance_threshold:.3f})")
    
    # Enhanced TF-IDF with better parameters
    vec = TfidfVectorizer(
        analyzer="char_wb", 
        ngram_range=(3,6),  # Slightly wider range
        min_df=1, 
        max_df=0.9,  # Ignore very common patterns
        max_features=5000  # More features for better discrimination
    )
    
    print("[INFO] Computing similarity matrix...")
    X = vec.fit_transform(df["processed"])  # Use processed version for similarity
    sim = cosine_similarity(X)
    dist = 1 - sim

    # Clustering with adjusted parameters
    kwargs = dict(linkage="average", distance_threshold=adjusted_threshold, n_clusters=None)
    skver = version.parse(sklearn.__version__)
    
    print("[INFO] Performing clustering...")
    try:
        if skver >= version.parse("1.4"):
            clust = AgglomerativeClustering(metric='precomputed', **kwargs).fit(dist)
        else:
            clust = AgglomerativeClustering(affinity='precomputed', **kwargs).fit(dist)
    except TypeError:
        try:
            clust = AgglomerativeClustering(affinity='precomputed', **kwargs).fit(dist)
        except TypeError:
            clust = AgglomerativeClustering(metric='precomputed', **kwargs).fit(dist)

    df["cluster_id"] = clust.labels_
    n_clusters = len(set(clust.labels_))
    print(f"[INFO] Found {n_clusters} clusters")
    
    # Language family mapping
    lang_family_map = get_language_family_mapping()
    
    # Enhanced result generation
    out = []
    for cid, sub in df.groupby("cluster_id"):
        claim = sub["claim"].mode().iloc[0]  # Most common canonical form
        coverage = sub["people"].nunique()
        support = len(sub)
        cultures = sorted(sub["people"].astype(str).unique().tolist())
        
        # Get language families for diversity scoring
        families = [lang_family_map.get(culture, 'Other') for culture in cultures]
        unique_families = list(set(families))
        
        # Enhanced quality scoring
        score = quality_score_enhanced(claim, coverage, support, unique_families)
        
        # Representative examples (up to 2 per language family)
        examples = {}
        family_counts = Counter(families)
        for culture in cultures:
            family = lang_family_map.get(culture, 'Other')
            if family_counts[family] <= 2 or len(examples) < 8:  # Limit examples but ensure diversity
                culture_examples = sub[sub["people"] == culture]["saying"].tolist()
                if culture_examples:
                    examples[culture] = culture_examples[0]
        
        out.append({
            "cluster_id": int(cid),
            "claim": claim,
            "wisdom_score": score,
            "coverage": int(coverage),
            "support": int(support),
            "cultures": cultures,
            "language_families": unique_families,
            "family_diversity": len(unique_families),
            "examples": examples,
            "avg_claim_length": round(np.mean([len(c.split()) for c in sub["claim"]]), 1),
            "processing_metadata": {
                "distance_threshold": adjusted_threshold,
                "original_threshold": distance_threshold
            }
        })
    
    # Sort by enhanced scoring
    out = sorted(out, key=lambda r: (-r["wisdom_score"], -r["family_diversity"], -r["coverage"]))
    
    print(f"[INFO] Generated {len(out)} cluster results")
    
    # Save results
    df_out = pd.DataFrame(out)
    df_out.to_csv(out_csv, index=False)
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    
    # Enhanced coordinate computation
    print("[INFO] Computing visualization coordinates...")
    try:
        coords = compute_enhanced_coords(df_out["claim"].tolist(), n_components=2)
        df_coords = df_out[["cluster_id","claim","coverage","support","wisdom_score","family_diversity"]].copy()
        df_coords["x"] = coords[:,0]
        df_coords["y"] = coords[:,1]
        df_coords.to_csv(coords_csv, index=False)
        print(f"[INFO] Saved coordinates to {coords_csv}")
    except Exception as e:
        print(f"[WARN] Coordinate computation failed: {e}")
        df_coords = pd.DataFrame(columns=["cluster_id","claim","coverage","support","wisdom_score","family_diversity","x","y"])
        df_coords.to_csv(coords_csv, index=False)
    
    print(f"[INFO] Enhanced extraction complete!")
    print(f"[INFO] Results: {out_json}, {out_csv}, {coords_csv}")
    
    return df_out

# Backwards compatible interface
def run(csv_path, out_json, out_csv, coords_csv, distance_threshold=0.35):
    """Backwards compatible interface"""
    return run_enhanced(csv_path, out_json, out_csv, coords_csv, distance_threshold)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="proverbs_clean_v2.csv")
    ap.add_argument("--out_json", default="wisdom_clusters.json")
    ap.add_argument("--out_csv", default="clusters.csv")
    ap.add_argument("--coords_csv", default="clusters_coords.csv")
    ap.add_argument("--distance_threshold", type=float, default=0.35)
    args = ap.parse_args()
    
    run_enhanced(args.csv, args.out_json, args.out_csv, args.coords_csv, args.distance_threshold)
