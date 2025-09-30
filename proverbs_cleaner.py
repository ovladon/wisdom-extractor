
import re, pandas as pd

def normalise_text(s: str) -> str:
    s = str(s or "")
    s = re.sub(r"(?i)\benglish\s*equivalent\s*:\s*","", s).strip()
    s = re.sub(r"\s+"," ", s).strip()
    return s

def is_name_like(txt: str) -> bool:
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", txt)
    if re.search(r"(?i)\b(ISBN|ISSN|Retrieved from|Privacy Policy|Terms of Use|Creative Commons|What links here|Category:)\b", txt):
        return True
    return (len(tokens) > 0 and len(tokens) <= 3 and all(t[:1].isupper() for t in tokens))

def has_proverb_cues(txt: str) -> bool:
    if re.search(r"[.!?;:]", txt): return True
    if re.search(r"(?i)\b(is|are|was|were|be|have|do|does|did|shall|will|can|may|should|must|let|make|take|give|go|come|keep|put|never|don'?t|avoid|if|when|where|who|what|speak|say|think|know|trust|bite|bark|save|waste|pay|buy|sell|wait|hasten|repent|many|too)\b", txt):
        return True
    return False

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["people","saying","english_equivalent","original","source_title","source_url","accessed"]:
        if col not in df.columns: df[col] = ""
    df["saying"] = df["saying"].apply(normalise_text)
    df["english_equivalent"] = df["english_equivalent"].apply(normalise_text)
    df["basis"] = df["english_equivalent"].replace("", pd.NA).fillna(df["saying"]).astype(str)
    keeps = []
    for _, row in df.iterrows():
        txt = (row["basis"] or "").strip()
        if not txt or len(txt) < 6: keeps.append(False); continue
        if is_name_like(txt) and not has_proverb_cues(txt): keeps.append(False); continue
        if re.search(r"(?i)\b(see also|external links|references|navigation|category|^edit$|^jump to)\b", txt): keeps.append(False); continue
        if re.search(r"(?i)\b(ISBN|ISSN)\b", txt) or re.search(r"(?i)\bp\.\s*\d+\b", txt): keeps.append(False); continue
        keeps.append(True)
    df2 = df[pd.Series(keeps, index=df.index)]
    df2 = df2.drop_duplicates(subset=["people","basis"]).copy()
    mask = df2["original"].astype(str).str.strip().eq("")
    df2.loc[mask, "original"] = df2.loc[mask, "saying"]
    return df2
