
# Offline heuristic + optional local LLM filter for proverb-like lines
import os, re

def _heuristic_is_biblio(txt: str) -> bool:
    t = (txt or "").strip()
    if not t: return True
    bad = [
        r"(?i)\bISBN\b", r"(?i)\bISSN\b", r"(?i)\bp\.\s*\d+\b",
        r"(?i)Text is available under the Creative Commons", r"(?i)Terms of Use", r"(?i)Privacy Policy",
        r"(?i)What links here", r"(?i)Navigation menu", r"(?i)External links", r"(?i)References\b",
        r"(?i)Retrieved from", r"(?i)Category:", r"(?i)This page was last edited",
        r"(?i)Infobase Publishing", r"(?i)DeProverbio\.com", r"(?i)WorldCat", r"(?i)\bISBN:?\s*\d"
    ]
    if any(re.search(p, t) for p in bad):
        return True
    # Author-year citation-like
    if re.search(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\s*\(\d{4}\)\.", t) and re.search(r"\.\s*[A-Z]", t):
        return True
    # Very long lines without punctuation are unlikely to be proverbs
    if len(t.split()) > 40 and not re.search(r"[.!?;:]", t):
        return True
    return False

def _has_proverb_cues(txt: str) -> bool:
    if re.search(r"[.!?;:]", txt): return True
    if re.search(r"(?i)\b(if|then|never|always|should|must|avoid|prefer|time|friend|truth|honesty|fortune|blood|work|early|slow|patience|many|too)\b", txt):
        return True
    if re.match(r"(?i)^(if|where|better|many|too|time|practice|honesty|fortune|blood|early|slow|truth|silence|friend|work)\b", txt):
        return True
    return False

_llm = None
def _ensure_llm(model_path="auto"):
    global _llm
    if _llm is not None:
        return _llm
    try:
        from llama_cpp import Llama
    except Exception:
        return None
    if model_path == "auto":
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        if os.path.isdir(models_dir):
            ggufs = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith(".gguf")]
            if ggufs:
                model_path = ggufs[0]
        if model_path == "auto":
            # Best-effort tiny model
            try:
                from huggingface_hub import hf_hub_download
                os.makedirs(models_dir, exist_ok=True)
                path = hf_hub_download(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0-GGUF",
                                        filename="TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf",
                                        local_dir=models_dir)
                model_path = path
            except Exception:
                return None
    if not os.path.exists(model_path):
        return None
    try:
        _llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4, verbose=False)
        return _llm
    except Exception:
        return None

def llm_is_proverb(txt: str, model_path="auto") -> bool:
    llm = _ensure_llm(model_path)
    if llm is None:
        return (not _heuristic_is_biblio(txt)) and _has_proverb_cues(txt)
    prompt = f"Decide PROVERB vs NOT_PROVERB for this line (short advice/aphorism vs citation/license/nav):\nLINE: {txt}\nANSWER:"
    try:
        out = llm.create_completion(prompt, max_tokens=2, temperature=0.0)
        ans = out["choices"][0]["text"].strip().upper()
        return "PROVERB" in ans
    except Exception:
        return (not _heuristic_is_biblio(txt)) and _has_proverb_cues(txt)

def filter_rows_with_ai(texts, model_path="auto"):
    keep = []
    for t in texts:
        t = (t or "").strip()
        if not t or _heuristic_is_biblio(t):
            keep.append(False); continue
        if _has_proverb_cues(t) and len(t.split()) <= 40:
            keep.append(True); continue
        keep.append(llm_is_proverb(t, model_path=model_path))
    return keep
