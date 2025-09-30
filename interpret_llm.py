
import os
from ai_filters import _ensure_llm

PROMPT = """You are a cultural analytics researcher. You receive summaries of cross-cultural proverb clusters.
Each item lists: claim, coverage (number of peoples), support (total instances), and top metadata (regions, families, climates, coastal/island/maritime, trade, migration, subsistence, staple, legal, urban, values).

TASK:
1) Identify 5–8 macro-themes capturing the highest-coverage claims.
2) For each theme, select 1–2 representative claims and provide a short, articulate rationale grounded ONLY in the supplied evidence.
3) Offer 3–5 testable hypotheses relating themes to context (climate, trade, subsistence, law, urbanisation, values).
4) Note any apparent contradictions and propose contexts where each side holds.
Be concise, structured, and avoid speculation beyond the evidence provided.
"""

def ensure_default_model():
    llm = _ensure_llm("auto")
    return (llm is not None)

def run_llm(model_path, summary_text, out_path="interpretation_llm.txt"):
    try:
        from llama_cpp import Llama
    except Exception as e:
        open(out_path,"w",encoding="utf-8").write(f"[WARN] llama-cpp-python unavailable: {e}")
        return out_path
    if model_path == "auto":
        llm = _ensure_llm("auto")
        if llm is None:
            open(out_path,"w",encoding="utf-8").write("[WARN] Model not found and auto-download failed. Use deterministic mode or place a .gguf under ./models.")
            return out_path
    else:
        if not os.path.exists(model_path):
            open(out_path,"w",encoding="utf-8").write(f"[WARN] Model not found at: {model_path}")
            return out_path
        llm = Llama(model_path=model_path, n_ctx=4096, n_threads=4, verbose=False)
    messages = [
        {"role":"system","content":"You analyse cultural data concisely and cautiously."},
        {"role":"user","content": PROMPT + "\n\n=== CLUSTERS ===\n" + summary_text + "\n=== END ==="}
    ]
    out = llm.create_chat_completion(messages=messages, temperature=0.4, max_tokens=900)
    text = out["choices"][0]["message"]["content"]
    open(out_path,"w",encoding="utf-8").write(text)
    return out_path

def build_summary_from_text(det_report_text):
    lines = det_report_text.splitlines()
    if len(lines) > 400: lines = lines[:400]
    return "\n".join(lines)
