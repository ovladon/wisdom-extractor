# Response plan to Dr. Pelican's observations (2026-07-18)

Source: "pt Vlad 18 iulie 2026.docx". Status per point:

| # | Observation | Status |
|---|---|---|
| 1a | Grow the annotation set substantially | **Platform ready** — desktop portal + mobile swipe app (v19.4/5); Waterloo annotator study is the planned vehicle |
| 1b | Report inter-annotator agreement (κ / Krippendorff's α) | **Implemented (v19.6)** — ordinal Krippendorff's α over multi-annotated pairs, shown in Diagnostics; reportable once pairs get ≥2 votes |
| 1c | Clear annotation protocol description | **Implemented** — her operational tests are now the in-app guide (mobile ⓘ overlay + portal expander) and will be the protocol section of the revised paper |
| 2 | Baselines: multilingual sentence embeddings / transformers / LLMs | **Planned** — the evaluation framework is representation-agnostic by design; add LaBSE/SBERT as second vectorizer scored on the same annotation set (top of research roadmap; good Waterloo joint experiment) |
| 3 | Define "same idea"; graded similarity levels | **Implemented (v19.6)** — her 6-level scheme (4..0, −1) is now the annotation instrument everywhere, two-stage on mobile (related? → which relation), with her per-level tests; hard-constraint mapping: ≥3 → must, ≤1 → cannot, 2 → recorded only |
| 4 | Evaluate canonicalization (impact + examples) | **Partially** — rewrite rate reported (10.9%); ablation (cluster with/without canonicalization, scored on annotations) planned |
| 5 | More clustering metrics + manual cluster validation | **Partially** — coverage bins, ARI, constraint agreement exist; manual sample validation of clusters = natural task for the annotation platform (add cluster-audit mode) |
| 6 | Systematic error analysis (FP/FN types) | **Partially** — must/cannot violations are listed in Diagnostics; typology write-up planned once the graded set grows |
| 7 | Non-European expansion + geographically balanced robustness subsets | **Planned** — sources are in the catalog; balanced-subset re-clustering is scriptable in the experiment suite |
| 8 | Control for shared historical/religious/linguistic influence; don't equate recurrence with universality | **Partially** — permutation triangulation vs family/region mixing is exactly this control and paper claims are already bounded; deeper controls (e.g., excluding Biblical/classical source lineages via the attestation layer) planned |
| 9 | Deeper theory (recurrence ↔ cultural evolution ↔ wisdom) | **Planned** — discussion section expansion for the revision; ties to Grossmann/Johnson folk-theories framing |

The scale's authorship is credited to her in-app ("Scheme: Dr. Elena Pelican") and the
protocol section of the revised manuscript should carry her scheme verbatim.

Note for the manuscript: existing 115 binary annotations remain valid (mapped 4/0 in α;
must/cannot for constraints); new annotations are graded. The paper revision should
describe both generations of the instrument.
