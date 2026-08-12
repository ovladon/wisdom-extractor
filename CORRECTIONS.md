# Corrections

Errors found in this project's own published statements, with what was wrong, how it was
found, and what was done. Entries are permanent and append-only: a correction that can be
edited away is not a correction.

An error discovered in our own work is recorded here whether or not anyone else noticed
it, and whether or not the corrected value flatters us. The second case matters more than
the first.

---

## 2026-08-12 — Agreement statistics counted judgments, not annotators

**Severity:** affects one published statistic and one release-trigger baseline. No
analysis in any manuscript is affected. The ConsILR-2025 conference paper is unaffected —
it reports no inter-annotator agreement statistic.

### What was wrong

Agreement was computed over pairs carrying two or more judgment **rows**, rather than two
or more **distinct annotators**. A pair that the same person judged twice therefore
counted as two independent opinions, and where that person's second score differed from
their first, the difference entered the estimate as inter-annotator disagreement.

Annotators were, in part, being compared with themselves.

### Where it was stated

| Artifact | Statement | Status |
|---|---|---|
| Zenodo corpus **v1.0** `README.md` (DOI [10.5281/zenodo.21439286](https://doi.org/10.5281/zenodo.21439286)) | "Ordinal Krippendorff's alpha over the **32 multi-annotated pairs** at time of deposit: **0.751**" | corrected in v1.1 with an erratum; v1.0 remains citable |
| `data/last_corpus_release.json` | `double_rated: 35`, `alpha: 0.752` | corrected in place, original values retained under `superseded` |
| `.github/release-notes/v19.34.md` | "double-rated pairs stand at 107 against a threshold of **185**" | superseded; correct threshold is **163**. The release note is left as published — it is a dated record |
| `scripts/iaa_report.py` output | `pairs_multi_annotated` | fixed in v19.36; field renamed `pairs_double_rated` |

Not affected, checked explicitly: the ConsILR-2025 paper; `annotations.csv` (the 115-pair
evaluation set analysed in the companion papers, which carries no per-annotator repeats);
`proverbs.csv`; `clusters.csv`; `people_metadata.csv`; the Zenodo record's own metadata
and `CITATION.cff`; the project website, which describes the practice but quotes no
figures.

### Corrected values

For the v1.0 deposit's graded annotation stream (237 judgments, 3 annotators):

| | published | correct |
|---|---|---|
| pairs with ≥2 distinct annotators | 32 | **13** |
| ordinal Krippendorff's α | 0.751 | **0.968** |
| 95% CI on α (bootstrap over units) | not stated | **[−0.09, 1.00]** |

**The corrected α is not a result and should not be quoted as one.** At 13 units the
interval spans essentially the whole admissible range. The correct conclusion is that the
deposit's double-rated base was too small to support any agreement estimate, and that no
α should have been quoted for it. The original figure was wrong in its value and, more
seriously, wrong in implying that a stable estimate existed.

For the current corpus the statistic is sound: **α = 0.754, 95% CI [0.630, 0.872], on 107
double-rated pairs** — computed with the corrected definition.

Separately, the same error inflated the raw and binarised agreement figures produced by
`scripts/iaa_report.py` on the current corpus: raw exact agreement 0.7529 → **0.7290**,
binarised (≥3 vs ≤1) 0.9494 → **0.9286**. 174 of the current 1,120 judgments are
self-repeats.

### Direction of the error

The error **understated** agreement, because self-repeats that differed were counted as
disagreement. A correction that improves our own numbers deserves more scrutiny than one
that worsens them, so the full chain is given for independent checking:

- `v19.21` — *latest-vote-per-person statistics* — fixed the statistics layer
  (`core/annotation_quality.py`, `core/science.py`), after the 19 July deposit
- `v19.36` — fixed `scripts/iaa_report.py`, which had continued to count rows
- both values are reproducible from the deposited file alone; see
  `journal_submission/draft/ERRATUM_corpus_v1.0.md`

### Consequence for the pre-registered release trigger

`CORPUS_RELEASE_CRITERIA.md` triggers a corpus release on "+150 double-rated pairs since
the last released version". The baseline recorded for v1.0 was 35; the correct value is
13. The trigger therefore fires at **163**, not 185.

This makes the next release fire **earlier** than the erroneous baseline would have. It
is flagged prominently because quietly adjusting a pre-registered threshold in the
direction that suits the authors is precisely what pre-registration exists to prevent.
The baseline correction is recorded in `CORPUS_RELEASE_CRITERIA.md` itself.

### How it was found

While instrumenting the annotation router (v19.34), the double-rated count reported by
`scripts/iaa_report.py` (174) was noticed to disagree with the count used by
`core/science.overlap_stats`, the release check and the Admin panel (107). The two
definitions were compared, and the row-counting one was found to be wrong.

### What changed so it cannot recur

- `krippendorff_alpha_ordinal` and `overlap_stats` deduplicate to the latest score per
  (pair, annotator), and say so in their docstrings.
- `scripts/iaa_report.py` reports `pairs_double_rated` and, alongside it,
  `repeat_judgments_superseded`, so the repeat volume is visible rather than folded in.
- Every judgment now records the routing strategy that produced it, so agreement can be
  recomputed with any sampling strategy removed.
- Manuscript figures are generated into LaTeX macros from the database
  (`journal_submission/draft/make_numbers.py`) rather than typed, so a number in the
  prose can always be re-derived.
