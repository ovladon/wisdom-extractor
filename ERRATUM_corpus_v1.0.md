# Erratum to corpus v1.0 — agreement statistics for the graded annotation stream

**Applies to:** The Wisdom Extractor Corpus v1.0, DOI
[10.5281/zenodo.21439286](https://doi.org/10.5281/zenodo.21439286) (version),
concept DOI [10.5281/zenodo.21439285](https://doi.org/10.5281/zenodo.21439285),
published 19 July 2026.

**Affects:** the `README.md` description of `annotations_graded_platform.csv`.
**Does not affect:** `annotations.csv` (the 115-pair evaluation set analysed in the
companion papers), `proverbs.csv`, `clusters.csv`, or `people_metadata.csv`. No data file
is changed by this erratum; only a statistic quoted about one of them.

## What was stated

> Ordinal Krippendorff's alpha over the 32 multi-annotated pairs at time of deposit: 0.751.

## What is correct

| | published | correct |
|---|---|---|
| pairs with ≥2 **distinct** annotators | 32 | **13** |
| ordinal Krippendorff's α | 0.751 | **0.968** |
| 95% CI on α (bootstrap over units) | not stated | **[−0.09, 1.00]** |

## Cause

The implementation counted judgment **rows** per pair rather than distinct annotators. Of
the 237 judgments in the deposited file, **20 are an annotator re-judging a pair they had
already judged**. Those repeats were treated as independent opinions, so annotators were
partly being compared with themselves. Where a repeat differed from the annotator's own
earlier score, it entered the estimate as inter-annotator disagreement.

Reproduce both values from the deposited file alone:

```bash
python - <<'PY'
import csv
from core.annotation_quality import krippendorff_alpha_ordinal
rows = sorted(csv.DictReader(open("annotations_graded_platform.csv")), key=lambda r: r["utc_time"])
cons = [{"a_id": int(r["a_id"]), "b_id": int(r["b_id"]), "score": int(r["score"]),
         "user": r["annotator"], "label": r["derived_label"]} for r in rows]
print(krippendorff_alpha_ordinal(cons))   # -> (0.9678..., 13)
PY
```

## The correct conclusion is not the corrected number

At 13 units the confidence interval spans essentially the whole admissible range. The
right statement is **not** that agreement was higher than reported; it is that the
deposit's double-rated base was too small to support any agreement estimate, and no α
should have been quoted for it. The published figure was wrong in its value and, more
importantly, wrong in implying a stable estimate existed.

## Direction of the error

The error **understated** agreement. A correction that improves our own numbers warrants
more scrutiny than one that worsens them, so the full chain is given here:

- fix to the statistics layer: `v19.21` — *latest-vote-per-person statistics*
- fix to the reporting script: `v19.36` — `scripts/iaa_report.py` counted rows, not annotators
- both reproducible from the deposited file with the snippet above

## Consequential correction to the release trigger

The same row-counting error inflated the double-rated count in
`data/last_corpus_release.json` (recorded 35; correct value 13). That figure is the
baseline for the pre-registered "+150 double-rated pairs" corpus release trigger. The
baseline is corrected to 13, which makes the next release fire **earlier** than the
erroneous baseline would have. This is flagged explicitly because quietly adjusting a
pre-registered trigger in the direction that suits the authors is exactly what
pre-registration exists to prevent.

## Action taken

Corpus **v1.1** is issued with a corrected `README.md` and this erratum included. v1.0
remains available and citable, as the version history is itself part of the record;
Zenodo links the versions, so a reader arriving at v1.0 reaches this correction.
