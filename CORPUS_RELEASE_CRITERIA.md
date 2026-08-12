# Corpus release criteria (pre-registered)

Stated in advance, before the numbers were known, so that dataset versions cannot be
accused of being timed to flatter the results.

## The rule that matters

**Releases are triggered by how much data exists, never by what the data shows.**

A version is cut when the corpus reaches a size milestone. It is *not* cut when
Krippendorff's alpha crosses a threshold, when an AUC looks good, or when a
correlation becomes significant. Publishing only when a statistic looks favourable
would make the version history a filtered view of the project, which is a form of
selection bias — the same family of error as reporting only successful experiments.

Consequence, accepted deliberately: **a scheduled version is published even when its
statistics are unflattering.** If agreement has fallen, that version is released with
the fall visible in it.

## Triggers (any one is sufficient)

| # | Condition | Rationale |
|---|-----------|-----------|
| 1 | +25% proverbs since the last released version | Material change in corpus scope |
| 2 | +100% judgments since the last released version | Material change in evidence base |
| 3 | +150 double-rated pairs since the last released version | The basis of every agreement statistic |
| 4 | A corpus-wide cleaning pass changes >1% of records | Data quality differs materially from the published copy |
| 5 | Twelve months since the last release | Guards against silent staleness |

## What is *not* a trigger

- Any value of alpha, AUC, rho, or p
- Reviewer requests to re-run an analysis (that cites an existing frozen version)
- Adding features to the software (the software has its own DOI, archived per release)

## Corrections to a recorded baseline

A trigger is computed against the figures recorded for the last released version. If one
of those figures is later found to be wrong, the baseline is corrected and the correction
is recorded in `CORRECTIONS.md` and here — never adjusted silently. Quietly moving a
pre-registered threshold is the failure mode this document exists to prevent, and it is
no less a failure when the movement is the result of an honest error.

**2026-08-12.** The v1.0 baseline `double_rated` was recorded as 35. It was computed over
pairs carrying two or more judgment *rows* rather than two or more *distinct annotators*,
and the correct value is **13**. Trigger 3 therefore fires at 163 double-rated pairs, not
185 — that is, **earlier** than the erroneous baseline would have allowed. The recorded
`alpha` for v1.0 is withdrawn rather than corrected: at 13 units no agreement estimate is
defensible. See `CORRECTIONS.md`.

## Human gate

The pipeline **detects** readiness and **prepares** the export. It never publishes.
Publication to Zenodo is permanent and public, so a person confirms it after checking
the export. Automating an irreversible public action on a threshold would mean that a
data bug, an import error, or a vandalised batch could publish itself.

## Citation policy

A paper cites the **exact frozen version DOI it analysed**, never the concept DOI.
Reproducibility requires the reader to obtain precisely the data behind the numbers,
not whatever is newest. The concept DOI is for people who want the current corpus.

Every released version records: proverb count, peoples, judgments, double-rated pairs,
annotator count, and the statistics as they stood — including when they are worse than
the previous version.
