# Wisdom Extractor — Reproducibility & Triangulation (proverbs_clean.csv stratified sample)

- Data used: stratified sample of 2999 from 3673 total rows
- Cophenetic correlation: 0.805
- τ grid: 0.25, 0.30, 0.35, 0.40, 0.45
- Primary τ (stability+silhouette composite): **0.25**

## Sensitivity table
|   threshold_tau |   n_items_evaluated |   n_clusters |   silhouette_cosine |   cophenetic_corr |   bootstrap_ARI_mean |
|----------------:|--------------------:|-------------:|--------------------:|------------------:|---------------------:|
|            0.25 |                2999 |         2153 |            0.337661 |          0.805311 |             0.999491 |
|            0.3  |                2999 |         2137 |            0.341894 |          0.805311 |             0.998658 |
|            0.35 |                2999 |         2115 |            0.346534 |          0.805311 |             0.998104 |
|            0.4  |                2999 |         2088 |            0.349897 |          0.805311 |             0.995345 |
|            0.45 |                2999 |         2059 |            0.351708 |          0.805311 |             0.996086 |

## Top clusters by cultural coverage @ τ=0.25 (sample)
|   cluster |   distinct_cultures | category                       |
|----------:|--------------------:|:-------------------------------|
|      1954 |                   6 | culture-specific (operational) |
|      1275 |                   5 | culture-specific (operational) |
|       411 |                   5 | culture-specific (operational) |
|       473 |                   4 | culture-specific (operational) |
|      1299 |                   4 | culture-specific (operational) |
|       474 |                   4 | culture-specific (operational) |
|      1882 |                   4 | culture-specific (operational) |
|       598 |                   4 | culture-specific (operational) |
|      1140 |                   4 | culture-specific (operational) |
|      1771 |                   4 | culture-specific (operational) |

## Triangulation vs. random-mixing baseline (percentiles)
Higher percentiles ⇒ broader diffusion/universality beyond genealogical/areal proximity (PERMS=200).

|   cluster |   distinct_cultures |   distinct_families |   distinct_regions |   families_percentile_vs_random |   regions_percentile_vs_random |
|----------:|--------------------:|--------------------:|-------------------:|--------------------------------:|-------------------------------:|
|      1954 |                   6 |                   5 |                  5 |                           88    |                          88    |
|      1768 |                   3 |                   3 |                  3 |                           86.5  |                          86    |
|       597 |                   3 |                   3 |                  3 |                           84.25 |                          84.5  |
|       789 |                   3 |                   3 |                  3 |                           83.25 |                          84.5  |
|      1236 |                   3 |                   3 |                  3 |                           83.25 |                          84    |
|       618 |                   3 |                   3 |                  3 |                           83.25 |                          82    |
|      1180 |                   2 |                   2 |                  2 |                           71.5  |                          72.25 |
|       892 |                   2 |                   2 |                  2 |                           71.5  |                          71    |
|      1596 |                   2 |                   2 |                  2 |                           70.5  |                          70.75 |
|      1743 |                   2 |                   2 |                  2 |                           70.25 |                          69    |

## Artifacts
- sensitivity_metrics.csv
- silhouette_vs_tau.png
- coverage_by_cluster_sample.csv
- triangulation_vs_random_sample.csv
