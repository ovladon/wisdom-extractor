# Changelog — The Wisdom Extractor

All notable transformations of this project. Versions before v19 predate this git
repository and are imported as dated archive commits (tags `v7.7`–`v18`); their
snapshots were recovered from the project's version folders and `v8-18.zip`.

## v19.25 — 2026-08-04
- **Hardened the adult-language report against abuse.** It was the least-guarded
  endpoint: IP rate-limit only, no per-user cap, no record of who reported what.
  Now it requires a claimed nickname, caps at 20 hides per annotator per day, and
  writes an attributable row for every report.
- **One-command revert**: Admin tab lists who has hidden how much and reverts an
  account's hides wholesale. Proverbs the word list catches, or that a second
  annotator also reported, stay hidden.

## v19.24 — 2026-08-04
- **Adult language hidden from public surfaces.** New `sensitive` flag: proverbs
  with unambiguous adult vocabulary stay in the corpus (removing them would censor
  the scientific record and these are real folk sayings) but are never served to
  annotators and never drawn on the public map. 106 flagged on the live corpus,
  0.3%, mostly earthy Finnish folk material.
- The word list is deliberately narrow: "ass" (donkey), "cock" (rooster) and
  "bitch" (female dog) are ordinary in historical collections and are NOT matched.
  A "🔞 Adult language" report button in the app catches what the list misses,
  hiding both proverbs of the pair immediately.
- Auto-flagging runs in every maintenance cycle, so new ingests are covered.

## v19.23 — 2026-08-03
- **Sharing, so the game can spread.** Standing screen gains an invite block:
  native share sheet on mobile, WhatsApp, LinkedIn, copy-link — the message is
  personalised with the sharer's judgment count.
- **Access code travels in the link** (`/?code=...`), stored on arrival, with a
  green confirmation on the setup screen. Invitees no longer need to be told a
  code out of band; they only pick a nickname.
- Landing: Open Graph/Twitter preview card (og-image.png) so pasted links render
  as a titled card; one-tap share row; Play button and /play redirect now carry
  the code.

## v19.22 — 2026-07-23
- **Live-first analysis app**: app.py now auto-syncs a snapshot of the server
  database on launch (10-min freshness window, manual refresh button, graceful
  offline fallback) — local numbers always reflect reality.
- **Workspaces**: separate databases for new/private proverb collections (sidebar
  create/select); scripts/merge_workspace.sh merges a workspace into the live corpus
  (hash-deduped) and runs full maintenance.
- **Admin tab**: corpus + annotation metrics (α, consensus, reliability table with
  nicknames, duplicate reports) and on-demand AUC computation in one place.
- **scripts/sensitivity.py**: automated τ-sweep scored against human consensus;
  first run shows τ=0.35 is precision-heavy (99.7% cannot-split, 27.7% must-linked;
  τ=0.45–0.50 reaches 0.82–0.84 constraint accuracy) — revision-paper evidence.

## v19.21 — 2026-07-22
- **Nickname-device binding** (link now spreading publicly): the browser silently
  generates a device key; the first device to judge under a nickname claims it
  (sha256 of key stored in annotators.key_hash). Another device using the same
  nickname gets a clear 409 and a prompt to pick a different name. Legacy clients
  without a key still work and claim on next reload; existing annotators are
  grandfathered on first return. Zero added friction — no accounts, no passwords.
- **IAA hardening**: consensus and Krippendorff's alpha now keep only each
  annotator's LATEST vote per pair — self-repeats can no longer masquerade as
  inter-annotator agreement or multiply one person's voice.

## v19.20 — 2026-07-22
- Navigation back to wisdomextractor.com from both public surfaces: 🏠 link in the
  map's pubbar and in the annotation app (footer + setup panel).
- New public /api/pubstats endpoint (1h cache, CORS-open, rate-limited) feeding live
  counters on the landing page.
- Landing: verified cross-cultural convergence showcase ("In wine there is truth"
  Chinese=Greek; stitch/steek French=Scottish 1870; Slovak=Hungarian fate), "Built in
  the open" strip (ConsILR method, Zenodo DOIs, GitHub), 1663 Stampoy spotlight,
  live counters with static fallback; numbers refreshed (32,000+/79).

## v19.19 — 2026-07-22
- **Corpus self-cleaning** (annotator-reported problem: OCR garbles like "ivants"
  and abundant near-duplicates). Three new pipeline steps each maintenance cycle:
  fix_ocr_artifacts() repairs the w->'iv' OCR confusion using the corpus as its own
  dictionary (rare word + common 'w' variant required); dedup_normalized() auto-
  excludes same-people twins identical after case/punctuation normalization (keeper
  inherits earliest attestation; cross-people twins preserved as data);
  apply_corrections() applies annotator-suggested fixes when typo-sized (>=0.85
  similarity), larger rewrites stay pending. Fixed rows get claim/gloss cleared for
  re-derivation; hash collisions resolve as exclusions.
- **"Fix A/B's spelling" buttons** in the app + /api/fix endpoint (code+human+rate
  guarded, fix must stay >=0.5 similar to the original); guide explains judging
  garbled-but-readable sayings by their intended meaning.

## v19.18 — 2026-07-21
- **Duplicate reports now feed the pipeline**: each maintenance cycle merges
  human-reported exact duplicates as attestations of one saying (keeper inherits the
  earliest first_seen year; the twin is excluded). Guards: cross-people twins are
  never merged (that's cross-cultural evidence), single-reporter merges require
  near-identical text (difflib >= 0.85), two+ reporters always merge. Idempotent;
  digest reports duplicates_merged.

## v19.17 — 2026-07-21
- **"Exact duplicate" option** at the top of the Related… levels: annotators can flag
  word-for-word twins (common across overlapping collections). Stored as a maximal
  same-rule judgment (4) for clustering PLUS a row in the new `duplicate_reports`
  table (pair-normalized, unique per user) — a human-verified dedup/attestation-merge
  list that leaves the -1..4 IAA scale untouched. Guide entry added; stats() now
  reports duplicate_reports.

## v19.16 — 2026-07-21
- **Fix: the game screen truly hides now.** An inline display:flex on the game panel
  overrode the .hidden class since v19.5, so the empty proverb card and buttons stayed
  visible above the human-verification step. Hidden is now !important and the flex
  layout lives in CSS: until verification passes, the check is the only thing on screen.

## v19.15 — 2026-07-21
- **Human check made legible**: full "Step 2 of 2" welcome card explaining WHY the
  proverbs haven't loaded yet (bot protection, one-time, device trusted a week), visible
  loading/failure states, always-available "get a different proverb" retry, Enter-to-
  submit, autofocus; mid-game re-checks announce themselves with a toast; and a failed
  pair fetch now shows tap-to-retry instead of a silent "…".

## v19.14 — 2026-07-21
- **Public living map**: `/map` on the annotate service — no access code, regenerated
  from the database every 6 hours, with a freshness header (sayings/peoples/judgments/
  updated date) and a contribution CTA; embeddable on project domains (CSP
  frame-ancestors). Netlify `_redirects` proxies it to wisdomextractor.com/map and
  /play to the game.
- **Landing page rebuilt** around the map-and-contribution story: dual CTAs, the
  "why your thirty seconds matter" narrative, live-ish corpus numbers, 1663/1562
  time-depth hooks.

## v19.13 — 2026-07-21
- **Nicknames are display-only.** New `annotators` table maps each nickname to a random
  pseudonym (`u_xxxxxxxx`); all judgments, reliability computations, consensus, exports,
  and analysis pipelines see only the pseudonym. Nicknames exist solely to render the
  leaderboard and never leave the server. Existing databases migrate automatically and
  idempotently on start. Erasing someone's nickname is a one-row deletion that leaves
  the science untouched.

## v19.12 — 2026-07-21
- **Privacy**: /privacy notice (controller, data, purpose, anonymized publication,
  rights, netcup DPA); consent line + privacy link on the annotator setup screen and
  landing footer; name field now nudges toward nicknames.

## v19.11 — 2026-07-21
- **Ordinal consensus engine**: aggregation now works natively on the Pelican scale —
  reliability-weighted consensus scores, ordinal annotator reliability (closeness, not
  binary match), theme-zone votes finally count, disputes from dispersion. Synthetic
  validation improves further (2-inverters scenario 0.54 → 0.86 accuracy). Live data:
  204 consensus pairs now yield 45 must + 140 cannot constraints (was 15/35).
- **Human check** ("finish the proverb"): judging requires a one-time human verification
  with a rotating classic-proverb completion; 7-day tokens; friendly modal.
- **Hardened onboarding**: /api/config drives required-field gating (name ≥2 chars,
  access code when configured) with clear messages; graceful 401/403/429 handling;
  wrong-code brute-force limiter (20/h/client); proxy-aware rate-limit keys
  (X-Forwarded-For) so per-user limits work correctly behind Caddy.
- **Landing page** (`landing/`) for the public domain root.

## v19.10 — 2026-07-21
- **Stratified pair serving** (Pelican revision): candidate pairs are bucketed by
  language-family match x region match x similarity band and served round-robin, so the
  growing annotation set is balanced across strata.
- **Log-odds consensus** (Dawid-Skene weighting): identified low-reliability annotators'
  votes now count against their choice; synthetic adversarial validation shows +0.18
  consensus-accuracy gain with an inverter present and clean honest/adversary
  reliability separation (`scripts/reliability_validation.py`).
- **Revision instruments**: `scripts/iaa_report.py` (raw/binarized agreement, ordinal
  Krippendorff alpha, weighted Cohen kappa, disagreement-examples table by boundary
  category), `scripts/baseline_embeddings.py` (char n-gram vs multilingual
  sentence-embedding AUC on annotated pairs), `scripts/audit_samples.py`
  (canonicalisation & cluster-quality human-audit CSV generators + summarizer).
- requirements: sentence-transformers/torch optional (baseline script only).

## v19.9 — 2026-07-19
- **API hardening**: per-client rate limiting (90 req/min), daily cap on
  "not a saying" exclusions (30/user/day), access code accepted via `X-Access-Code`
  header (kept out of URL logs; query param still accepted for compatibility),
  and standard security headers (nosniff, no-referrer, frame deny).
- Frontend sends the code as a header.

## v19.8 — 2026-07-19
- **Dedicated-phone hosting kit** (`deploy/android/`): run the annotation service on a
  spare Android phone (no root; Termux + proot Ubuntu) — `setup_phone.sh` one-command
  install, `start_phone.sh` (server + tunnel + daily backup loop + weekly maintenance
  loop), Termux:Boot recipe for reboot self-healing, and an optional permanent link via
  a Netlify redirect-site auto-updated on every tunnel restart
  (`update_netlify_redirect.sh`). Guide: `deploy/android/PHONE_SERVER.md`.
- `requirements-server.txt`: server-only dependency subset (no Streamlit) for ARM/proot.

## v19.7 — 2026-07-18
- **Self-maintenance** (`scripts/maintain.py`): one cron-able command that scrapes the
  next N catalog sources (round-robin cursor), backfills culture/family/years/glosses,
  canonicalizes new rows, aggregates all annotations into consensus constraints, and
  reclusters the corpus — new sayings become annotatable automatically, and accumulated
  human judgments reshape the clusters on every run. Auto-loads the sources catalog
  into DBs that lack it.
- `deploy/backup.sh` (nightly DB backups, 30-day retention) + cron recipes in DEPLOY.md.
- Noise filter: web-nav/ad debris patterns (scraped-page ads no longer enter the corpus).

## v19.6 — 2026-07-18
- **Graded semantic-equivalence scheme (Dr. Elena Pelican).** Binary same/different is
  replaced everywhere by her 6-level scale — 4 same rule, 3 same advice, 2 same theme,
  1 related/different lesson, 0 unrelated, −1 contradictory — with her operational
  annotator tests as the in-app guide. Mobile uses her two-stage flow: swipe left =
  unrelated; swipe right = related → pick the relation. Scores are stored raw;
  hard clustering constraints derive as ≥3 → must-link, ≤1 → cannot-link, 2 → recorded.
  Legacy binary annotations remain valid.
- **Inter-annotator agreement**: ordinal Krippendorff's α over multi-annotated pairs,
  shown in Diagnostics (with legacy-binary mapping so mixed data stays measurable).
- **English glosses** (`core/gloss.py`): annotators are never shown a pair they cannot
  read — an English gloss is extracted from markers ("Translation:", "Literally:"),
  bilingual texts, or English originals (60.9% of corpus, 11,093 items); unglossed rows
  stay in the corpus but out of annotation pools. Gloss shown big, original beneath.
- Review-response plan: `docs/pelican_review_plan.md`.

## v19.5 — 2026-07-18
- **Mobile swipe app** (`mobile/`): phone-first PWA for annotation in spare moments —
  swipe right = same idea, left = different; tap fallbacks; streak counter with haptics;
  leaderboard, rank and consistency score; "not a saying" reporting; installable to the
  home screen (manifest); light/dark theme. Served by a four-endpoint FastAPI backend
  (`mobile/mobile_api.py`) over the same SQLite database and constraints table the
  reliability model and clustering already use; same `ANNOTATOR_CODE` gate; candidate
  pairs from the uncertain-zone kNN pool with periodic refresh, disputed pairs mixed in.
- Deployment: second service + subdomain in `deploy/docker-compose.yml` / `Caddyfile`;
  requirements gain fastapi/uvicorn.

## v19.4 — 2026-07-18
- **Annotator portal** (`annotator_app.py`): a slim, safe entry point for annotators —
  pair-judgment game (uncertain / disputed / random strategies), live leaderboard,
  personal consistency score, world map. No admin tabs, no destructive actions,
  optional `ANNOTATOR_CODE` access gate. Deploy this for annotators; `app.py` stays
  the full lab bench on the same database.
- **Deployment kit** (`Dockerfile`, `deploy/`): docker-compose with automatic HTTPS
  (Caddy), persistent volume for the database, and `deploy/DEPLOY.md` comparing VPS /
  Hugging Face Spaces / Streamlit Cloud hosting (Netlify cannot run Streamlit).
- One-click Windows launcher (`run_windows.bat`) + annotator guide (`WINDOWS.md`)
  for the install-locally fallback.

## v19.3 — 2026-07-17
- **World map integrated into the app** (`core/mapview.py`, Results tab): motif view
  (attestation points + arcs, per-culture proverb tooltips), shared-wisdom network
  (120 strongest culture pairs), Europe zoom, and a **time slider** — "attested by
  year N" — driven by the first_seen bounds, with undated attestations shown faded.
  Fully self-contained HTML rendered via components.html; no external requests.
- New data assets: `data/world_map_paths.svg` (compact public-domain world outline),
  `data/people_coords.csv` (67 cultural centroids, extendable).

## v19.2 — 2026-07-17
- **Attestation years / historical timeline.** `extract_attestation_year()` harvests
  "attested no later than" years from citation tails before stripping;
  `backfill_attestation_years()` dates existing rows from citations or dated sources
  (`data/source_years.json`). On the current corpus: 5,738 rows dated (31%), earliest
  bounds 1611–1758. Scrape/seed capture years automatically; tab 2 gains a 🕰️ backfill
  button; tab 5 shows a per-cluster attestation timeline (year × culture).
- New boilerplate rules drop source-book headers ("Nathan Bailey (1721). …") that
  previously survived as fake proverbs.
- Third-party warning noise (TensorFlow plugin registration, sklearn FutureWarning,
  numba/TBB) suppressed at app start.

## v19.1 — 2026-07-17
- **Annotation confidence hierarchy** (`core/annotation_quality.py`): reliability-weighted
  consensus per annotated pair (Dawid–Skene-lite; annotator reliability learned from
  agreement with consensus, prior 0.7). Clustering enforces only consensus constraints
  above a confidence slider; tied/low-confidence pairs are excluded and queued.
- Annotate tab: new "Verify disputed / low-confidence" strategy serving the pairs whose
  confidence benefits most from another vote.
- Diagnostics: annotation consensus stats, per-annotator reliability table, review queue.
- Fix: empty-database crash in Diagnostics (`cached_proverbs` now returns explicit columns).

## v19.0 — 2026-07-16 — the unification
Merged the two development lineages and fixed the defects of both.
- From the paper pipeline (v10): canonicalization cascade, lexical normalization,
  char 3–5-gram TF-IDF, agglomerative clustering (τ cut), wisdom score
  (coverage + 0.3·support), 2D projection, τ-sweep/silhouette/bootstrap-ARI diagnostics,
  permutation triangulation, themed interpretation, ecological correlations,
  21,378-row culture-labelled seed dataset.
- From Wisdom Lab (v18): depth-1 concurrent scraper (robots.txt, MediaWiki API
  fallbacks), SQLite persistence with WAL + hash dedup, annotation game with batch
  writes and leaderboard.
- New in the merge: **constraint-aware clustering** (must/cannot-links enforced via
  constrained union-find); **constraint-agreement evaluation** (annotations double as
  evaluation set); **scalable sparse clustering** (radius-neighbour graph; 18k rows in
  ~14 s vs the 2.7 GB dense matrix that v18 required); culture inference from URLs +
  backfill + family/region enrichment; citation stripping; in-place schema migration
  for v15–v18 databases; empty-catalog fix (v18 shipped a 19-byte sources_catalog.json).

## v18 — 2025-10-08 (archive)
Wisdom Lab "fast annotate": batch/instant write modes, autosave, cached pairs, WAL
tuning, richer scraper (stop button, uncapped depth-1). Analysis layer unchanged since
v15; culture labels still not captured (defect fixed in v19).

## v16 — 2025-10-01 (archive)
Wisdom Lab: scraper gains MediaWiki API fallback and XML parser auto-detect; DB exports.

## v15 — 2025-10-01 (archive)
First "Wisdom Lab" pivot: Streamlit app with SQLite persistence, sources catalog,
basic scraper, TF-IDF paraphrase graph + connected components, annotation game
(must/cannot/not-a-saying), survival score, leaderboard. Analysis simpler than the
paper lineage (word n-grams, no canonicalization, no diagnostics).

## v10 — 2025-09-30 (archive)
Final paper-lineage version; produced the numbers in the ConsILR-2025 submission
(τ sensitivity table, cophenetic 0.805, bootstrap ARI, permutation triangulation,
canonicalization/back-translation audit templates). Enhanced canonicalization
(~20 rule families), lexical normalization, adaptive thresholds, trust-score
diagnostics, simple validation tab.

## v7.7 — 2025-08-21 (archive)
The version described in the ConsILR-2025 paper text: five-tab Streamlit pipeline
(Dataset Builder, Wisdom Extractor, Results & Visualisation, Interpretation,
Diagnostics), regex canonicalization, char n-gram clustering, deterministic + optional
local-LLM interpretation.

## Earlier (v1–v6, v8–v9, v11–v14, v17) — not imported
Intermediate experiments; folders exist in the project archive but are not part of
this repository's history.
