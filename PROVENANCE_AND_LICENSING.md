# The Wisdom Extractor — origin of the software, and its licensing position

Vlad Belciug & Elena Pelican · 4 August 2026

A note prepared in response to two questions: how the world map was built, and
whether any part of the project depends on proprietary or restricted material.

---

## 1. The map: how it works

The map is not produced by a mapping library. There is no Folium, no Plotly, no
GeoPandas, no Leaflet, no Google or Mapbox component, and no JavaScript library of
any kind. It is a self-contained HTML document that our own Python code writes,
about 250 lines in `core/mapview.py`, and it has three parts.

**A base outline stored as plain SVG paths.** `data/world_map_paths.svg` holds the
continental coastlines as ordinary `<path>` coordinates in a 1000 × 520 canvas. It
is geometry in a text file, not a map service: there are no tiles to fetch, no API
key, no usage quota, and no network access at view time.

**A projection written out in one line.** Each culture has a latitude and longitude
in `data/people_coords.csv`, converted to canvas coordinates by

```python
x = (lon + 180) / 360 * W
y = (LAT_TOP - lat) / (LAT_TOP - LAT_BOT) * H      # W=1000, H=520, LAT_TOP=85, LAT_BOT=-60
```

This is the equirectangular (plate carrée) projection: longitude scaled linearly
across the width, latitude linearly down the height, truncated at 85° N and 60° S to
remove empty polar space. It distorts area at high latitudes, which is irrelevant
here because the map places labelled points and arcs rather than measuring surfaces.

**Data injected as JSON.** Python queries the database, assembles the clusters, the
arcs and the per-culture attestation years, and substitutes three placeholders into
an HTML template: the outline, a JSON data block, and the header. Plain JavaScript
in the page then draws the circles and arcs over the SVG and re-filters them when the
time slider moves. Inside the Streamlit application it is displayed with
`components.html`; on the server the same document is served as an ordinary page.

The design is deliberate rather than incidental. A library-based map would fetch
JavaScript from a third-party host when the page is opened, which would make the
figure depend on that host still existing, and would prevent the map from being
regenerated identically years later. The present approach keeps the whole pipeline
offline and reproducible, which is the methodological claim the paper makes about
every other stage. The cost is that panning, zooming and basemap styling do not
exist; for this purpose that is an acceptable trade.

---

## 2. Provenance of the outline

The world outline is **derived from Natural Earth**, the 1:110m "land" vector layer,
obtained from the `natural-earth-vector` repository maintained by Nathaniel Vaughn
Kelso and Tom Patterson.

Natural Earth is explicitly placed in the **public domain**. Its terms of use state
that no permission is needed to use it and that crediting the authors is unnecessary.
We credit it regardless, as a matter of scholarly practice.

The conversion is performed by our own script, `scripts/build_world_svg.py`, which
reads the Natural Earth GeoJSON, applies the projection above, simplifies sub-pixel
detail, and writes the SVG. The generating script, the source file
(`data/ne_110m_land.source.geojson`) and the output are all in the repository, so
any reader can reproduce the outline from the source and verify it independently.

A note on history, for completeness. An earlier version of this file, added on
17 July 2026, carried no embedded provenance information. Its origin could not be
confirmed from the file itself, and an undocumented asset is not an acceptable basis
for a published work. It has therefore been regenerated from Natural Earth so that
the chain from public-domain source to published figure is documented and verifiable.
The projection is unchanged, so all previously published figures remain accurate.

---

## 3. Is the software our own work?

**Yes.** Every line of the pipeline — the scraper, the cleaning filters, the
canonicalisation rules, the clustering, the annotation platform, the consensus and
reliability model, the statistics, the map generator — was written for this project.
No third-party source code is copied into the repository, and no code is vendored:
there are no bundled JavaScript libraries, no minified assets, and no embedded fonts.
The shipped web pages request no external scripts, stylesheets or fonts whatsoever;
they rely on fonts already present on the reader's system.

The software was developed by the authors with the assistance of an AI coding
assistant, used as a writing and implementation tool under our direction, review and
testing. This is disclosed wherever the project's outputs are reported. The
resulting work is our own in the sense that matters here: we specified it, we
directed it, we tested it, we are responsible for it, and no part of it reproduces a
third party's protected expression.

---

## 4. Dependencies, and their licences

The project builds on general-purpose scientific Python libraries, all of them free
and open-source under permissive licences that allow unrestricted academic and
commercial use, modification and redistribution:

| Component | Licence |
|---|---|
| pandas, NumPy, SciPy | BSD |
| scikit-learn | BSD-3-Clause |
| Streamlit | Apache-2.0 |
| FastAPI | MIT |
| Uvicorn | BSD-3-Clause |
| Requests | Apache-2.0 |
| BeautifulSoup4 | MIT |
| lxml | BSD-3-Clause |
| Pillow (image generation) | MIT-CMU |
| ReportLab (PDF generation) | BSD |

None is copyleft, none imposes conditions on our own code, none requires a licence
fee, and none is proprietary. There is no commercial software, no trial component,
no paid API, and no service subscription anywhere in the pipeline. The system runs
entirely on our own machines and on a rented virtual server.

---

## 5. The corpus, which is the one item with conditions attached

The software is unencumbered. The **data** carries obligations, and these deserve to
be stated plainly, because they affect anyone who reuses the corpus.

- Material collected from **Wikimedia projects** (Wikiquote, Wiktionary) is licensed
  **CC BY-SA**. This is a *share-alike* licence: derivative collections must be
  released under the same terms, with attribution.
- Material from **Project Gutenberg** and from **Internet Archive** scans of
  nineteenth-century collections is in the **public domain** by age.
- Consequently the published corpus is distributed as **CC BY-SA**, and this is
  stated on the deposit and on the project website. A partner intending to
  redistribute a derivative dataset must observe the share-alike condition. This is
  a condition, not an obstacle, and it is the ordinary situation for corpora built
  on Wikimedia sources.
- Human annotations are contributed by volunteers who are informed that their
  judgments become open research data. Annotators are identified in the database
  only by a random pseudonym; nicknames exist for a leaderboard and are never
  exported. No personal data is collected: there are no accounts, no email
  addresses, and no tracking.

---

## 6. Summary

The software is our own original work, built on permissively licensed open-source
libraries, and contains no proprietary, restricted or undocumented components. The
world outline derives from a public-domain source and is regenerated from that source
by a script included in the repository. The corpus is openly licensed, with a
share-alike obligation inherited from its Wikimedia components, and is deposited with
a DOI. The project can be reproduced, inspected, redistributed and built upon by
anyone, without seeking permission from us or from any third party.

Repository: https://github.com/ovladon/wisdom-extractor
Software archive: https://doi.org/10.5281/zenodo.21413838
Corpus: https://doi.org/10.5281/zenodo.21439285
