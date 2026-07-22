"""In-app world map of proverb motifs (v19.3).

Generates a self-contained HTML document from the live database: attestation points,
motif arcs, a shared-wisdom network, and a temporal slider driven by the
"attested no later than" years (first_seen). Rendered in Streamlit via components.html.

Honest semantics, printed on the map itself: arcs are co-occurrence in the corpus,
not documented transmission paths; the time slider filters by earliest *dated* source,
an upper bound on first attestation.
"""
import csv
import json
import os
from collections import defaultdict

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

W, H, LAT_TOP, LAT_BOT = 1000, 520, 85.0, -60.0

FAMILY_GROUP = {
    "Slavic": "Slavic", "Germanic": "Germanic", "Romance": "Romance", "Uralic": "Uralic",
    "Celtic": "Other Indo-European", "Baltic": "Other Indo-European",
    "Indo-Iranian": "Other Indo-European", "Hellenic": "Other Indo-European",
}


def _xy(lat, lon):
    return round((lon + 180) / 360 * W, 1), round((LAT_TOP - lat) / (LAT_TOP - LAT_BOT) * H, 1)


def _load_coords():
    coords = {}
    with open(os.path.join(DATA_DIR, "people_coords.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            coords[row["people"]] = _xy(float(row["lat"]), float(row["lon"]))
    return coords


def _group_of(family):
    fam = str(family or "")
    for key, grp in FAMILY_GROUP.items():
        if key.lower() in fam.lower():
            return grp
    if fam and "indo-european" in fam.lower():
        return "Other Indo-European"
    return "Other / mixed" if not fam else "Asian & Pacific"


def build_map_data(df, min_coverage=4, max_clusters=120, max_edges=120):
    """df: proverbs frame with people, claim/text, cluster_id, first_seen, family."""
    coords = _load_coords()
    d = df.dropna(subset=["cluster_id", "people"]).copy()
    d = d[d["people"].isin(coords)]
    textcol = "claim" if d["claim"].notna().any() else "text"

    cov = d.groupby("cluster_id")["people"].nunique().sort_values(ascending=False)
    clusters = []
    for cid in cov[cov >= min_coverage].index[:max_clusters]:
        sub = d[d["cluster_id"] == cid]
        claims = sub[textcol].dropna().astype(str)
        claim = claims.mode().iloc[0] if len(claims) else ""
        examples, years = {}, {}
        for p, g in sub.groupby("people"):
            examples[p] = str(g["text"].iloc[0])[:160]
            ys = g["first_seen"].dropna()
            if len(ys):
                years[p] = int(ys.min())
        if len(examples) >= min_coverage:
            clusters.append({"claim": claim[:110], "coverage": len(examples),
                            "support": int(len(sub)), "examples": examples, "years": years})
    clusters.sort(key=lambda c: -c["coverage"])

    pair_w = defaultdict(int)
    cov2 = d.groupby("cluster_id")["people"].nunique()
    for cid in cov2[cov2 >= 2].index:
        ps = sorted(d[d["cluster_id"] == cid]["people"].unique())
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                pair_w[(ps[i], ps[j])] += 1
    edges = sorted(pair_w.items(), key=lambda kv: -kv[1])[:max_edges]

    fam_by_people = {}
    for p, g in d.groupby("people"):
        fams = g["family"].dropna()
        fam_by_people[p] = str(fams.iloc[0]) if len(fams) else ""

    return {
        "W": W, "H": H,
        "coords": {p: list(coords[p]) for p in d["people"].unique() if p in coords},
        "family": fam_by_people,
        "group": {p: _group_of(fam_by_people.get(p)) for p in d["people"].unique()},
        "clusters": clusters,
        "edges": [[a, b, w] for (a, b), w in edges],
    }


def build_map_html(df, min_coverage=4, meta=None):
    """meta: optional dict {proverbs, peoples, judgments, generated} for the public header."""
    data = build_map_data(df, min_coverage=min_coverage)
    land = open(os.path.join(DATA_DIR, "world_map_paths.svg"), encoding="utf-8").read()
    m = meta or {}
    import datetime
    header = ""
    if m:
        header = (
            '<div class="pubbar">'
            '<div><b>The living map of human wisdom</b> — '
            f'{m.get("proverbs", 0):,} sayings · {m.get("peoples", 0)} peoples · '
            f'shaped by {m.get("judgments", 0):,} human judgments · '
            f'updated {m.get("generated", datetime.date.today().isoformat())}</div>'
            '<a class="home" href="https://wisdomextractor.com">🏠 wisdomextractor.com</a>'
            '<a href="https://annotate.wisdomextractor.com">🧭 Add your judgment — it redraws this map</a>'
            "</div>")
    return (_TEMPLATE.replace("__SVGMAP__", land)
                     .replace("__DATA__", json.dumps(data, ensure_ascii=False))
                     .replace("__PUBBAR__", header))


_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><style>
:root{--bg:#f7f6f2;--surface:#fdfdfb;--land:#e6e3da;--land-stroke:#d0ccc0;--ink:#171613;
--ink-2:#5a5850;--ink-3:#8a877c;--line:#dedbd2;--accent:#2a78d6;
--c-slavic:#2a78d6;--c-germanic:#1baf7a;--c-romance:#eda100;--c-uralic:#008300;
--c-oie:#4a3aa7;--c-asia:#e34948;--c-other:#8a877c;--arc:#171613;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 "Seravek","Ubuntu",Calibri,"DejaVu Sans",sans-serif;padding:8px}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 10px}
.pubbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:space-between;
  background:var(--surface);border:1px solid var(--line);border-radius:8px;
  padding:10px 14px;margin:0 0 12px;font-size:13.5px;color:var(--ink-2)}
.pubbar b{color:var(--ink)}
.pubbar a{background:var(--accent);color:#fff;text-decoration:none;font-weight:650;
  padding:8px 14px;border-radius:8px;white-space:nowrap}
.pubbar a.home{background:transparent;color:var(--accent);font-weight:600;padding:8px 4px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.seg button{border:0;background:var(--surface);color:var(--ink-2);padding:6px 12px;cursor:pointer;font:inherit;font-size:13px}
.seg button[aria-pressed="true"]{background:var(--accent);color:#fff}
select{font:inherit;font-size:13px;color:var(--ink);background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:6px 9px;max-width:min(480px,60vw)}
.timebox{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--ink-2)}
.timebox input[type=range]{width:180px}
.mapbox{position:relative;border:1px solid var(--line);border-radius:8px;background:var(--surface);overflow:hidden}
svg{display:block;width:100%;height:auto}
.land path{fill:var(--land);stroke:var(--land-stroke);stroke-width:.5}
.arc{fill:none;stroke:var(--arc);opacity:.16}
.arc.net{stroke:var(--accent);opacity:.35}
.pt{stroke:var(--surface);stroke-width:1.5;cursor:pointer}
.pt.undated{fill-opacity:.35}
.tooltip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--line);border-radius:6px;
padding:7px 10px;font-size:12.5px;max-width:290px;box-shadow:0 4px 14px rgba(0,0,0,.15);opacity:0;transition:opacity .12s}
@media (prefers-reduced-motion:reduce){.tooltip{transition:none}}
.tooltip b{display:block}.tooltip .fam{color:var(--ink-3);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.tooltip .txt{color:var(--ink-2);margin-top:3px;font-style:italic}
.legend{display:flex;flex-wrap:wrap;gap:5px 14px;margin:8px 2px;font-size:12.5px;color:var(--ink-2)}
.legend span{display:inline-flex;align-items:center;gap:5px}
.legend i{width:9px;height:9px;border-radius:50%;display:inline-block}
.panel{margin-top:12px;border:1px solid var(--line);border-radius:8px;background:var(--surface);padding:12px 14px;font-size:13px}
.panel h2{margin:0 0 2px;font-size:16px}
.meta{color:var(--ink-3);font-size:11.5px;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px}
.exlist{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:6px 18px;margin:0;padding:0;list-style:none}
.exlist li{color:var(--ink-2);border-top:1px solid var(--line);padding:6px 0 2px}
.exlist b{color:var(--ink)}
.note{color:var(--ink-3);font-size:12px;margin-top:8px}
</style></head><body>
__PUBBAR__
<div class="controls">
  <div class="seg"><button id="bMotif" aria-pressed="true">Motif view</button><button id="bNet" aria-pressed="false">Network</button></div>
  <div class="seg"><button id="bWorld" aria-pressed="true">World</button><button id="bEur" aria-pressed="false">Europe</button></div>
  <select id="motifSel" aria-label="Choose a motif"></select>
  <div class="timebox" id="timebox">
    <label>Attested by <b id="yearLabel">2026</b></label>
    <input type="range" id="yearSlider" min="1600" max="2026" value="2026" step="1">
    <label><input type="checkbox" id="showUndated" checked> include undated</label>
  </div>
</div>
<div class="mapbox">
  <svg id="map" viewBox="0 0 1000 520"><g class="land">__SVGMAP__</g><g id="arcs"></g><g id="pts"></g></svg>
  <div class="tooltip" id="tip"></div>
</div>
<div class="legend" id="legend"></div>
<div class="panel" id="panel"></div>
<div class="note">Lines are co-occurrence of a motif in the corpus, not documented transmission.
The year slider filters by earliest <i>dated</i> source (an upper bound on first attestation);
faded points are attestations without a dated source yet.</div>
<script>
const D=__DATA__;
const GC={"Slavic":"var(--c-slavic)","Germanic":"var(--c-germanic)","Romance":"var(--c-romance)",
"Uralic":"var(--c-uralic)","Other Indo-European":"var(--c-oie)","Asian & Pacific":"var(--c-asia)","Other / mixed":"var(--c-other)"};
const NS="http://www.w3.org/2000/svg";
const arcs=document.getElementById("arcs"),pts=document.getElementById("pts"),tip=document.getElementById("tip"),
panel=document.getElementById("panel"),sel=document.getElementById("motifSel"),map=document.getElementById("map"),
slider=document.getElementById("yearSlider"),yearLabel=document.getElementById("yearLabel"),
showUndated=document.getElementById("showUndated"),timebox=document.getElementById("timebox");
let mode="motif",idx=0;
D.clusters.forEach((c,i)=>{const o=document.createElement("option");o.value=i;
o.textContent=`(${c.coverage}) ${c.claim}`;sel.appendChild(o);});
const lg=document.getElementById("legend");
Object.entries(GC).forEach(([g,c])=>{const s=document.createElement("span");
s.innerHTML=`<i style="background:${c}"></i>${g}`;lg.appendChild(s);});
function el(n,a){const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);return e;}
function arcPath(a,b){const[x1,y1]=a,[x2,y2]=b,mx=(x1+x2)/2,my=(y1+y2)/2,dx=x2-x1,dy=y2-y1,d=Math.hypot(dx,dy);
return`M${x1} ${y1}Q${mx-dy*.18} ${my+dx*.18-d*.06} ${x2} ${y2}`;}
function showTip(ev,html){tip.innerHTML=html;tip.style.opacity=1;
const r=map.getBoundingClientRect(),b=tip.getBoundingClientRect();
let x=ev.clientX-r.left+14,y=ev.clientY-r.top-10;
if(x+b.width>r.width-8)x-=b.width+24;if(y+b.height>r.height-8)y=r.height-b.height-8;
tip.style.left=x+"px";tip.style.top=Math.max(4,y)+"px";}
function hideTip(){tip.style.opacity=0;}
function renderMotif(){
  const c=D.clusters[idx],yMax=+slider.value,withUndated=showUndated.checked;
  const active=Object.keys(c.examples).filter(p=>{
    const y=c.years[p];return y!==undefined?y<=yMax:withUndated;});
  arcs.innerHTML="";pts.innerHTML="";
  for(let i=0;i<active.length;i++)for(let j=i+1;j<active.length;j++){
    const a=D.coords[active[i]],b=D.coords[active[j]];
    if(a&&b)arcs.appendChild(el("path",{d:arcPath(a,b),class:"arc","stroke-width":1}));}
  active.forEach(p=>{const xy=D.coords[p];if(!xy)return;
    const dated=c.years[p]!==undefined;
    const dot=el("circle",{cx:xy[0],cy:xy[1],r:5,class:"pt"+(dated?"":" undated"),fill:GC[D.group[p]||"Other / mixed"]});
    dot.addEventListener("mousemove",e=>showTip(e,`<b>${p}${dated?" · ≤"+c.years[p]:""}</b><span class="fam">${D.family[p]||""}</span><div class="txt">“${c.examples[p]}”</div>`));
    dot.addEventListener("mouseleave",hideTip);pts.appendChild(dot);});
  const dated=Object.entries(c.years).sort((a,b)=>a[1]-b[1]);
  panel.innerHTML=`<h2>“${c.claim}”</h2>
   <div class="meta">${c.coverage} cultures · ${c.support} attestations · ${dated.length} dated${dated.length?` · earliest ≤${dated[0][1]} (${dated[0][0]})`:""}</div>
   <ul class="exlist">${active.sort().map(p=>`<li><b>${p}${c.years[p]?` (≤${c.years[p]})`:""}:</b> “${c.examples[p]}”</li>`).join("")}</ul>`;}
function renderNet(){
  arcs.innerHTML="";pts.innerHTML="";
  const maxW=D.edges.length?D.edges[0][2]:1;
  D.edges.forEach(([a,b,w])=>{const pa=D.coords[a],pb=D.coords[b];if(!pa||!pb)return;
    const p=el("path",{d:arcPath(pa,pb),class:"arc net","stroke-width":(0.6+3.4*w/maxW).toFixed(2)});
    p.style.pointerEvents="stroke";
    p.addEventListener("mousemove",e=>showTip(e,`<b>${a} ↔ ${b}</b><span class="fam">${w} shared motifs</span>`));
    p.addEventListener("mouseleave",hideTip);arcs.appendChild(p);});
  Object.entries(D.coords).forEach(([p,xy])=>{
    const dot=el("circle",{cx:xy[0],cy:xy[1],r:4,class:"pt",fill:GC[D.group[p]||"Other / mixed"]});
    dot.addEventListener("mousemove",e=>showTip(e,`<b>${p}</b><span class="fam">${D.family[p]||""}</span>`));
    dot.addEventListener("mouseleave",hideTip);pts.appendChild(dot);});
  panel.innerHTML=`<h2>Shared-wisdom network</h2><div class="meta">line weight = motifs two cultures share</div>`;}
function render(){mode==="motif"?renderMotif():renderNet();}
function press(on,off){document.getElementById(on).setAttribute("aria-pressed","true");
document.getElementById(off).setAttribute("aria-pressed","false");}
document.getElementById("bMotif").onclick=()=>{mode="motif";press("bMotif","bNet");sel.style.display="";timebox.style.display="";render();};
document.getElementById("bNet").onclick=()=>{mode="net";press("bNet","bMotif");sel.style.display="none";timebox.style.display="none";render();};
document.getElementById("bWorld").onclick=()=>{press("bWorld","bEur");map.setAttribute("viewBox","0 0 1000 520");};
document.getElementById("bEur").onclick=()=>{press("bEur","bWorld");map.setAttribute("viewBox","430 60 260 190");};
sel.onchange=()=>{idx=+sel.value;render();};
slider.oninput=()=>{yearLabel.textContent=slider.value;render();};
showUndated.onchange=render;
render();
</script></body></html>"""
