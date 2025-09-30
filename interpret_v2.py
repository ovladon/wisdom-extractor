
import json, pandas as pd, re
from collections import Counter

THEMES = [
  ("Cooperation / Social Support", ["cooperat","together","friend","help","union","many hands","community","neighbor","neighbour"]),
  ("Prudence / Risk & Uncertainty", ["avoid","never","caution","careful","haste","risk","danger","guard","beware","look before","prudence","slow","steady"]),
  ("Effort / Time / Persistence", ["work","practice","effort","time","early","persever","patience","diligence","delay"]),
  ("Trust / Honesty / Deception", ["honest","truth","deception","lie","false","trust","cheat","thief","wolf","fox"]),
  ("Family / Kinship / Obligation", ["family","blood","kin","father","mother","son","daughter","home","house"]),
  ("Fate / Fortune / Opportunity", ["fortune","luck","opportunity","chance","bold","fate","destiny","star"]),
  ("Knowledge / Speech / Silence", ["word","speak","silence","listen","tongue","knowledge","book","learn","wisdom"]),
]

def theme_of(text):
    low = str(text).lower()
    for theme, keys in THEMES:
        if any(k in low for k in keys):
            return theme
    return "Other"

def top_counts(xs, k=3):
    return ", ".join([f"{a} ({b})" for a,b in Counter(xs).most_common(k)]) if xs else "-"

def _with_pct(label_counts: str, coverage: int) -> str:
    if not label_counts or coverage in (0, None):
        return label_counts
    out = []
    import re as _re
    for piece in [p.strip() for p in label_counts.split(",")]:
        m = _re.search(r"(.+?)\s*\((\d+)\)", piece)
        if m:
            k = m.group(1).strip(); n = int(m.group(2))
            pct = int(round(100.0*n/max(1,coverage)))
            out.append(f"{k} ({n}/{coverage}; {pct}%)")
    return ", ".join(out) if out else label_counts

def summarize(clusters_path, metadata_path, out_path):
    data = json.load(open(clusters_path,"r",encoding="utf-8"))
    meta = pd.read_csv(metadata_path)
    enriched = []
    for c in data:
        regs=fams=clims=coast=isl=mar=trade=migr=subs=stap=legal=urban=indiv=unc = []
        regs, fams, clims, coast, isl, mar, trade, migr, subs, stap, legal, urban, indiv, unc = ([] for _ in range(14))
        for peep in c.get("cultures", []):
            m = meta[meta["people"].str.lower() == str(peep).lower()]
            for _,r in m.iterrows():
                regs.append(r["region"]); fams.append(r["language_family"]); clims.append(r["climate_zone"])
                coast.append(r.get("coastal","?")); isl.append(r.get("is_island","?")); mar.append(r.get("maritime_orientation","?"))
                trade.append(r.get("trade_route_proximity","?")); migr.append(r.get("migration_hub","?")); subs.append(r.get("subsistence_bias","?"))
                stap.append(r.get("staple_crop","?")); legal.append(r.get("legal_origin","?")); urban.append(r.get("urbanization_level","?"))
                indiv.append(r.get("individualism_proxy","?")); unc.append(r.get("uncertainty_avoidance_proxy","?"))
        c["theme"] = theme_of(c.get("claim",""))
        c["t_regions"] = _with_pct(top_counts(regs), c.get("coverage",0))
        c["t_families"] = _with_pct(top_counts(fams), c.get("coverage",0))
        c["t_climates"] = _with_pct(top_counts(clims), c.get("coverage",0))
        c["t_coastal"] = _with_pct(top_counts(coast), c.get("coverage",0))
        c["t_island"] = _with_pct(top_counts(isl), c.get("coverage",0))
        c["t_maritime"] = _with_pct(top_counts(mar), c.get("coverage",0))
        c["t_trade"] = _with_pct(top_counts(trade), c.get("coverage",0))
        c["t_migration"] = _with_pct(top_counts(migr), c.get("coverage",0))
        c["t_subsistence"] = _with_pct(top_counts(subs), c.get("coverage",0))
        c["t_staple"] = _with_pct(top_counts(stap), c.get("coverage",0))
        c["t_legal"] = _with_pct(top_counts(legal), c.get("coverage",0))
        c["t_urban"] = _with_pct(top_counts(urban), c.get("coverage",0))
        c["t_indiv"] = _with_pct(top_counts(indiv), c.get("coverage",0))
        c["t_uncert"] = _with_pct(top_counts(unc), c.get("coverage",0))
        enriched.append(c)

    enriched.sort(key=lambda c: (-c.get("coverage",0), -c.get("support",0)))
    top = enriched[:30]
    sections = []
    sections.append("# Cross-cultural interpretation (offline; deterministic)")
    sections.append("**How to read this report.** ‘Coverage’ = number of distinct peoples. "
                    "‘Europe (SE) (1/3; 33%)’ ⇒ among 3 covered peoples, 1 is in Europe (SE). Percentages are descriptive, not causal.")
    for theme, _ in THEMES + [("Other", [])]:
        themelist = [c for c in top if c["theme"]==theme]
        if not themelist: continue
        sections.append(f"## {theme}")
        for c in themelist[:4]:
            sections.append(
                f"- **Claim:** {c['claim']}\n"
                f"  - Coverage: {c.get('coverage',0)} peoples; Support: {c.get('support',0)} instances; Wisdom score: {c.get('wisdom_score','-')}\n"
                f"  - Regions/Families/Climates: {c['t_regions']} / {c['t_families']} / {c['t_climates']}\n"
                f"  - Coastal/Island/Maritime: {c['t_coastal']} / {c['t_island']} / {c['t_maritime']}\n"
                f"  - Trade/Migration: {c['t_trade']} / {c['t_migration']}\n"
                f"  - Subsistence/Staple: {c['t_subsistence']} / {c['t_staple']}\n"
                f"  - Legal/Urban: {c['t_legal']} / {c['t_urban']}\n"
                f"  - Individualism/Uncertainty: {c['t_indiv']} / {c['t_uncert']}"
            )
    text = "\n\n".join(sections)
    open(out_path,"w",encoding="utf-8").write(text)
    ej = out_path.replace(".txt","_enriched.json")
    with open(ej,"w",encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    return out_path
