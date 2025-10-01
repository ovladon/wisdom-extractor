import os, random, json
import streamlit as st
import pandas as pd

from core.proposition_extractor import extract_proposition
from core.meaning_graph import build_edges, communities_from_edges, nearest_pairs
from core.survival_score import survival_score
from core.persistence import (
    init_db, upsert_source, list_sources, insert_proverb, list_proverbs,
    mark_excluded, save_proposition, add_constraint, stats, leaderboard, export_annotations
)
from scraper.basic_scraper import fetch, discover_links, extract_items, crawl_source

CATALOG = {
  "sources": [
    {"name":"Wikiquote — English Proverbs","url":"https://en.wikiquote.org/wiki/English_proverbs","tags":["wikiquote","english","list"]},
    {"name":"Wikiquote — Romanian Proverbs","url":"https://ro.wikiquote.org/wiki/Proverbe_rom%C3%A2ne%C5%9fti","tags":["wikiquote","romanian","list"]},
    {"name":"Wikiquote — Russian Proverbs","url":"https://ru.wikiquote.org/wiki/%D0%9F%D0%BE%D1%81%D0%BB%D0%BE%D0%B2%D0%B8%D1%86%D1%8B","tags":["wikiquote","russian","list"]},
    {"name":"Wikiquote — French Proverbs","url":"https://fr.wikiquote.org/wiki/Proverbes_fran%C3%A7ais","tags":["wikiquote","french","list"]},
    {"name":"Wikiquote — Spanish Proverbs","url":"https://es.wikiquote.org/wiki/Refranes","tags":["wikiquote","spanish","list"]},
    {"name":"Wikiquote — German Proverbs","url":"https://de.wikiquote.org/wiki/Sprichwort","tags":["wikiquote","german","list"]},
    {"name":"Wikiquote — Italian Proverbs","url":"https://it.wikiquote.org/wiki/Proverbi_italiani","tags":["wikiquote","italian","list"]},
    {"name":"Wikiquote — Portuguese Proverbs","url":"https://pt.wikiquote.org/wiki/Prov%C3%A9rbios_portugueses","tags":["wikiquote","portuguese","list"]},
    {"name":"Wikiquote — Dutch Proverbs","url":"https://nl.wikiquote.org/wiki/Spreekwoord","tags":["wikiquote","dutch","list"]},
    {"name":"Wikiquote — Finnish Proverbs","url":"https://fi.wikiquote.org/wiki/Sanonnat","tags":["wikiquote","finnish","list"]},
    {"name":"Wikiquote — Polish Proverbs","url":"https://pl.wikiquote.org/wiki/Przys%C5%82owia_polskie","tags":["wikiquote","polish","list"]},
    {"name":"Wikiquote — Czech Proverbs","url":"https://cs.wikiquote.org/wiki/P%C5%99%C3%ADslov%C3%AD","tags":["wikiquote","czech","list"]},
    {"name":"Wikiquote — Greek Proverbs","url":"https://el.wikiquote.org/wiki/%CE%A0%CE%B1%CF%81%CE%BF%CE%B9%CE%BC%CE%AF%CE%B5%CF%82","tags":["wikiquote","greek","list"]},
    {"name":"Wikiquote — Turkish Proverbs","url":"https://tr.wikiquote.org/wiki/Atas%C3%B6zleri","tags":["wikiquote","turkish","list"]},
    {"name":"Wikiquote — Arabic Proverbs","url":"https://ar.wikiquote.org/wiki/%D8%A3%D9%85%D8%AB%D8%A7%D9%84_%D8%B9%D8%B1%D8%A8%D9%8A%D8%A9","tags":["wikiquote","arabic","list"]},
    {"name":"Wikiquote — Persian Proverbs","url":"https://fa.wikiquote.org/wiki/%D8%B6%D8%B1%D8%A7%D8%A6%D8%A8_%D8%A7%D9%84%D8%A3%D9%85%D8%AB%D8%A7%D9%84","tags":["wikiquote","persian","list"]},
    {"name":"Wikiquote — Hindi Proverbs","url":"https://hi.wikiquote.org/wiki/%E0%A4%B2%E0%A5%8B%E0%A4%95%E0%A5%8B%E0%A4%95%E0%A5%8D%E0%A4%A4%E0%A4%BF%E0%A4%AF%E0%A4%BE%E0%A4%81","tags":["wikiquote","hindi","list"]},
    {"name":"Wikiquote — Chinese Proverbs","url":"https://zh.wikiquote.org/wiki/%E8%A8%80%E8%91%89","tags":["wikiquote","chinese","list"]},
    {"name":"Wikiquote — Japanese Proverbs","url":"https://ja.wikiquote.org/wiki/%E8%91%89%E5%8F%A5","tags":["wikiquote","japanese","list"]},
    {"name":"Wikiquote — Korean Proverbs","url":"https://ko.wikiquote.org/wiki/%EC%86%8C%EA%B0%9C:%EC%86%8C%EC%8A%A4%EB%9F%AC","tags":["wikiquote","korean","list"]}
  ]
}

def seed_if_empty():
    from core.persistence import list_sources, upsert_source
    if not list_sources():
        for s in CATALOG.get("sources", []):
            upsert_source(s.get("name", s.get("url","(no name)")), s["url"], ",".join(s.get("tags",[])))

st.set_page_config(page_title='Wisdom Lab — Full Plus (Seeded v16)', layout='wide')
st.title('Wisdom Lab — Collect • Cluster • Annotate • Persist (v16)')

init_db()
seed_if_empty()

DB_PATH = os.environ.get("WISDOM_DB_PATH", "wisdom.db")
st.caption(f"DB: {os.path.abspath(DB_PATH)} • CWD: {os.getcwd()}")

seed_col1, seed_col2 = st.columns(2)
if seed_col1.button("Seed built-in catalog now"):
    seed_if_empty(); st.success("Catalog seeded.")
if seed_col2.button("Reset DB sources and re-seed"):
    import sqlite3
    con = sqlite3.connect(DB_PATH, check_same_thread=False); cur = con.cursor()
    try:
        cur.execute("DELETE FROM sources"); con.commit(); st.info("Cleared sources.")
    finally:
        con.close()
    seed_if_empty(); st.success("Re-seeded catalog.")

st.sidebar.subheader('Who are you?')
user = st.sidebar.text_input('Your name (for leaderboard)', value='(anon)')

tabs = st.tabs(['Sources & Scrape','Import CSV','Propositions','Graph','Communities','Annotate • Play','Candidates','Leaderboard & Export'])

with tabs[0]:
    st.header('Sources & Scrape')
    srcs = list_sources()
    st.caption(f"Sources in DB: {len(srcs)}")
    st.dataframe(pd.DataFrame(srcs))

    colA, colB = st.columns([2,1])
    with colA:
        st.subheader('Scrape settings')
        respect_robots = st.checkbox('Respect robots.txt (recommended)', value=True)
        workers = st.slider('Concurrent fetch workers', 1, 16, 8, 1)
        tag_filter = st.text_input('Filter sources by tags substring (optional)', '')
        filtered = [s for s in srcs if (tag_filter.strip().lower() in (s['tags'] or '').lower())] if tag_filter else srcs
        ids = [s['id'] for s in filtered]
        labels = [f"{s['name']} ({s['url']})" for s in filtered]
        pick = st.multiselect('Pick sources (empty = all)', options=ids, format_func=lambda i: labels[ids.index(i)] if i in ids else str(i))
        crawl_btn = st.button('🚀 Crawl now (depth‑1, uncapped)')

    with colB:
        st.subheader('Import/Export catalog (JSON)')
        up = st.file_uploader('Import catalog JSON', type=['json'], key='cat_up')
        if up is not None:
            try:
                data = json.load(up)
                for s in data.get('sources', []):
                    upsert_source(s.get('name', s.get('url','(no name)')), s['url'], ','.join(s.get('tags',[])))
                st.success('Catalog imported.')
            except Exception as e:
                st.error(f'Failed: {e}')
        if st.button('Export catalog JSON'):
            srcs = list_sources()
            out = {'sources':[{'name':s['name'],'url':s['url'],'tags':s['tags'].split(',') if s['tags'] else []} for s in srcs]}
            st.download_button('Download sources_catalog.json', data=json.dumps(out, ensure_ascii=False, indent=2), file_name='sources_catalog.json', mime='application/json')

    if crawl_btn:
        target_srcs = filtered if not pick else [s for s in filtered if s['id'] in pick]
        if not target_srcs:
            st.warning('No sources selected (and none after filter).')
        else:
            total_new = 0
            progress = st.progress(0.0)
            for idx, s in enumerate(target_srcs):
                st.write(f"**Crawling:** {s['name']} — {s['url']}")
                try:
                    pages, items = crawl_source(s['url'], respect_robots=respect_robots, workers=workers)
                    st.caption(f"Discovered {len(pages)} pages; extracted {len(items)} items")
                    for it in items:
                        pid = insert_proverb(s['id'], it['text'], it['url'])
                        if pid: total_new += 1
                except Exception as e:
                    st.warning(f"Failed source {s['url']}: {e}")
                progress.progress((idx+1)/max(1,len(target_srcs)))
            st.success(f"Done. New proverbs saved: {total_new}")

with tabs[1]:
    st.header('Import CSV')
    up = st.file_uploader('CSV file', type=['csv'])
    if up is not None:
        df = pd.read_csv(up)
        cols = df.columns.tolist()
        def guess(cands):
            for c in cols:
                if any(k in c.lower() for k in cands): return c
            return cols[0]
        text_col = st.selectbox('Text', cols, index=cols.index(guess(['text','claim','proverb','saying','quote'])))
        lang_col = st.selectbox('Language', ['<none>']+cols)
        fam_col  = st.selectbox('Family', ['<none>']+cols)
        reg_col  = st.selectbox('Region', ['<none>']+cols)
        if st.button('Import into DB'):
            sid = upsert_source('CSV Import', f'file://{up.name}', 'csv')
            newc = 0
            for _,r in df.iterrows():
                text = str(r.get(text_col,''))
                if not text.strip(): 
                    continue
                lang = None if lang_col=='<none>' else r.get(lang_col)
                fam  = None if fam_col=='<none>' else r.get(fam_col)
                reg  = None if reg_col=='<none>' else r.get(reg_col)
                if insert_proverb(sid, text, url=f'file://{up.name}', language=lang, family=fam, region=reg):
                    newc += 1
            st.success(f'Imported {newc} rows.')

with tabs[2]:
    st.header('Propositions')
    rows = list_proverbs(excluded=False)
    st.caption(f'Active proverbs: {len(rows)}')
    if st.button('Compute idea_formula & frame for all (light rules)') and rows:
        for r in rows:
            p = extract_proposition(r['text'])
            save_proposition(r['id'], p['idea_formula'], p['frame'])
        st.success('Updated proposition fields.')
    df = pd.DataFrame(list_proverbs(excluded=False))
    if not df.empty:
        st.dataframe(df.head(50))

with tabs[3]:
    st.header('Graph (TF‑IDF paraphrase fallback)')
    df = pd.DataFrame(list_proverbs(excluded=False))
    if df.empty:
        st.info('No data. Scrape or import first.')
    else:
        thr = st.slider('Paraphrase threshold', 0.30, 0.90, 0.42, 0.01)
        if st.button('Build graph now'):
            work = df.rename(columns={'id':'id','text':'text'})
            edges = build_edges(work, 'text', thr)
            st.session_state['edges'] = edges
            st.write(edges.head(20))
            st.success(f'Edges: {len(edges)}')

with tabs[4]:
    st.header('Communities')
    edges = st.session_state.get('edges')
    if edges is None or edges.empty:
        st.info('Build the graph first.')
    else:
        comp = communities_from_edges(edges)
        df = pd.DataFrame(list_proverbs(excluded=False))
        df['community_id'] = df['id'].map(comp).fillna(-1).astype(int)
        st.session_state['communities'] = df[['id','community_id']]
        st.dataframe(df[['id','text','community_id']].head(50))

with tabs[5]:
    st.header('Annotate — Play Mode')
    df = pd.DataFrame(list_proverbs(excluded=False))
    if df.empty:
        st.info('No active proverbs. Scrape or import first.')
    else:
        pos, neg = nearest_pairs(df.rename(columns={'id':'id','text':'text'}), 'text', k=8, hi=0.85, lo=0.35)
        strategy = st.radio('Pair strategy', ['Surprise me','Likely same idea','Likely different idea','Cross‑language'], horizontal=True)
        def pick_pair():
            if strategy=='Likely same idea' and pos: return random.choice(pos)
            if strategy=='Likely different idea' and neg: return random.choice(neg)
            if strategy=='Cross‑language' and 'language' in df.columns:
                i1,i2 = random.sample(range(len(df)),2); return (df.iloc[i1]['id'], df.iloc[i2]['id'], 0.0)
            a,b = random.sample(df['id'].tolist(), 2); return (a,b,0.0)
        if 'pair' not in st.session_state:
            st.session_state['pair'] = pick_pair()
        a,b,s = st.session_state['pair']
        ra = df[df['id']==a].iloc[0]
        rb = df[df['id']==b].iloc[0]
        c1,c2 = st.columns(2)
        with c1:
            st.subheader('Proverb A')
            st.write(ra['text']); st.caption(f"ID: {int(ra['id'])}")
            if st.button('❌ Not a saying (A)'):
                mark_excluded(int(ra['id']), True); st.success('Excluded A'); st.session_state['pair']=pick_pair(); st.rerun()
        with c2:
            st.subheader('Proverb B')
            st.write(rb['text']); st.caption(f"ID: {int(rb['id'])}")
            if st.button('❌ Not a saying (B)'):
                mark_excluded(int(rb['id']), True); st.success('Excluded B'); st.session_state['pair']=pick_pair(); st.rerun()
        d1,d2,d3 = st.columns(3)
        if d1.button('✅ Must‑Link'):
            add_constraint(int(ra['id']), int(rb['id']), 'must', user); st.session_state['pair']=pick_pair(); st.rerun()
        if d2.button('🚫 Cannot‑Link'):
            add_constraint(int(ra['id']), int(rb['id']), 'cannot', user); st.session_state['pair']=pick_pair(); st.rerun()
        if d3.button('⏭️ Skip'):
            st.session_state['pair']=pick_pair(); st.rerun()
        st.markdown('---'); st.subheader('Live stats'); st.json(stats())

with tabs[6]:
    st.header('Candidates (Survival Score)')
    df = pd.DataFrame(list_proverbs(excluded=False))
    if df.empty or 'communities' not in st.session_state:
        st.info('Need communities. Go to Graph → Communities first.')
    else:
        com = st.session_state['communities']
        merged = df.merge(com, on='id', how='left').fillna({'community_id':-1})
        rows = []
        for c, g in merged.groupby('community_id'):
            if c == -1: 
                continue
            rows.append({'community_id': int(c), 'size': int(len(g)), 'S': float(survival_score(g.to_dict('records')))})
        cand = pd.DataFrame(rows).sort_values(['S','size'], ascending=[False,False])
        st.dataframe(cand.head(100))
        st.download_button('Download candidates.csv', cand.to_csv(index=False), 'candidates.csv', 'text/csv')

with tabs[7]:
    st.header('Leaderboard & Export')
    st.subheader('Leaderboard'); st.dataframe(pd.DataFrame(leaderboard()))
    st.subheader('Export annotations'); exp = export_annotations()
    st.download_button('Download annotation_export.json', data=json.dumps(exp, ensure_ascii=False, indent=2), file_name='annotation_export.json', mime='application/json')
    st.subheader('Download proverbs CSV')
    df = pd.DataFrame(list_proverbs(excluded=False))
    if df.empty:
        st.info('No data.')
    else:
        st.download_button('Download proverbs.csv', df.to_csv(index=False), 'proverbs.csv', 'text/csv')
    st.subheader('Download database')
    DB_PATH = os.environ.get('WISDOM_DB_PATH', 'wisdom.db')
    if os.path.exists(DB_PATH):
        with open(DB_PATH,'rb') as f:
            st.download_button('Download wisdom.db', f.read(), 'wisdom.db', 'application/octet-stream')
