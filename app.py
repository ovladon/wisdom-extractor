import os, random, json, time
import streamlit as st
import pandas as pd

from core.proposition_extractor import extract_proposition
from core.meaning_graph import build_edges, communities_from_edges, nearest_pairs
from core.survival_score import survival_score
from core.persistence import (
    init_db, upsert_source, list_sources, insert_proverb, list_proverbs,
    mark_excluded, save_proposition, add_constraint, stats, leaderboard, export_annotations, bulk_apply
)
from scraper.basic_scraper import crawl_source

CATALOG = json.load(open(os.path.join('data','sources_catalog.json'), 'r', encoding='utf-8')) if os.path.exists(os.path.join('data','sources_catalog.json')) else {"sources":[]}

def seed_if_empty():
    from core.persistence import list_sources, upsert_source
    if not list_sources():
        for s in CATALOG.get("sources", []):
            upsert_source(s.get("name", s.get("url","(no name)")), s["url"], ",".join(s.get("tags",[])))

st.set_page_config(page_title='Wisdom Lab — Full Plus (Seeded v18)', layout='wide')
st.title('Wisdom Lab — Collect • Cluster • Annotate • Persist (v18 — fast annotate)')

init_db()
seed_if_empty()

DB_PATH = os.environ.get("WISDOM_DB_PATH", "wisdom.db")
st.caption(f"DB: {os.path.abspath(DB_PATH)} • CWD: {os.getcwd()}")

if 'db_version' not in st.session_state: st.session_state['db_version'] = 0
if 'pending_ops' not in st.session_state: st.session_state['pending_ops'] = []
if 'excluded_pending' not in st.session_state: st.session_state['excluded_pending'] = set()

st.sidebar.subheader('Who are you?')
user = st.sidebar.text_input('Your name (for leaderboard)', value='(anon)')
write_mode = st.sidebar.radio('Write mode', ['Batch (faster)','Instant'], index=0, help='Batch: queue updates and write with one click. Instant: write to DB immediately.')
autosave_after = st.sidebar.number_input('Auto-save after N actions (batch)', min_value=1, max_value=200, value=20, step=1)

@st.cache_data(show_spinner=False)
def cached_list_proverbs(db_version: int):
    return pd.DataFrame(list_proverbs(excluded=False))

@st.cache_data(show_spinner=False)
def cached_pairs(ids, texts, hi, lo, k):
    df = pd.DataFrame({'id': ids, 'text': texts})
    return nearest_pairs(df, 'text', k=k, hi=hi, lo=lo)

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
        c1, c2 = st.columns(2)
        crawl_btn = c1.button('🚀 Crawl now (depth‑1, uncapped)')
        stop_btn = c2.button('🛑 Stop after current source')
        if stop_btn:
            st.session_state['stop_crawl'] = True
            st.info('Stop requested: the crawl will halt after the current source finishes.')

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
        st.session_state['stop_crawl'] = False
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
                if st.session_state.get('stop_crawl'):
                    st.info('Stopping as requested.'); break
            st.success(f"Done. New proverbs saved: {total_new}")
            st.session_state['db_version'] += 1

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
            st.session_state['db_version'] += 1

with tabs[2]:
    st.header('Propositions')
    df = cached_list_proverbs(st.session_state['db_version'])
    st.caption(f'Active proverbs: {len(df)}')
    if st.button('Compute idea_formula & frame for all (light rules)') and not df.empty:
        for _, r in df.iterrows():
            p = extract_proposition(r['text'])
            save_proposition(int(r['id']), p['idea_formula'], p['frame'])
        st.success('Updated proposition fields.')
        st.session_state['db_version'] += 1
    if not df.empty:
        st.dataframe(df.head(50))

with tabs[3]:
    st.header('Graph (TF‑IDF paraphrase fallback)')
    df = cached_list_proverbs(st.session_state['db_version'])
    if df.empty:
        st.info('No data. Scrape or import first.')
    else:
        thr = st.slider('Paraphrase threshold', 0.30, 0.90, 0.42, 0.01)
        if st.button('Build graph now'):
            work = df[['id','text']].copy()
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
        df = cached_list_proverbs(st.session_state['db_version'])
        comp = communities_from_edges(edges)
        df['community_id'] = df['id'].map(comp).fillna(-1).astype(int)
        st.session_state['communities'] = df[['id','community_id']]
        st.dataframe(df[['id','text','community_id']].head(50))

with tabs[5]:
    st.header('Annotate — Play Mode (fast)')
    df = cached_list_proverbs(st.session_state['db_version'])
    if st.session_state['excluded_pending']:
        df = df[~df['id'].isin(st.session_state['excluded_pending'])]
    if df.empty:
        st.info('No active proverbs. Scrape or import first.')
    else:
        hi = st.slider('Hi threshold', 0.50, 0.95, 0.85, 0.01)
        lo = st.slider('Lo threshold', 0.05, 0.60, 0.35, 0.01)
        k = st.slider('Neighbors per anchor (k)', 2, 20, 8, 1)
        if 'pairs' not in st.session_state or st.button('Refresh pairs'):
            ids = df['id'].tolist(); texts = df['text'].astype(str).tolist()
            pos, neg = cached_pairs(ids, texts, hi, lo, k)
            st.session_state['pairs'] = {'pos':pos, 'neg':neg}

        pairs = st.session_state.get('pairs', {'pos':[], 'neg':[]})
        strategy = st.radio('Pair strategy', ['Surprise me','Likely same idea','Likely different idea','Cross‑language'], horizontal=True)

        def pick_pair():
            import random
            if strategy=='Likely same idea' and pairs['pos']:
                return random.choice(pairs['pos'])
            if strategy=='Likely different idea' and pairs['neg']:
                return random.choice(pairs['neg'])
            if strategy=='Cross‑language' and 'language' in df.columns and len(df) >= 2:
                i1,i2 = random.sample(range(len(df)),2); return (int(df.iloc[i1]['id']), int(df.iloc[i2]['id']), 0.0)
            a,b = random.sample(df['id'].tolist(), 2); return (int(a), int(b), 0.0)

        if 'pair' not in st.session_state:
            st.session_state['pair'] = pick_pair()
        a,b,s = st.session_state['pair']
        ra = df[df['id']==a].iloc[0] if (df['id']==a).any() else df.sample(1).iloc[0]
        rb = df[df['id']==b].iloc[0] if (df['id']==b).any() else df.sample(1).iloc[0]

        c1,c2 = st.columns(2)
        with c1:
            st.subheader('Proverb A')
            st.write(ra['text']); st.caption(f"ID: {int(ra['id'])}")
            if st.button('❌ Not a saying (A)'):
                if write_mode.startswith('Instant'):
                    mark_excluded(int(ra['id']), True)
                    st.session_state['db_version'] += 1
                else:
                    st.session_state['pending_ops'].append({'op':'exclude','pid': int(ra['id'])})
                    st.session_state['excluded_pending'].add(int(ra['id']))
                if 'pairs' in st.session_state:
                    st.session_state['pairs']['pos'] = [p for p in st.session_state['pairs']['pos'] if p[0]!=int(ra['id']) and p[1]!=int(ra['id'])]
                    st.session_state['pairs']['neg'] = [p for p in st.session_state['pairs']['neg'] if p[0]!=int(ra['id']) and p[1]!=int(ra['id'])]
                st.toast('Excluded A', icon='❌')
                st.session_state['pair'] = pick_pair()

        with c2:
            st.subheader('Proverb B')
            st.write(rb['text']); st.caption(f"ID: {int(rb['id'])}")
            if st.button('❌ Not a saying (B)'):
                if write_mode.startswith('Instant'):
                    mark_excluded(int(rb['id']), True)
                    st.session_state['db_version'] += 1
                else:
                    st.session_state['pending_ops'].append({'op':'exclude','pid': int(rb['id'])})
                    st.session_state['excluded_pending'].add(int(rb['id']))
                if 'pairs' in st.session_state:
                    st.session_state['pairs']['pos'] = [p for p in st.session_state['pairs']['pos'] if p[0]!=int(rb['id']) and p[1]!=int(rb['id'])]
                    st.session_state['pairs']['neg'] = [p for p in st.session_state['pairs']['neg'] if p[0]!=int(rb['id']) and p[1]!=int(rb['id'])]
                st.toast('Excluded B', icon='❌')
                st.session_state['pair'] = pick_pair()

        d1,d2,d3 = st.columns(3)
        if d1.button('✅ Must‑Link'):
            if write_mode.startswith('Instant'):
                add_constraint(int(ra['id']), int(rb['id']), 'must', user)
                st.session_state['db_version'] += 1
            else:
                st.session_state['pending_ops'].append({'op':'constraint','a':int(ra['id']),'b':int(rb['id']),'label':'must','user':user})
            st.toast('Saved MUST link', icon='✅')
            st.session_state['pair']=pick_pair()

        if d2.button('🚫 Cannot‑Link'):
            if write_mode.startswith('Instant'):
                add_constraint(int(ra['id']), int(rb['id']), 'cannot', user)
                st.session_state['db_version'] += 1
            else:
                st.session_state['pending_ops'].append({'op':'constraint','a':int(ra['id']),'b':int(rb['id']),'label':'cannot','user':user})
            st.toast('Saved CANNOT link', icon='🚫')
            st.session_state['pair']=pick_pair()

        if d3.button('⏭️ Skip'):
            st.session_state['pair']=pick_pair()

        if write_mode.startswith('Batch') and len(st.session_state['pending_ops']) >= autosave_after:
            bulk_apply(st.session_state['pending_ops'])
            st.session_state['pending_ops'].clear()
            st.session_state['excluded_pending'].clear()
            st.session_state['db_version'] += 1
            st.toast('Auto-saved pending annotations to DB', icon='💾')

        st.markdown('---')
        cA,cB = st.columns(2)
        if write_mode.startswith('Batch'):
            if cA.button('💾 Save pending now'):
                bulk_apply(st.session_state['pending_ops'])
                st.session_state['pending_ops'].clear()
                st.session_state['excluded_pending'].clear()
                st.session_state['db_version'] += 1
                st.success('Saved to DB.')
        cB.caption(f"Pending ops: {len(st.session_state['pending_ops'])}")

        st.subheader('Live stats'); st.json(stats())

with tabs[6]:
    st.header('Candidates (Survival Score)')
    df = cached_list_proverbs(st.session_state['db_version'])
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
    df = cached_list_proverbs(st.session_state['db_version'])
    if df.empty:
        st.info('No data.')
    else:
        st.download_button('Download proverbs.csv', df.to_csv(index=False), 'proverbs.csv', 'text/csv')
    st.subheader('Download database')
    DB_PATH = os.environ.get('WISDOM_DB_PATH', 'wisdom.db')
    if os.path.exists(DB_PATH):
        with open(DB_PATH,'rb') as f:
            st.download_button('Download wisdom.db', f.read(), 'wisdom.db', 'application/octet-stream')
