# Hosting the annotator portal (send people a URL)

**TL;DR:** Netlify cannot run this app (static hosting; Streamlit is a persistent
Python server). The cheapest solid home is a ~€4/month VPS; a managed alternative is
Hugging Face Spaces with a $5/month persistent disk.

## What annotators get
`annotator_app.py` — the safe portal: annotation game, leaderboard, personal
consistency score, world map. No admin tabs, optional access code, all judgments in
one shared database. You keep the full `app.py` for yourself, on the same database.

## Option A (recommended): small VPS — ~€4/month, full control, data is yours
1. Create an Ubuntu VPS (Hetzner CX22 ≈ €3.8/mo, or DigitalOcean/OVH equivalent).
2. Point a DNS A record (e.g. `wisdom.yourdomain.com`) at the server IP.
   (A domain via Netlify DNS works fine for this — that part of Netlify IS usable.)
3. On the server:
   ```bash
   apt update && apt install -y docker.io docker-compose-v2 git
   git clone https://github.com/ovladon/wisdom-extractor.git && cd wisdom-extractor/deploy
   # edit Caddyfile: replace wisdom.example.com with your domain
   echo "ANNOTATOR_CODE=choose-a-code" > .env      # optional gate
   docker compose up -d
   ```
4. Seed the shared database once: `docker compose exec wisdom python -c "..."` or
   temporarily set APP_FILE=app.py, seed via tab 2, then switch back.
5. Backups (the annotations are the treasure):
   `docker compose cp wisdom:/data/wisdom.db ./backup-$(date +%F).db` — cron it weekly.

## Option B: Hugging Face Spaces — managed, ~$5/month for persistence
1. Create a Space (SDK: Docker), push this repo to it.
2. In Space settings: add **persistent storage** (small tier) and set variables
   `WISDOM_DB_PATH=/data/wisdom.db`, `APP_FILE=annotator_app.py`, `ANNOTATOR_CODE=...`.
3. URL is `https://huggingface.co/spaces/<you>/<space>`. Free tier works but wipes
   the database on every restart — do not run the study without persistent storage.

## Option C: Streamlit Community Cloud — free, for a demo only
Deploys straight from the GitHub repo in ~3 clicks, but storage is ephemeral:
annotations vanish on redeploy/restart. Fine for showing Waterloo the interface;
wrong for collecting real data.

## What your existing accounts are good for
- **Netlify**: a landing page (project story, map screenshots, big button → the app URL)
  and/or DNS for the subdomain. Not the app itself.
- **Buttondown**: the annotator community newsletter — recruitment, weekly
  "top contributor + newly settled motifs" digests, study updates.

## The mobile app (v19.5)

`mobile/` is a phone-first swipe interface (PWA): judge pairs with a swipe
(right = same idea, left = different), streaks, leaderboard, consistency score.
It shares the same database and access code as the portal. The compose file serves it
as a second subdomain (see Caddyfile) — that's the URL to share for
"annotate whenever you have a minute". On a phone, "Add to Home Screen" installs it
like an app. Local test: `uvicorn mobile.mobile_api:app --port 8600`.
