#!/usr/bin/env bash
# Start (or restart) the Wisdom Lab on the phone: server + tunnel (+ stable Netlify link
# if configured) + daily backup loop + weekly maintenance loop. Prints the public URL.
set -u
CODE="${ANNOTATOR_CODE:-proverbe}"
DB=/root/wisdom-live/wisdom.db
REPO=/root/wisdom-extractor
VENVBIN=/root/wisdom-venv/bin
LOG=/root/wisdom-live
[ -f "$DB" ] || { echo "database missing: $DB — copy it first (PHONE_SERVER.md step 5)"; exit 1; }
[ -f /root/.netlify_env ] && . /root/.netlify_env

pkill -f 'mobile_ap[i]' 2>/dev/null; pkill -f 'cloudflare[d]' 2>/dev/null
pkill -f 'wisdom_backup_loo[p]' 2>/dev/null; pkill -f 'wisdom_maintain_loo[p]' 2>/dev/null
sleep 1

cd "$REPO"
WISDOM_DB_PATH="$DB" ANNOTATOR_CODE="$CODE" \
  nohup "$VENVBIN/uvicorn" mobile.mobile_api:app --host 127.0.0.1 --port 8600 \
  >> "$LOG/server.log" 2>&1 &
sleep 6
curl -s -o /dev/null --max-time 15 http://127.0.0.1:8600/ || { echo "server failed — see $LOG/server.log"; exit 1; }

: > "$LOG/tunnel.log"
nohup /root/cloudflared tunnel --url http://127.0.0.1:8600 --no-autoupdate \
  > "$LOG/tunnel.log" 2>&1 &
URL=""
for i in $(seq 1 25); do
  sleep 2
  URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$LOG/tunnel.log" | tail -1)
  [ -n "$URL" ] && break
done
[ -n "$URL" ] || { echo "tunnel failed — see $LOG/tunnel.log"; exit 1; }

# daily backup loop (keeps 30) + weekly self-maintenance loop
nohup bash -c "exec -a wisdom_backup_loop bash -c 'while :; do
  cp $DB /root/wisdom-live/backups/wisdom-\$(date +%F).db 2>/dev/null
  ls -t /root/wisdom-live/backups/wisdom-*.db 2>/dev/null | tail -n +31 | xargs -r rm
  sleep 86400; done'" >/dev/null 2>&1 &
nohup bash -c "exec -a wisdom_maintain_loop bash -c 'while :; do sleep 604800
  cd $REPO && WISDOM_DB_PATH=$DB $VENVBIN/python scripts/maintain.py --scrape 2 >> $LOG/maintain.log 2>&1
  done'" >/dev/null 2>&1 &

STABLE=""
if [ -n "${NETLIFY_AUTH_TOKEN:-}" ] && [ -n "${NETLIFY_SITE_ID:-}" ]; then
  bash "$REPO/deploy/android/update_netlify_redirect.sh" "$URL" && STABLE="(stable link updated)"
fi
echo
echo "  LINK:  $URL   $STABLE"
echo "  CODE:  $CODE"
