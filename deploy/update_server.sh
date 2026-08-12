#!/usr/bin/env bash
# Update the live server, with a health gate and an automatic rollback.
#
# The previous procedure was a bare `docker compose up -d --build`: if the new build
# started and then failed to serve, the annotation service simply stayed down until
# somebody noticed. This backs the database up first, deploys, checks that the API
# actually answers, and returns to the previous commit if it does not.
#
# Usage:  ./deploy/update_server.sh [git-ref]     (default: origin/main)
set -euo pipefail

REF="${1:-origin/main}"
HOST="${WISDOM_HOST:-}"
if [ -z "$HOST" ] && [ -f "$(dirname "$0")/host.conf" ]; then
  . "$(dirname "$0")/host.conf"
  HOST="${WISDOM_HOST:-}"
fi
[ -n "$HOST" ] || { echo "Set WISDOM_HOST (user@host) or create deploy/host.conf" >&2; exit 1; }

echo "Deploying $REF to $HOST"

ssh "$HOST" bash -s -- "$REF" <<'RMT'
set -euo pipefail
REF="$1"
cd /root/wisdom-extractor

# The server keeps local edits to deploy/Caddyfile (the real domain names). A pull that
# would clobber them should stop here rather than take the site down for a hostname.
if ! git diff --quiet -- . ':(exclude)deploy/Caddyfile'; then
  echo "!! server has uncommitted changes outside deploy/Caddyfile — refusing to deploy"
  git status --short
  exit 1
fi

PREV=$(git rev-parse HEAD)
echo "current commit: $PREV"

cd deploy
MP=$(docker volume inspect $(docker volume ls -q | grep wisdom_data | head -1) --format '{{.Mountpoint}}')
STAMP=$(date +%F-%H%M%S)
echo "backing up the database before migrating..."
docker compose exec -T mobile python - <<'PY'
import sqlite3
s = sqlite3.connect("/data/wisdom.db"); d = sqlite3.connect("/data/predeploy.db")
s.backup(d); d.close(); s.close()
print("  consistent snapshot taken")
PY
mkdir -p /root/predeploy
cp "$MP/predeploy.db" "/root/predeploy/wisdom-$STAMP.db"
ls -t /root/predeploy/wisdom-*.db | tail -n +11 | xargs -r rm
echo "  saved /root/predeploy/wisdom-$STAMP.db"

health() {
  docker compose exec -T mobile python - <<'PY' 2>/dev/null
import sys, urllib.request, json
try:
    with urllib.request.urlopen("http://127.0.0.1:8600/api/config", timeout=10) as r:
        d = json.load(r)
    sys.exit(0 if d.get("corpus", {}).get("proverbs", 0) > 0 else 1)
except Exception:
    sys.exit(1)
PY
}

deploy() {
  cd /root/wisdom-extractor
  git fetch --all --tags --quiet
  git checkout --quiet -- deploy/Caddyfile 2>/dev/null || true
  git stash push --quiet -- deploy/Caddyfile 2>/dev/null || true
  git reset --hard --quiet "$1"
  git stash pop --quiet 2>/dev/null || true
  cd deploy
  docker compose up -d --build
}

deploy "$REF"

echo "waiting for the API to answer..."
OK=0
for i in $(seq 1 30); do
  sleep 4
  if health; then OK=1; echo "  healthy after $((i*4))s"; break; fi
done

if [ "$OK" != "1" ]; then
  echo "!! new build did not serve — rolling back to $PREV"
  deploy "$PREV"
  for i in $(seq 1 30); do
    sleep 4
    if health; then echo "  rolled back and healthy"; exit 1; fi
  done
  echo "!! ROLLBACK ALSO UNHEALTHY — the database backup is /root/predeploy/wisdom-$STAMP.db"
  exit 2
fi

cd /root/wisdom-extractor
echo "deployed: $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
RMT

echo
echo "Verifying from outside..."
sleep 3
curl -s --max-time 20 https://annotate.wisdomextractor.com/api/config || {
  echo "!! public endpoint did not answer — check Caddy on the server" >&2; exit 1; }
echo
echo "Live."
