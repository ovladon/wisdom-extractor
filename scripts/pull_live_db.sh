#!/usr/bin/env bash
# Pull a consistent snapshot of the LIVE server database into data/live_snapshot.db,
# so the local analysis app (app.py) shows the real, current state.
# Usage:  ./scripts/pull_live_db.sh   then run the printed streamlit command.
set -euo pipefail
# Host is supplied by the environment or deploy/host.conf (both untracked), so the
# server address is not published with the source.
HOST="${WISDOM_HOST:-}"
if [ -z "$HOST" ] && [ -f "$(dirname "$0")/../deploy/host.conf" ]; then
  . "$(dirname "$0")/../deploy/host.conf"
  HOST="${WISDOM_HOST:-}"
fi
if [ -z "$HOST" ]; then
  echo "Set WISDOM_HOST (user@host) or create deploy/host.conf" >&2; exit 1
fi
HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$HERE/data/live_snapshot.db}"
ssh "$HOST" bash -s <<'RMT'
cd /root/wisdom-extractor/deploy
docker compose exec -T mobile python - <<'PY'
import sqlite3
s = sqlite3.connect("/data/wisdom.db"); d = sqlite3.connect("/data/snap.db")
s.backup(d); d.close()
PY
MP=$(docker volume inspect $(docker volume ls -q | grep wisdom_data | head -1) --format '{{.Mountpoint}}')
cp "$MP/snap.db" /root/snap.db
RMT
scp -q "$HOST":/root/snap.db "$OUT"
echo
echo "Live snapshot saved to: $OUT"
echo "See the real status locally with:"
echo "  WISDOM_DB_PATH=\"$OUT\" streamlit run app.py"
