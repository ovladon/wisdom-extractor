#!/usr/bin/env bash
# Pull a consistent snapshot of the LIVE server database into data/live_snapshot.db,
# so the local analysis app (app.py) shows the real, current state.
# Usage:  ./scripts/pull_live_db.sh   then run the printed streamlit command.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$HERE/data/live_snapshot.db}"
ssh root@188.68.56.176 bash -s <<'RMT'
cd /root/wisdom-extractor/deploy
docker compose exec -T mobile python - <<'PY'
import sqlite3
s = sqlite3.connect("/data/wisdom.db"); d = sqlite3.connect("/data/snap.db")
s.backup(d); d.close()
PY
MP=$(docker volume inspect $(docker volume ls -q | grep wisdom_data | head -1) --format '{{.Mountpoint}}')
cp "$MP/snap.db" /root/snap.db
RMT
scp -q root@188.68.56.176:/root/snap.db "$OUT"
echo
echo "Live snapshot saved to: $OUT"
echo "See the real status locally with:"
echo "  WISDOM_DB_PATH=\"$OUT\" streamlit run app.py"
