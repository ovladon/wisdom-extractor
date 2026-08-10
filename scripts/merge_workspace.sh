#!/usr/bin/env bash
# Merge a workspace database's proverbs into the LIVE server corpus (hash-deduped),
# then run full maintenance so glosses/claims/clusters regenerate.
# Usage: ./scripts/merge_workspace.sh data/workspaces/my_collection.db
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
WS="${1:?usage: merge_workspace.sh <workspace.db>}"
scp -q "$WS" "$HOST":/root/merge_in.db
ssh "$HOST" bash -s <<'RMT'
cd /root/wisdom-extractor/deploy
MP=$(docker volume inspect $(docker volume ls -q | grep wisdom_data | head -1) --format '{{.Mountpoint}}')
cp /root/merge_in.db "$MP/merge_in.db"
docker compose exec -T mobile python - <<'PY'
import sqlite3, time
from core.persistence import connect, init_db, _hash_text
init_db()
src = sqlite3.connect("/data/merge_in.db")
have = {r[1] for r in src.execute("PRAGMA table_info(proverbs)")}
sel = [c for c in ("text","people","language","family","region","original",
                   "claim","gloss","url","first_seen","last_seen") if c in have]
con = connect(); added = 0
for r in src.execute(f"SELECT {','.join(sel)} FROM proverbs WHERE COALESCE(excluded,0)=0"):
    d = dict(zip(sel, r))
    if not (d.get("text") or "").strip():
        continue
    try:
        con.execute(f"INSERT INTO proverbs(hash,added_at,{','.join(sel)}) "
                    f"VALUES(?,?{',?'*len(sel)})",
                    (_hash_text(d["text"]), time.time(), *[d[c] for c in sel]))
        added += 1
    except sqlite3.IntegrityError:
        pass
con.commit()
print("merged new rows:", added)
PY
docker compose exec -T mobile python scripts/maintain.py | tail -5
rm -f "$MP/merge_in.db" /root/merge_in.db
RMT
echo "Merge complete — the live corpus absorbed the workspace (duplicates skipped)."
