#!/usr/bin/env bash
# Apply annotation settings to the LIVE server database over SSH.
#
# The Admin panel runs on the researcher's laptop against a snapshot, so writing a
# setting there changes nothing in production. This pushes the values to the server's
# own database, which is where the annotation service reads them from.
#
# Deliberately over SSH rather than through a new authenticated endpoint: the server is
# public, and an admin write path on it would be a permanent attack surface added for
# the sake of a few knobs. The SSH trust already exists and is used by pull_live_db.sh.
#
# Usage:  ./scripts/push_settings.sh corroborate_adaptive=1 corroborate_target=0.45
set -euo pipefail

HOST="${WISDOM_HOST:-}"
if [ -z "$HOST" ] && [ -f "$(dirname "$0")/../deploy/host.conf" ]; then
  . "$(dirname "$0")/../deploy/host.conf"
  HOST="${WISDOM_HOST:-}"
fi
if [ -z "$HOST" ]; then
  echo "Set WISDOM_HOST (user@host) or create deploy/host.conf" >&2; exit 1
fi
[ "$#" -gt 0 ] || { echo "usage: push_settings.sh key=value [key=value ...]" >&2; exit 1; }

for kv in "$@"; do
  case "$kv" in
    *=*) ;;
    *) echo "not a key=value pair: $kv" >&2; exit 1 ;;
  esac
done

ssh "$HOST" bash -s -- "$@" <<'RMT'
set -euo pipefail
cd /root/wisdom-extractor/deploy
printf '%s\n' "$@" | docker compose exec -T mobile python - <<'PY'
import sys
from core.persistence import init_db, set_setting, SETTING_DEFAULTS, get_setting
init_db()
for line in sys.stdin.read().split():
    key, _, value = line.partition("=")
    if key not in SETTING_DEFAULTS:
        print(f"  refused unknown setting: {key}")
        continue
    set_setting(key, value, "admin-panel")
    print(f"  {key} = {get_setting(key)}")
PY
RMT
echo "Live settings updated. The annotation service picks them up within a minute."
