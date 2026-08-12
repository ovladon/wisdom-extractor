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
# The values are read back and compared afterwards. An earlier version piped them into
# `python -`, which consumes stdin for the program itself, so nothing was ever read and
# the script reported success while changing nothing.
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

OUT=$(ssh "$HOST" bash -s -- "$@" <<'RMT'
set -euo pipefail
cd /root/wisdom-extractor/deploy
docker compose exec -T mobile python -c '
import sys
from core.persistence import init_db, set_setting, get_setting, SETTING_DEFAULTS
init_db()
bad = 0
for arg in sys.argv[1:]:
    key, _, value = arg.partition("=")
    if key not in SETTING_DEFAULTS:
        print("REFUSED unknown setting: " + key); bad = 1; continue
    set_setting(key, value, "admin-panel")
    got = str(get_setting(key))
    ok = got == str(value)
    print(("  OK   " if ok else "  FAIL ") + key + " = " + got)
    if not ok:
        bad = 1
sys.exit(bad)
' "$@"
RMT
)
STATUS=$?
echo "$OUT"
if [ "$STATUS" -ne 0 ]; then
  echo "!! settings were NOT applied cleanly" >&2
  exit "$STATUS"
fi
echo "Live settings updated and verified. The service picks them up within a minute."
