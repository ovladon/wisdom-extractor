#!/usr/bin/env bash
# Apply gloss-review and corpus-withdrawal decisions to the LIVE server database.
#
# The Admin panel's default data source is data/live_snapshot.db — a local copy pulled
# from the server. Editing it changes nothing in production and is thrown away by the
# next refresh, which for a curation pass means hours of review silently lost. Decisions
# are therefore applied on the server first, and only mirrored locally once that
# succeeds.
#
# Over SSH rather than through an endpoint, for the same reason as push_settings.sh: an
# admin write path on a public host is a permanent attack surface.
#
# Usage:  ./scripts/push_curation.sh <base64-of-json>
#   JSON: [{"pid": 123, "action": "accept|reject|withdraw", "reason": "...", "user": "..."}]
set -euo pipefail

HOST="${WISDOM_HOST:-}"
if [ -z "$HOST" ] && [ -f "$(dirname "$0")/../deploy/host.conf" ]; then
  . "$(dirname "$0")/../deploy/host.conf"
  HOST="${WISDOM_HOST:-}"
fi
[ -n "$HOST" ] || { echo "Set WISDOM_HOST (user@host) or create deploy/host.conf" >&2; exit 1; }
PAYLOAD="${1:?usage: push_curation.sh <base64-json>}"

ssh "$HOST" bash -s -- "$PAYLOAD" <<'RMT'
set -euo pipefail
cd /root/wisdom-extractor/deploy
docker compose exec -T mobile python -c '
import base64, json, sys
from core.persistence import init_db, review_gloss, mark_excluded, edit_gloss, connect
init_db()
items = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
con = connect()
ok = bad = 0
for it in items:
    pid, action = int(it["pid"]), it["action"]
    if con.execute("SELECT 1 FROM proverbs WHERE id=?", (pid,)).fetchone() is None:
        print("  MISSING pid %d" % pid); bad += 1; continue
    if action == "accept":
        review_gloss(pid, True)
    elif action == "reject":
        review_gloss(pid, False)
    elif action == "withdraw":
        mark_excluded(pid, True, user=it.get("user"), reason=it.get("reason"))
    elif action == "restore":
        mark_excluded(pid, False)
    elif action == "edit_gloss":
        good, n = edit_gloss(pid, it.get("gloss", ""), user=it.get("user"),
                             reason=it.get("reason"))
        if not good:
            print("  REJECTED gloss edit for pid %d (empty or unchanged)" % pid)
            bad += 1; continue
        if n:
            print("  note: pid %d had %d judgment(s) on the earlier wording" % (pid, n))
    else:
        print("  UNKNOWN action %r for pid %d" % (action, pid)); bad += 1; continue
    ok += 1
con.close()
print("  applied %d, failed %d" % (ok, bad))
sys.exit(1 if bad else 0)
' "$1"
RMT
