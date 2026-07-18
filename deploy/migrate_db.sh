#!/usr/bin/env bash
# Run FROM YOUR LAPTOP: copies the live database (all proverbs + annotations) to the VPS.
#   ./migrate_db.sh <server-ip> [db-path]
set -euo pipefail
IP="${1:?usage: migrate_db.sh <server-ip> [db-path]}"
DB="${2:-$HOME/CSML/Conferinte/ConsIRL/ConsILR2025/wisdom-extractor/live/wisdom.db}"
[ -f "$DB" ] || { echo "database not found: $DB"; exit 1; }
scp "$DB" root@"$IP":/tmp/wisdom.db
ssh root@"$IP" 'cd /root/wisdom-extractor/deploy \
  && docker compose stop wisdom mobile \
  && docker compose cp /tmp/wisdom.db wisdom:/data/wisdom.db \
  && docker compose start wisdom mobile \
  && rm /tmp/wisdom.db && echo "✓ database migrated, services restarted"'
