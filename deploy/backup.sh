#!/usr/bin/env bash
# Nightly database backup — keeps 30 days. Cron: 15 3 * * * /path/to/deploy/backup.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p backups
docker compose cp wisdom:/data/wisdom.db "backups/wisdom-$(date +%F).db"
ls -t backups/wisdom-*.db | tail -n +31 | xargs -r rm
