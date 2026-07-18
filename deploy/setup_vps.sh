#!/usr/bin/env bash
# One-command Wisdom Lab VPS bootstrap (fresh Ubuntu 24.04, run as root):
#
#   curl -sL https://raw.githubusercontent.com/ovladon/wisdom-extractor/main/deploy/setup_vps.sh \
#     | bash -s -- wisdom.YOURDOMAIN annotate.YOURDOMAIN YOURCODE
#
# Installs docker, clones/updates the repo, writes Caddyfile+.env, starts everything,
# and installs the cron jobs (nightly backup, weekly scrape+digest). Idempotent.
set -euo pipefail
PORTAL_DOMAIN="${1:?usage: setup_vps.sh <portal-domain> <mobile-domain> <annotator-code>}"
MOBILE_DOMAIN="${2:?usage: setup_vps.sh <portal-domain> <mobile-domain> <annotator-code>}"
CODE="${3:?usage: setup_vps.sh <portal-domain> <mobile-domain> <annotator-code>}"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq docker.io docker-compose-v2 git curl
systemctl enable --now docker

if [ -d /root/wisdom-extractor/.git ]; then
  git -C /root/wisdom-extractor pull --ff-only
else
  git clone https://github.com/ovladon/wisdom-extractor.git /root/wisdom-extractor
fi
cd /root/wisdom-extractor/deploy

cat > Caddyfile <<CADDY
$PORTAL_DOMAIN {
    reverse_proxy wisdom:8501
}
$MOBILE_DOMAIN {
    reverse_proxy mobile:8600
}
CADDY
echo "ANNOTATOR_CODE=$CODE" > .env

docker compose up -d --build

( crontab -l 2>/dev/null | grep -v wisdom-extractor || true
  echo "15 3 * * * cd /root/wisdom-extractor/deploy && ./backup.sh # wisdom-extractor"
  echo "0 4 * * 0 cd /root/wisdom-extractor/deploy && docker compose exec -T wisdom python scripts/maintain.py --scrape 2 # wisdom-extractor"
) | crontab -

echo
echo "=== Wisdom Lab deployed ==="
docker compose ps
echo
echo "Portal:  https://$PORTAL_DOMAIN"
echo "Mobile:  https://$MOBILE_DOMAIN   (share this one + the code)"
echo "Next: FROM YOUR LAPTOP run  deploy/migrate_db.sh <this-server-ip>  to import your existing database."
