#!/usr/bin/env bash
# Wisdom Lab phone-server setup. Run INSIDE the proot Ubuntu on the phone:
#   curl -sL https://raw.githubusercontent.com/ovladon/wisdom-extractor/main/deploy/android/setup_phone.sh | bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git python3-venv python3-pip curl ca-certificates procps
if [ -d /root/wisdom-extractor/.git ]; then
  git -C /root/wisdom-extractor pull --ff-only
else
  git clone https://github.com/ovladon/wisdom-extractor.git /root/wisdom-extractor
fi
if [ ! -d /root/wisdom-venv ]; then python3 -m venv /root/wisdom-venv; fi
/root/wisdom-venv/bin/pip install -q --upgrade pip
/root/wisdom-venv/bin/pip install -q -r /root/wisdom-extractor/requirements-server.txt
mkdir -p /root/wisdom-live/backups
if [ ! -x /root/cloudflared ]; then
  curl -sL -o /root/cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64
  chmod +x /root/cloudflared
fi
echo
echo "=== setup complete ==="
echo "1. Copy the database to /root/wisdom-live/wisdom.db (see PHONE_SERVER.md step 5)"
echo "2. Start with:  bash /root/wisdom-extractor/deploy/android/start_phone.sh"
