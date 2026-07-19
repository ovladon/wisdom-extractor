#!/usr/bin/env bash
# Point the permanent Netlify redirect-site at the current tunnel URL.
# Env: NETLIFY_AUTH_TOKEN, NETLIFY_SITE_ID (in /root/.netlify_env). Arg: target URL.
set -euo pipefail
TARGET="${1:?usage: update_netlify_redirect.sh <target-url>}"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
printf '/* %s 302!\n' "$TARGET" > "$TMP/_redirects"
SHA=$(sha1sum "$TMP/_redirects" | cut -d' ' -f1)
DEPLOY_ID=$(curl -s -X POST "https://api.netlify.com/api/v1/sites/$NETLIFY_SITE_ID/deploys" \
  -H "Authorization: Bearer $NETLIFY_AUTH_TOKEN" -H "Content-Type: application/json" \
  -d "{\"files\":{\"/_redirects\":\"$SHA\"}}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
curl -s -X PUT "https://api.netlify.com/api/v1/deploys/$DEPLOY_ID/files/_redirects" \
  -H "Authorization: Bearer $NETLIFY_AUTH_TOKEN" -H "Content-Type: application/octet-stream" \
  --data-binary @"$TMP/_redirects" > /dev/null
echo "stable Netlify link now redirects to: $TARGET"
