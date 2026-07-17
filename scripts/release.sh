#!/usr/bin/env bash
# Release helper: ./scripts/release.sh <version> "<one-line summary>"
# Updates VERSION, commits everything, tags, and builds a versioned zip in dist/.
# Detailed notes go in CHANGELOG.md (edit it BEFORE running this).
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION="${1:?usage: release.sh <version> \"summary\"}"
SUMMARY="${2:?usage: release.sh <version> \"summary\"}"
echo "$VERSION" > VERSION
git add -A
git commit -m "v$VERSION: $SUMMARY"
git tag -a "v$VERSION" -m "$SUMMARY"
mkdir -p dist
git archive --format=zip -o "dist/wisdom-extractor-v$VERSION.zip" HEAD
echo "Committed, tagged v$VERSION, built dist/wisdom-extractor-v$VERSION.zip"
echo "Push with: git push origin HEAD --tags"
