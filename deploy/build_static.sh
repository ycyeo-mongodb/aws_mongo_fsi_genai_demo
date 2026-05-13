#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# Build the production version of static/index.html with the CloudFront
# path prefix baked in. Output lands in build/static/ and is uploaded to
# S3 by deploy.sh.
#
# Why: the source HTML uses `/api/...` for fetches so local dev (uvicorn
# on :8001) works as-is. Production traffic arrives via CloudFront at
# /fsi_digital_bank_demo/api/... so every fetch URL needs the prefix.
#
# This script is intentionally a single sed expression so it's easy to
# audit. If you add new fetch patterns to index.html, extend the regex.
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC="${REPO_ROOT}/static/index.html"
OUT_DIR="${REPO_ROOT}/build/static"
OUT="${OUT_DIR}/index.html"
PREFIX="${PATH_PREFIX:-/fsi_digital_bank_demo}"

if [[ ! -f "${SRC}" ]]; then
  echo "ERROR: source not found: ${SRC}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

# Rewrites — order matters; the longest patterns first to avoid double-prefixing.
#   fetch('/api/...')    →  fetch('/fsi_digital_bank_demo/api/...')
#   fetch("/api/...")    →  fetch("/fsi_digital_bank_demo/api/...")
#   '/api/...'  used inside template strings → prefix it once
#   "/api/...           → prefix it once
#
# We only touch URLs that START with /api/ so absolute external URLs
# (like https://example.com/api/) and substrings inside JSON are safe.
sed \
  -e "s|fetch('/api/|fetch('${PREFIX}/api/|g" \
  -e "s|fetch(\"/api/|fetch(\"${PREFIX}/api/|g" \
  -e "s|'/api/|'${PREFIX}/api/|g" \
  -e "s|\"/api/|\"${PREFIX}/api/|g" \
  -e "s|href=\"/api/|href=\"${PREFIX}/api/|g" \
  -e "s|src=\"/api/|src=\"${PREFIX}/api/|g" \
  "${SRC}" > "${OUT}"

# Sanity check — fail if we still see a bare /api/ that wasn't prefixed.
# (Allowed exception: comments, or URLs already prefixed.)
if grep -E "[\"'\(]/api/" "${OUT}" | grep -v "${PREFIX}/api" > /dev/null; then
  echo "WARNING: bare /api/ references still present after rewrite:" >&2
  grep -nE "[\"'\(]/api/" "${OUT}" | grep -v "${PREFIX}/api" >&2 || true
  echo "(continuing — they may be in comments or be intentional)" >&2
fi

echo "✓ Built ${OUT} with prefix ${PREFIX}"
wc -l "${OUT}"
