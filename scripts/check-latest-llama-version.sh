#!/usr/bin/env bash

set -euo pipefail

API_URL="https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"

headers=(
  -H "Accept: application/vnd.github+json"
  -H "X-GitHub-Api-Version: 2022-11-28"
)

if [[ -n "${TOKEN}" ]]; then
  headers+=(-H "Authorization: Bearer ${TOKEN}")
fi

response="$(curl -fsSL "${headers[@]}" "${API_URL}")"

if command -v jq >/dev/null 2>&1; then
  version="$(jq -r '.tag_name // empty' <<<"${response}")"
else
  version="$(printf '%s' "${response}" | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n1)"
fi

if [[ -z "${version}" ]]; then
  echo "Error: unable to parse latest version from GitHub response" >&2
  exit 1
fi

echo "${version}"
