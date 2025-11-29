#!/usr/bin/env bash
set -euo pipefail

# Wrapper to download SA-V archives listed in data/sa-v.txt
# Usage:
#   ./download_sa_v.sh [INPUT_TSV] [OUTPUT_DIR] [CONCURRENCY]
# Defaults:
#   INPUT_TSV   = data/sa-v.txt
#   OUTPUT_DIR  = data/sa-v/
#   CONCURRENCY = 1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="$SCRIPT_DIR/download_sa1b.sh"

if [[ ! -x "$BASE_SCRIPT" ]]; then
  echo "Base downloader not found or not executable: $BASE_SCRIPT" >&2
  echo "Make sure data/download_sa1b.sh exists and is executable." >&2
  exit 1
fi

INPUT_TSV="${1:-"$SCRIPT_DIR/sa-v-1p.txt"}"
OUTPUT_DIR="${2:-"$SCRIPT_DIR/sa-v"}"
CONCURRENCY="${3:-1}"

exec "$BASE_SCRIPT" "$INPUT_TSV" "$OUTPUT_DIR" "$CONCURRENCY"


