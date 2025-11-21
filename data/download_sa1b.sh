#!/usr/bin/env bash
# Order: INPUT_TSV, OUTPUT_DIR, CONCURRENCY(optional, default: 4)
# usage: bash data/download_sa1b.sh data/sa-1b-10p.txt data/sa-1b-10p 8
# Successfully downloaded filenames are recorded in downloaded_ok.txt

set -uo pipefail

INPUT_TSV="$1"
OUTPUT_DIR="$2"
CONCURRENCY="${3:-4}"   # Number of parallel downloads, default: 4

mkdir -p "$OUTPUT_DIR"

# File to record successfully downloaded filenames
SUCCESS_LIST="$OUTPUT_DIR/downloaded_ok.txt"
: > "$SUCCESS_LIST"   # Truncate/create the file

i=0

download_one() {
  local file_name="$1"
  local url="$2"
  local idx="$3"

  local dest="$OUTPUT_DIR/$file_name"

  echo "[$idx] START $file_name"

  # Wrap wget in an if-block so failures won't stop the main script
  if wget -c --tries=5 --timeout=30 -O "$dest" "$url"; then
    echo "[$idx] DONE  $file_name"
    echo "$file_name" >> "$SUCCESS_LIST"
  else
    echo "[$idx] FAIL  $file_name" >&2
    # Uncomment the next line if you don't want to keep partial files on failure:
    # rm -f "$dest"
  fi
}

# Read TSV line by line: file_name<TAB>cdn_link
while IFS=$'\t' read -r file_name url; do
  # Skip header
  if [[ "$file_name" == "file_name" && "$url" == "cdn_link" ]]; then
    continue
  fi

  # Skip empty lines
  [[ -z "${file_name:-}" || -z "${url:-}" ]] && continue

  download_one "$file_name" "$url" "$i" &   # Run in background (parallel)
  i=$((i+1))

  # Control concurrency: wait after every CONCURRENCY jobs
  if (( i % CONCURRENCY == 0 )); then
    wait
  fi
done < "$INPUT_TSV"

# Wait for remaining background jobs
wait

success_count=$(wc -l < "$SUCCESS_LIST" 2>/dev/null || echo 0)
echo "Total downloaded: $success_count file(s)."
echo "Success list saved to: $SUCCESS_LIST"
