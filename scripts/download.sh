#!/usr/bin/env sh
# Download the ALMRRC 2021 training inputs over plain HTTPS (no AWS CLI needed).
# travel_times.json (1.8 GB) is skipped by default; pass --with-travel-times to include it.
set -e
BASE=https://amazon-last-mile-challenges.s3.amazonaws.com/almrrc2021/almrrc2021-data-training/model_build_inputs
DEST="$(dirname "$0")/../data/raw"
mkdir -p "$DEST"
FILES="route_data actual_sequences invalid_sequence_scores package_data"
[ "$1" = "--with-travel-times" ] && FILES="$FILES travel_times"
for f in $FILES; do
  echo "downloading $f.json"
  curl -sSf -C - -o "$DEST/$f.json" "$BASE/$f.json"
done
