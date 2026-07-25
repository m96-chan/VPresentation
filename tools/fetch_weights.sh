#!/usr/bin/env bash
# Download and extract the THA4 pretrained models into data/tha4/.
# Weights are ~610MB and licensed CC BY-NC 4.0 (non-commercial) — they are
# intentionally NOT committed to the repo (.gitignore). Run this once.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/data/tha4"
URL="https://www.dropbox.com/scl/fi/7wec0sur7449iqgtlpi3n/tha4-models.zip?rlkey=0f9d1djmbvjjjn09469s1adx8&dl=1"

mkdir -p "$DEST"
if [ -f "$DEST/body_morpher.pt" ] && [ -f "$DEST/upscaler.pt" ]; then
  echo "THA4 weights already present in $DEST — nothing to do."
  exit 0
fi

echo "Downloading THA4 models (~610MB) to $DEST/tha4-models.zip ..."
curl -L --fail -o "$DEST/tha4-models.zip" "$URL"

echo "Extracting ..."
unzip -o "$DEST/tha4-models.zip" -d "$DEST"

echo "Done. Expected files:"
ls -1 "$DEST"/*.pt
