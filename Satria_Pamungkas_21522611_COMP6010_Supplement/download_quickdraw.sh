#!/usr/bin/env bash
# Acquire the Quick, Draw! source data for COMP6010 Part C (option C2).
#
# The .npy files are ~100 MB each and are therefore NOT included in this ZIP, per the
# assignment's instruction not to ship large public datasets. This script downloads them.
#
# Source     : Google Quick, Draw! Dataset, "numpy_bitmap" format (28x28 uint8 grayscale)
# Repository : https://github.com/googlecreativelab/quickdraw-dataset
# Licence    : CC BY 4.0 (Google Creative Lab). Attribution required.
# Retrieved  : 10 September 2026
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data/quickdraw"
mkdir -p "$DIR"
BASE="https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
for CAT in cat dog fish; do
  if [ -s "$DIR/$CAT.npy" ]; then
    echo "$CAT.npy already present, skipping"
  else
    echo "downloading $CAT.npy ..."
    curl -fSL -o "$DIR/$CAT.npy" "$BASE/$CAT.npy"
  fi
done
echo
echo "Expected SHA-256 checksums are recorded in results/partC/C_data_manifest.json"
echo "(field: categories[].sha256). Verify with:  sha256sum $DIR/*.npy"
ls -la "$DIR"
