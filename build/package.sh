#!/usr/bin/env bash
# Build the final submission artefacts. Safe to re-run.
#
# Copies the supplement to a staging directory before stripping the large dataset files, so the
# working tree keeps its data and the notebooks stay runnable.
set -euo pipefail
ROOT="/home/21522611/Documents/PHUIML-Assignment"
NAME="Satria_Pamungkas_21522611_COMP6010_Supplement"
SRC="$ROOT/$NAME"
STAGE="$(mktemp -d)/$NAME"

echo "== staging =="
mkdir -p "$STAGE"
rsync -a --exclude '__pycache__' --exclude '.ipynb_checkpoints' "$SRC/" "$STAGE/"

echo "== refreshing manifest copies =="
cp "$STAGE/results/partA/A_manifest.json" "$STAGE/manifest/"
cp "$STAGE/results/partB/B_manifest.json" "$STAGE/manifest/"
cp "$STAGE/results/partC/C_data_manifest.json" "$STAGE/manifest/"
cp "$STAGE/results/partC/C_reproduction_record.json" "$STAGE/manifest/"
cp "$STAGE/results/partC/C_split_summary.csv" "$STAGE/manifest/"

echo "== removing large public dataset (acquisition script is shipped instead) =="
rm -f "$STAGE/data/quickdraw/"*.npy
cat > "$STAGE/data/quickdraw/README.txt" <<'EOF'
The Quick, Draw! .npy files (cat, dog, fish; ~320 MB total) are intentionally NOT included,
per the assignment instruction not to ship large public datasets.

Run ../../download_quickdraw.sh from the supplement root to fetch them.
Expected SHA-256 checksums: results/partC/C_data_manifest.json -> categories[].sha256
EOF

echo "== redundant checkpoints =="
# the small-data run is only ever loaded at epoch 60
rm -f "$STAGE/checkpoints/small_data/ckpt_epoch005.pt" \
      "$STAGE/checkpoints/small_data/ckpt_epoch015.pt"

echo "== zipping =="
rm -f "$ROOT/$NAME.zip"
( cd "$(dirname "$STAGE")" && zip -qr "$ROOT/$NAME.zip" "$NAME" )

echo "== report =="
cp "$ROOT/report/report.pdf" "$ROOT/Satria_Pamungkas_21522611_COMP6010_Report.pdf"

echo
echo "== verifying =="
python3 - "$ROOT" "$NAME" <<'PY'
import sys, zipfile, os
root, name = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(f"{root}/{name}.zip")
names = z.namelist()
required = [
    f"{name}/README.md", f"{name}/AI_USE_LOG.md", f"{name}/requirements.txt",
    f"{name}/download_quickdraw.sh",
    f"{name}/notebooks/01_PartA_Bayesian_Network.ipynb",
    f"{name}/notebooks/02_PartB_Beta_Bernoulli.ipynb",
    f"{name}/notebooks/03_PartC_FlowMatching.ipynb",
    f"{name}/src/data_quickdraw.py", f"{name}/src/flow_model.py",
    f"{name}/src/train_flow.py", f"{name}/src/evaluate.py",
    f"{name}/checkpoints/baseline/ckpt_epoch060.pt",
    f"{name}/checkpoints/eval_classifier.pt",
]
missing = [r for r in required if r not in names]
print("missing required files:", missing or "none")
assert not missing
assert not [n for n in names if n.endswith(".npy")], "dataset leaked into the zip"
assert not [n for n in names if "__pycache__" in n or ".ipynb_checkpoints" in n]
print(f"files in zip : {len(names)}")
print(f"zip size     : {os.path.getsize(f'{root}/{name}.zip')/1e6:.1f} MB")
pdf = f"{root}/Satria_Pamungkas_21522611_COMP6010_Report.pdf"
print(f"report size  : {os.path.getsize(pdf)/1e6:.2f} MB")
PY
rm -rf "$(dirname "$STAGE")"
echo "done."
