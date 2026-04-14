#!/usr/bin/env bash
# run_all.sh — Reproducible end-to-end pipeline
# All random states fixed at 42.
#
# Usage:
#   bash run_all.sh
#
# Prerequisites:
#   1. Virtual env activated: source env/bin/activate
#   2. GEE authenticated: earthengine authenticate
#   3. Set GEE_PROJECT in scripts/01b_gee_extract.py
#   4. S2 patches downloaded from Google Drive to data/raw/s2_patches/
#      (run steps 1 and 2 manually, wait for GEE exports, then resume at step 3)
#
# Flags:
#   --skip-gee     Skip GEE export submission (patches already downloaded)
#   --skip-chirps  Skip CHIRPS download (already extracted)

set -euo pipefail
SKIP_GEE=${1:-""}
SKIP_CHIRPS=${2:-""}

echo "=== Step 1: Download GROW-Africa labels ==="
python scripts/01_download.py

echo ""
echo ""
echo "=== Step 1b: Stratified sample for GEE export (~8K observations) ==="
python scripts/01e_sample.py

echo ""
echo "=== Step 2: Extract Sentinel-2 patches via GEE ==="
if [[ "$SKIP_GEE" == "--skip-gee" ]]; then
    echo "Skipping GEE export (--skip-gee set). Verifying existing patches..."
    python scripts/01b_gee_extract.py --verify
else
    for COUNTRY in Rwanda Kenya Tanzania Malawi Nigeria; do
        echo "  Submitting GEE exports for $COUNTRY..."
        python scripts/01b_gee_extract.py --country "$COUNTRY"
    done
    echo ""
    echo "GEE exports submitted. Download GeoTIFFs from Google Drive to data/raw/s2_patches/"
    echo "Then re-run: bash run_all.sh --skip-gee"
    exit 0
fi

echo ""
echo "=== Step 3: Extract CHIRPS rainfall features ==="
if [[ "$SKIP_CHIRPS" == "--skip-chirps" ]]; then
    echo "Skipping CHIRPS extraction (--skip-chirps set)."
else
    python scripts/01c_chirps.py
fi

echo ""
echo "=== Step 4: Build master dataset ==="
# Add --cropland-mask-dir /path/to/dea_cropland if you have the DEA cropland mask
python scripts/02_preprocess.py

echo ""
echo "=== Step 5: Extract embeddings ==="
python scripts/03_extract_embeddings.py

echo ""
echo "=== Step 6: Train and evaluate (LOCO CV + random CV) ==="
python scripts/04_train_eval.py

echo ""
echo "=== Step 7: Generate figures ==="
python scripts/05_figures.py

echo ""
echo "Pipeline complete. Results in results/metrics/ and results/figures/"
