# Yield Africa — Reproducibility Package

**Paper:** *Do Foundation Model Embeddings Improve Cross-Country Crop Yield Generalisation? A Leave-One-Country-Out Evaluation in Sub-Saharan Africa*

**Author:** Yaw Osei Adjei, Department of Computer Science, KNUST, Ghana

---

## Overview

This repository contains the full code pipeline to reproduce all experiments and figures in the paper. The pipeline evaluates 18 experimental conditions (3 feature sets × 3 regressors × 2 CV schemes) for smallholder maize yield prediction across Kenya, Malawi, Nigeria, Rwanda, and Tanzania.

**Key result:** All LOCO R² are negative. Frozen Prithvi-EO embeddings do not outperform 10-band Sentinel-2 spectral features for cross-country yield prediction.

---

## Repository Structure

```
yield_africa/
├── scripts/
│   ├── 01_download.py          # download GROW-Africa labels from Zenodo
│   ├── 01b_gee_extract.py      # export S2 patches via Google Earth Engine
│   ├── 01c_chirps.py           # extract CHIRPS rainfall features
│   ├── 01d_harveststat.py      # merge HarvestStat Africa (Nigeria coverage)
│   ├── 01e_sample.py           # stratified sampling for GEE export
│   ├── 02_preprocess.py        # build master_dataset.parquet
│   ├── 03_extract_embeddings.py # extract Prithvi-EO and ViT-Base embeddings
│   ├── 04_train_eval.py        # train + evaluate all 18 conditions
│   ├── 05_figures.py           # generate all paper figures
│   └── prithvi_mae.py          # Prithvi-EO model architecture (from HF repo)
├── data/
│   ├── raw/                    # raw downloads (not tracked by git)
│   └── processed/              # results_all.csv, results_loco_country.csv (tracked)
├── figures/                    # all 6 paper figures (PDF)
├── models/                     # Prithvi model weights (not tracked — download below)
├── paper/
│   ├── main.tex                # LaTeX source
│   └── references.bib          # BibTeX references
├── requirements.txt
└── run_all.sh                  # end-to-end pipeline script
```

---

## Quick Reproduction (Results Only)

If you only want to reproduce the figures from the pre-computed results:

```bash
git clone https://github.com/yoadjei/yield-africa.git
cd yield-africa
python -m venv env && source env/bin/activate   # Windows: env\Scripts\activate
pip install -r requirements.txt
python scripts/05_figures.py
```

All figures are saved to `figures/`. The processed CSVs (`results_all.csv`, `results_loco_country.csv`) are included in the repository.

---

## Full Pipeline Reproduction

### Step 0 — Environment

```bash
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ required. GPU optional but strongly recommended for Steps 3 (embedding extraction).

### Step 1 — Download Yield Labels

```bash
python scripts/01_download.py
```

Downloads GROW-Africa from [Zenodo doi:10.5281/zenodo.14961637](https://doi.org/10.5281/zenodo.14961637). No account required.

For Nigeria coverage, also run:

```bash
python scripts/01d_harveststat.py
```

Downloads HarvestStat Africa from [Dryad doi:10.5061/dryad.vq83bk42w](https://doi.org/10.5061/dryad.vq83bk42w).

### Step 2 — Sentinel-2 Patches via Google Earth Engine

**Prerequisites:**
- Google Earth Engine account (free for research): [signup](https://signup.earthengine.google.com/)
- Authenticate: `earthengine authenticate`
- Set your GEE project ID in `scripts/01b_gee_extract.py` (line: `GEE_PROJECT = "your-project-id"`)

```bash
python scripts/01e_sample.py        # stratified sample for export
python scripts/01b_gee_extract.py   # submit GEE export tasks
```

GEE exports take 30–90 minutes per country. Download the exported GeoTIFFs from Google Drive to `data/raw/s2_patches/<Country>/`.

### Step 3 — CHIRPS Rainfall

```bash
python scripts/01c_chirps.py
```

Downloads CHIRPS precipitation data from UCSB servers (no account required).

### Step 4 — Build Master Dataset

```bash
python scripts/02_preprocess.py
```

Outputs `data/processed/master_dataset.parquet`.

### Step 5 — Download Prithvi-EO Model Weights

```bash
mkdir -p models
# Download from Hugging Face (requires git-lfs)
wget -O models/Prithvi_EO_V1_100M.pt \
  "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M/resolve/main/Prithvi_EO_V1_100M.pt"
```

File size: ~453 MB.

### Step 6 — Extract Embeddings

```bash
python scripts/03_extract_embeddings.py
```

Extracts 768-dim CLS tokens from Prithvi-EO and ViT-Base for all 6,404 fields.
Checkpoints every 50 batches — safe to interrupt and resume.
GPU runtime: ~2 hours (CPU: ~8 hours).

Outputs:
- `data/processed/embeddings_prithvi.parquet`
- `data/processed/embeddings_vit.parquet`

### Step 7 — Train and Evaluate

```bash
python scripts/04_train_eval.py
```

Runs all 18 conditions. CPU runtime: ~20 minutes.

Outputs:
- `data/processed/results_all.csv`
- `data/processed/results_loco_country.csv`

### Step 8 — Generate Figures

```bash
python scripts/05_figures.py
```

Outputs 6 PDFs to `figures/`.

---

## Or Run Everything at Once

```bash
bash run_all.sh --skip-gee --skip-chirps   # if patches + CHIRPS already downloaded
```

---

## Data Access Summary

| Data | Source | Access |
|---|---|---|
| GROW-Africa yield labels | [Zenodo 14961637](https://doi.org/10.5281/zenodo.14961637) | Open |
| HarvestStat Africa | [Dryad vq83bk42w](https://doi.org/10.5061/dryad.vq83bk42w) | Open |
| Sentinel-2 imagery | Google Earth Engine | Free (research account) |
| CHIRPS rainfall | UCSB Climate Hazards Group | Open |
| Prithvi-EO weights | [HuggingFace ibm-nasa-geospatial](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M) | Open |
| ViT-Base weights | HuggingFace `google/vit-base-patch16-224` | Open |

---

## Pre-Computed Results

The following files are tracked in git and allow figure reproduction without re-running the pipeline:

| File | Description |
|---|---|
| `data/processed/results_all.csv` | 18-condition results table |
| `data/processed/results_loco_country.csv` | Per-country LOCO breakdown (45 rows) |

---

## Figures

| Figure | File | Description |
|---|---|---|
| 1 | `fig1_loco_r2_heatmap.pdf` | LOCO R² heatmap: feature × model |
| 2 | `fig2_random_vs_loco.pdf` | Within-country vs cross-country R² |
| 3 | `fig3_loco_country_rmse.pdf` | Per-country RMSE + sample sizes |
| 4 | `fig4_generalization_gap.pdf` | Generalisation gap (random − LOCO R²) |
| 5 | `fig5_naive_baseline.pdf` | Model RMSE vs naive mean-predictor per country |
| 6 | `fig6_pred_vs_actual.pdf` | Predicted vs actual scatter (Prithvi-EO/Ridge/LOCO) |

---

## Citation

```bibtex
@article{adjei2025yield,
  author  = {Adjei, Yaw Osei},
  title   = {Do Foundation Model Embeddings Improve Cross-Country Crop Yield Generalisation?
             A Leave-One-Country-Out Evaluation in Sub-Saharan Africa},
  year    = {2026},
  journal = {under review}
}
```

---

## Licence

Code: MIT. Data: subject to respective source licences (see links above).
