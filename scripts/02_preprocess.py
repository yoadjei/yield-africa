"""
Phase 2 — Build the master dataset.

Merges:
  - GROW-Africa filtered labels (from 01_download.py)
  - Sentinel-2 band medians extracted from patch GeoTIFFs (from 01b_gee_extract.py)
  - CHIRPS rainfall features (from 01c_chirps.py)

Computes spectral indices: NDVI, EVI, LSWI, NDWI

Applies quality filters:
  - Drop rows where S2 patch cloud cover > 20%
  - Drop rows that failed the cropland mask filter
  - Drop yield outliers beyond 3 SD from country mean
  - Log-transform yield if distribution is right-skewed (skewness > 1)

Outputs:
    data/processed/master_dataset.parquet

Run:
    python scripts/02_preprocess.py [--cropland-mask-dir path/to/dea_cropland/]

Digital Earth Africa cropland mask:
  Product: "crop_mask" from Digital Earth Africa
  Download via GEE or DEA STAC API.
  Expected as a GeoTIFF covering your study area at ≥10 m resolution.
  If not available, the script runs without this filter and flags the limitation.
"""

import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from scipy.stats import skew
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PATCHES_DIR = RAW_DIR / "s2_patches"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Band order in the exported GeoTIFFs — must match 01b_gee_extract.py S2_BANDS
BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# Quality thresholds
MAX_CLOUD_COVER_PCT = 20       # drop patch if estimated cloud cover exceeds this
MIN_CROPLAND_RADIUS_M = 500    # radius to check cropland fraction
MIN_CROPLAND_FRACTION = 0.50   # drop observation if cropland < this within radius
YIELD_OUTLIER_SD = 3           # drop observations beyond this many SDs from country mean


# ---------------------------------------------------------------------------
# S2 patch loading
# ---------------------------------------------------------------------------

def load_patch_bands(patch_path: Path) -> tuple[dict[str, float], float]:
    """
    Load median band values and estimate cloud fraction from a GeoTIFF patch.

    Returns:
        band_values: dict mapping band name to median reflectance (float)
        cloud_fraction: fraction of pixels masked (proxy for cloud cover)
    """
    with rasterio.open(patch_path) as src:
        band_values = {}
        masked_counts = []
        total_pixels = None

        for i, band_name in enumerate(BAND_NAMES, start=1):
            if i > src.count:
                band_values[band_name] = np.nan
                continue
            data = src.read(i, masked=True)
            if total_pixels is None:
                total_pixels = data.size
            masked_counts.append(data.mask.sum())
            valid = data.compressed()
            band_values[band_name] = float(np.median(valid)) if len(valid) > 0 else np.nan

        cloud_fraction = max(masked_counts) / total_pixels if total_pixels else 1.0

    return band_values, cloud_fraction


def compute_spectral_indices(bands: dict[str, float]) -> dict[str, float]:
    """Compute NDVI, EVI, LSWI, NDWI from band median values."""
    B2 = bands.get("B2", np.nan)
    B3 = bands.get("B3", np.nan)
    B4 = bands.get("B4", np.nan)
    B8 = bands.get("B8", np.nan)
    B11 = bands.get("B11", np.nan)

    def safe_ratio(num, den):
        if np.isnan(num) or np.isnan(den) or (num + den) == 0:
            return np.nan
        return (num - den) / (num + den)

    ndvi = safe_ratio(B8, B4)

    if not any(np.isnan(v) for v in [B8, B4, B2]):
        denom = B8 + 6 * B4 - 7.5 * B2 + 1
        evi = 2.5 * (B8 - B4) / denom if denom != 0 else np.nan
    else:
        evi = np.nan

    lswi = safe_ratio(B8, B11)
    ndwi = safe_ratio(B3, B8)

    return {"NDVI": ndvi, "EVI": evi, "LSWI": lswi, "NDWI": ndwi}


# ---------------------------------------------------------------------------
# Cropland mask filter
# ---------------------------------------------------------------------------

def check_cropland_fraction(lon: float, lat: float,
                             mask_dir: Path | None) -> float | None:
    """
    Return the fraction of pixels classified as cropland within
    MIN_CROPLAND_RADIUS_M of the given GPS point.

    Returns None if no cropland mask is available (caller decides how to handle).
    """
    if mask_dir is None:
        return None

    tif_files = list(mask_dir.glob("*.tif"))
    if not tif_files:
        return None

    # Use the first tile that covers the point
    for tif in tif_files:
        try:
            with rasterio.open(tif) as src:
                # Check if point falls within this tile's bounds
                bounds = src.bounds
                if not (bounds.left <= lon <= bounds.right and
                        bounds.bottom <= lat <= bounds.top):
                    continue

                # Buffer ~radius pixels around the point
                pixel_size_m = abs(src.transform.a)
                radius_px = int(MIN_CROPLAND_RADIUS_M / pixel_size_m)
                row, col = src.index(lon, lat)
                row_min = max(0, row - radius_px)
                row_max = min(src.height, row + radius_px + 1)
                col_min = max(0, col - radius_px)
                col_max = min(src.width, col + radius_px + 1)

                window = rasterio.windows.Window(
                    col_min, row_min,
                    col_max - col_min, row_max - row_min
                )
                data = src.read(1, window=window)
                # DEA cropland mask: 1 = cropland, 0 = non-cropland
                cropland_frac = (data == 1).sum() / data.size
                return float(cropland_frac)
        except Exception:
            continue

    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_master_dataset(df: pd.DataFrame,
                          chirps_df: pd.DataFrame,
                          mask_dir: Path | None) -> pd.DataFrame:
    rows = []
    n_cloud_dropped = 0
    n_cropland_dropped = 0

    print(f"\nBuilding master dataset from {len(df)} observations...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        field_id = row["field_id"]
        year = int(row["year"])
        country = row["country"]
        lat = float(row["lat"])
        lon = float(row["lon"])

        patch_path = PATCHES_DIR / country / f"{field_id}_{year}.tif"

        if not patch_path.exists():
            # Patch not yet downloaded — skip silently (will appear as missing)
            continue

        # Load S2 bands and estimate cloud cover
        band_values, cloud_frac = load_patch_bands(patch_path)
        cloud_pct = cloud_frac * 100

        if cloud_pct > MAX_CLOUD_COVER_PCT:
            n_cloud_dropped += 1
            continue

        # Cropland mask filter
        cropland_frac = check_cropland_fraction(lon, lat, mask_dir)
        quality_flag = "pass"
        if cropland_frac is not None and cropland_frac < MIN_CROPLAND_FRACTION:
            n_cropland_dropped += 1
            continue
        elif cropland_frac is None:
            quality_flag = "no_cropland_mask"

        # Spectral indices
        indices = compute_spectral_indices(band_values)

        # CHIRPS features
        chirps_row = chirps_df[chirps_df["field_id"] == field_id]
        if len(chirps_row) > 0:
            chirps_total = chirps_row.iloc[0]["chirps_total"]
            chirps_mean = chirps_row.iloc[0]["chirps_mean"]
            chirps_cv = chirps_row.iloc[0]["chirps_cv"]
        else:
            chirps_total = chirps_mean = chirps_cv = np.nan

        record = {
            "field_id": field_id,
            "country": country,
            "lat": lat,
            "lon": lon,
            "year": year,
            "yield_kgha": float(row["yield_kgha"]),
            "cloud_pct": round(cloud_pct, 1),
            "cropland_frac": cropland_frac,
            "quality_flag": quality_flag,
            "s2_patch_path": str(patch_path),
            **{f"S2_{b}": band_values.get(b, np.nan) for b in BAND_NAMES},
            **indices,
            "chirps_total": chirps_total,
            "chirps_mean": chirps_mean,
            "chirps_cv": chirps_cv,
        }
        rows.append(record)

    master = pd.DataFrame(rows)
    print(f"\n  Dropped (cloud cover > {MAX_CLOUD_COVER_PCT}%): {n_cloud_dropped}")
    print(f"  Dropped (cropland fraction < {MIN_CROPLAND_FRACTION}): {n_cropland_dropped}")
    print(f"  Rows after cloud + cropland filters: {len(master)}")
    return master


def apply_yield_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Remove yield outliers and apply log transform if skewed."""
    n_before = len(df)

    # Outlier removal: per country, drop beyond 3 SD
    df = df.copy()
    outlier_mask = pd.Series(False, index=df.index)
    for country, group in df.groupby("country"):
        mean = group["yield_kgha"].mean()
        sd = group["yield_kgha"].std()
        outliers = (group["yield_kgha"] - mean).abs() > YIELD_OUTLIER_SD * sd
        outlier_mask.loc[group.index[outliers]] = True

    n_outliers = outlier_mask.sum()
    df = df[~outlier_mask]
    print(f"  Yield outliers removed (>{YIELD_OUTLIER_SD} SD from country mean): {n_outliers}")

    # Log transform if skewed
    yield_skew = skew(df["yield_kgha"].dropna())
    print(f"  Yield skewness: {yield_skew:.3f}")
    if abs(yield_skew) > 1.0:
        print("  Applying log1p transform to yield (skewness > 1).")
        df["yield_log"] = np.log1p(df["yield_kgha"])
        df["yield_transform"] = "log1p"
    else:
        df["yield_log"] = df["yield_kgha"]
        df["yield_transform"] = "none"

    print(f"  Final row count: {len(df)} (removed {n_before - len(df)} total)")
    return df


def report_sample_counts(df: pd.DataFrame):
    """Print per-country sample counts and flag countries below threshold."""
    counts = df.groupby("country").size().sort_values(ascending=False)
    print(f"\nFinal sample counts per country:")
    for country, n in counts.items():
        flag = "  <-- DROP from study" if n < 80 else ""
        print(f"  {country}: {n}{flag}")

    low = counts[counts < 80].index.tolist()
    if low:
        print(f"\nWARNING: Countries with < 80 observations will be dropped from LOCO CV: {low}")
        print("  Consider supplementing with HarvestStat Africa data.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cropland-mask-dir", default=None,
        help="Path to directory containing Digital Earth Africa cropland mask GeoTIFFs."
             " If omitted, cropland mask filter is skipped (noted in quality_flag column)."
    )
    args = parser.parse_args()

    # Load inputs
    labels_path = RAW_DIR / "grow_africa_final.parquet"
    chirps_path = PROCESSED_DIR / "chirps_features.parquet"

    if not labels_path.exists():
        sys.exit(f"Run 01_download.py first. Expected: {labels_path}")
    if not chirps_path.exists():
        print(f"WARNING: CHIRPS features not found at {chirps_path}. "
              "CHIRPS columns will be NaN. Run 01c_chirps.py to populate them.")
        chirps_df = pd.DataFrame(columns=["field_id", "chirps_total", "chirps_mean", "chirps_cv"])
    else:
        chirps_df = pd.read_parquet(chirps_path)

    df = pd.read_parquet(labels_path)

    mask_dir = Path(args.cropland_mask_dir) if args.cropland_mask_dir else None
    if mask_dir is None:
        print("No cropland mask directory provided. Cropland filter will be skipped.")
        print("Set --cropland-mask-dir to enable GPS-jitter mitigation (recommended).")

    master = build_master_dataset(df, chirps_df, mask_dir)

    if len(master) == 0:
        sys.exit(
            "Master dataset is empty after filters.\n"
            "Check that S2 patches have been downloaded to data/raw/s2_patches/ "
            "by running 01b_gee_extract.py --verify"
        )

    master = apply_yield_quality_filters(master)
    report_sample_counts(master)

    out_path = PROCESSED_DIR / "master_dataset.parquet"
    master.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(master)} rows, {master.shape[1]} columns)")

    # Summary stats for the paper
    summary = master.groupby("country").agg(
        n=("yield_kgha", "count"),
        yield_mean=("yield_kgha", "mean"),
        yield_sd=("yield_kgha", "std"),
        yield_min=("yield_kgha", "min"),
        yield_max=("yield_kgha", "max"),
        ndvi_mean=("NDVI", "mean"),
        chirps_total_mean=("chirps_total", "mean"),
    ).round(2)
    summary_path = PROCESSED_DIR / "dataset_summary.csv"
    summary.to_csv(summary_path)
    print(f"Saved summary stats: {summary_path}")
    print(f"\n{summary.to_string()}")


if __name__ == "__main__":
    main()
