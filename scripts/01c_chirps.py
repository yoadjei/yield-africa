"""
Phase 1.3 — Extract CHIRPS seasonal rainfall features for each field.

For each field in grow_africa_raw.parquet, extracts monthly precipitation
from CHIRPS v2.0 raster files and aggregates to three seasonal features:
  - chirps_total:  seasonal total precipitation (mm)
  - chirps_mean:   mean monthly precipitation (mm/month)
  - chirps_cv:     coefficient of variation across months (dimensionless)

The season window used matches 01b_gee_extract.py's SEASON_CALENDAR.

CHIRPS download:
  Monthly GeoTIFF files at 0.05° resolution (~5.5 km):
  https://www.chc.ucsb.edu/data/chirps
  Direct path: https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_monthly/tifs/

  Files are named: chirps-v2.0.YYYY.MM.tif.gz

  This script downloads only the months required for your field-year observations.
  Total download is typically 200–800 MB depending on year/month coverage.

Run:
    python scripts/01c_chirps.py

Outputs:
    data/raw/chirps/       (downloaded monthly CHIRPS rasters)
    data/processed/chirps_features.parquet
"""

import os
import sys
import gzip
import shutil
import requests
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CHIRPS_DIR = RAW_DIR / "chirps"
PROCESSED_DIR = ROOT / "data" / "processed"
CHIRPS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CHIRPS_BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_monthly/tifs"

# Must match 01b_gee_extract.py
SEASON_CALENDAR = {
    "Nigeria":   (4,  9),
    "Ethiopia":  (6, 11),
    "Tanzania":  (3,  8),
    "Uganda":    (3,  8),
    "Malawi":    (11, 4),
}


def season_months(year: int, country: str) -> list[tuple[int, int]]:
    """
    Return list of (year, month) tuples covering the growing season
    for a given country and harvest year.
    """
    m_start, m_end = SEASON_CALENDAR[country]
    months = []
    if m_start <= m_end:
        for m in range(m_start, m_end + 1):
            months.append((year, m))
    else:
        # Wraps year boundary: season starts in year-1
        for m in range(m_start, 13):
            months.append((year - 1, m))
        for m in range(1, m_end + 1):
            months.append((year, m))
    return months


def chirps_filename(year: int, month: int) -> str:
    return f"chirps-v2.0.{year}.{month:02d}.tif"


def download_chirps_month(year: int, month: int) -> Path:
    """Download a single CHIRPS monthly raster if not already present."""
    fname = chirps_filename(year, month)
    local = CHIRPS_DIR / fname
    if local.exists():
        return local

    gz_fname = fname + ".gz"
    url = f"{CHIRPS_BASE_URL}/{gz_fname}"
    gz_local = CHIRPS_DIR / gz_fname

    print(f"  Downloading {gz_fname}...")
    with requests.get(url, stream=True, timeout=120) as r:
        if r.status_code == 404:
            raise FileNotFoundError(
                f"CHIRPS file not found: {url}\n"
                "Check that the year/month is within CHIRPS coverage (1981–present)."
            )
        r.raise_for_status()
        with open(gz_local, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)

    with gzip.open(gz_local, "rb") as f_in, open(local, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_local.unlink()
    return local


def extract_pixel_value(tif_path: Path, lon: float, lat: float) -> float:
    """Extract the pixel value at (lon, lat) from a raster file."""
    with rasterio.open(tif_path) as src:
        row, col = src.index(lon, lat)
        window = rasterio.windows.Window(col, row, 1, 1)
        data = src.read(1, window=window)
        nodata = src.nodata
        val = data[0, 0]
        if nodata is not None and val == nodata:
            return np.nan
        return float(val)


def extract_chirps_for_field(lon: float, lat: float, year: int, country: str) -> dict:
    """
    Download required CHIRPS files and extract pixel values for
    the growing season. Returns dict with chirps_total, chirps_mean, chirps_cv.
    """
    months = season_months(year, country)
    values = []
    for y, m in months:
        try:
            tif = download_chirps_month(y, m)
            val = extract_pixel_value(tif, lon, lat)
            values.append(val)
        except Exception as e:
            print(f"    Warning: could not extract CHIRPS ({y}-{m:02d}) for "
                  f"({lat:.3f}, {lon:.3f}): {e}")
            values.append(np.nan)

    values = np.array(values, dtype=float)
    valid = values[~np.isnan(values)]

    if len(valid) == 0:
        return {"chirps_total": np.nan, "chirps_mean": np.nan, "chirps_cv": np.nan}

    total = float(valid.sum())
    mean = float(valid.mean())
    cv = float(valid.std() / mean) if mean > 0 else np.nan
    return {"chirps_total": total, "chirps_mean": mean, "chirps_cv": cv}


def main():
    data_path = RAW_DIR / "grow_africa_raw.parquet"
    if not data_path.exists():
        sys.exit(f"Run 01_download.py first. Expected: {data_path}")

    df = pd.read_parquet(data_path)
    required = ["field_id", "lat", "lon", "year", "country"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        sys.exit(f"Missing columns in grow_africa_raw.parquet: {missing_cols}")

    results = []
    print(f"Extracting CHIRPS features for {len(df)} fields...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        country = row["country"]
        if country not in SEASON_CALENDAR:
            print(f"  No season calendar for {country} — skipping field {row['field_id']}")
            results.append({
                "field_id": row["field_id"],
                "chirps_total": np.nan,
                "chirps_mean": np.nan,
                "chirps_cv": np.nan,
            })
            continue

        features = extract_chirps_for_field(
            lon=float(row["lon"]),
            lat=float(row["lat"]),
            year=int(row["year"]),
            country=country,
        )
        features["field_id"] = row["field_id"]
        results.append(features)

    out_df = pd.DataFrame(results)
    nan_count = out_df[["chirps_total", "chirps_mean", "chirps_cv"]].isna().any(axis=1).sum()
    print(f"\nExtraction complete. Fields with any NaN CHIRPS value: {nan_count}/{len(out_df)}")

    out_path = PROCESSED_DIR / "chirps_features.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
