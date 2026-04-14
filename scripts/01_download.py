"""
Phase 1.1 — Download and filter GROW-Africa yield labels from Zenodo.

Source: doi:10.5281/zenodo.14961637
Target: data/raw/grow_africa_raw.parquet

Filters applied:
  - Date range 2017–2022 (Sentinel-2 reliable from 2017)
  - Point-level GPS observations only (no admin-level polygon centroids)
  - Maize as primary crop
  - Countries with sufficient coverage for LOCO CV (see CANDIDATE_COUNTRIES)

GPS jitter note: LSMS-ISA coordinates are jittered 0–5 km for privacy.
Cropland mask filtering is applied in 02_preprocess.py, not here.

Run:
    python scripts/01_download.py

Outputs:
    data/raw/grow_africa_raw.parquet
    data/raw/sample_counts.csv
"""

import os
import sys
import requests
import zipfile
import io
import openpyxl
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

ZENODO_RECORD = "14961637"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"

# Countries targeted for LOCO CV — must each have ≥100 point observations
# after filtering. Adjust after running the sample count check below.
CANDIDATE_COUNTRIES = ["Rwanda", "Kenya", "Tanzania", "Malawi", "Nigeria"]

PRIMARY_CROP = "maize"
YEAR_MIN = 2017
YEAR_MAX = 2022
MIN_OBSERVATIONS_PER_COUNTRY = 100  # drop country if below this after filtering


def fetch_zenodo_files(record_id: str) -> list[dict]:
    """Return file metadata from the Zenodo record."""
    resp = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=30)
    resp.raise_for_status()
    return resp.json()["files"]


def download_grow_africa() -> pd.DataFrame:
    """
    Download GROW-Africa from Zenodo, load into a DataFrame, and return it.
    Handles both CSV and parquet formats.
    """
    print(f"Fetching Zenodo record {ZENODO_RECORD}...")
    files = fetch_zenodo_files(ZENODO_RECORD)

    # Find the main data file — prefer parquet, fall back to CSV
    target = None
    for f in files:
        name = f["key"].lower()
        if name.endswith(".parquet") and "grow" in name:
            target = f
            break
    if target is None:
        for f in files:
            name = f["key"].lower()
            if name.endswith(".csv") and "grow" in name:
                target = f
                break
    if target is None:
        # Fall back to the first file that isn't a README
        for f in files:
            if not f["key"].lower().endswith(".md"):
                target = f
                break

    if target is None:
        raise FileNotFoundError(
            "Could not identify the main GROW-Africa data file in the Zenodo record. "
            "Check the record manually at https://zenodo.org/records/14961637 and "
            "download the data file to data/raw/ manually, then re-run with --skip-download."
        )

    url = target["links"]["self"]
    fname = target["key"]
    local_path = RAW_DIR / fname

    if local_path.exists():
        print(f"  Already downloaded: {local_path.name} — skipping.")
    else:
        print(f"  Downloading {fname} ({target['size'] / 1e6:.1f} MB)...")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(local_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        print(f"  Saved to {local_path}")

    # Load
    if fname.endswith(".parquet"):
        df = pd.read_parquet(local_path)
    elif fname.endswith(".zip"):
        with zipfile.ZipFile(local_path) as zf:
            all_names = [n for n in zf.namelist() if not n.startswith("__MACOSX")]

            # Load GPS-level files only: Point + LSMS_cropcut + LSMS_survey.
            # Exclude Regional.xlsx — it is admin-level (no GPS) and has a
            # completely different schema that causes duplicate columns on concat.
            GPS_FILES = ("point", "lsms_cropcut", "lsms_survey")
            data_files = [
                n for n in all_names
                if (n.endswith(".xlsx") or n.endswith(".csv"))
                and any(tag in n.lower() for tag in GPS_FILES)
            ]
            if not data_files:
                raise ValueError(
                    f"ZIP contains no usable data file (.xlsx or .csv). Contents: {all_names}"
                )

            frames = []
            for entry in data_files:
                print(f"  Reading from ZIP entry: {entry}")
                raw = zf.read(entry)
                try:
                    if entry.endswith(".xlsx"):
                        frame = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
                    else:
                        frame = pd.read_csv(io.BytesIO(raw))
                    # Tag which source file each row came from — useful for debugging
                    frame["_source_file"] = entry
                    frames.append(frame)
                    print(f"    {len(frame):,} rows loaded from {entry}")
                except Exception as e:
                    print(f"    WARNING: could not load {entry}: {e} — skipping")

            if not frames:
                raise ValueError("All files in ZIP failed to load.")

            df = pd.concat(frames, ignore_index=True)
            print(f"  Combined shape after concatenation: {df.shape}")
    else:
        df = pd.read_csv(local_path, low_memory=False)

    print(f"  Raw shape: {df.shape}")
    return df


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase column names and attempt to identify canonical columns.
    GROW-Africa column names may vary between versions — adapt as needed.
    """
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # After concat of Point + LSMS_cropcut + LSMS_survey, each file contributes
    # different column names for the same concept (e.g. Latitude vs GPS_lat).
    # Coalesce: fill the canonical column from all aliases in priority order.
    aliases = {
        "crop_name":  ["crop", "crop_type", "cropname"],
        "yield_kgha": ["yield_ton_ha_", "yield_kg_ha", "yield_kg/ha",
                       "grain_yield", "yield_t_ha", "yieldtonha"],
        "lat":        ["latitude", "gps_lat", "y"],
        "lon":        ["longitude", "gps_lon", "x"],
        "year":       ["harvestyear", "agyearend", "agyearstart",
                       "harvest_year", "survey_year", "crop_year"],
        "country":    ["country_name", "adm0_name"],
        "obs_type":   ["spatialprecision_km_", "observation_type",
                       "data_type", "spatial_type"],
    }

    for canonical, options in aliases.items():
        # Start with the canonical column if it already exists, else empty
        if canonical in df.columns:
            base = df[canonical]
        else:
            base = pd.Series(pd.NA, index=df.index)
        # Fill NaN from each alias column in priority order
        for alt in options:
            if alt in df.columns:
                base = base.combine_first(df[alt])
        df[canonical] = base

    # If 'country' is still missing, try to infer it from the source file name.
    # GROW-Africa LSMS files are typically named GROW-Africa_LSMS_Ethiopia.xlsx etc.
    if "country" not in df.columns and "_source_file" in df.columns:
        def _infer_country(src: str) -> str:
            for candidate in [
                "Nigeria", "Ethiopia", "Tanzania", "Uganda", "Malawi",
                "Kenya", "Ghana", "Zambia", "Rwanda", "Mali",
            ]:
                if candidate.lower() in src.lower():
                    return candidate
            return "Unknown"
        df["country"] = df["_source_file"].apply(_infer_country)
        inferred = df["country"].value_counts().to_dict()
        print(f"  Inferred 'country' from filename: {inferred}")

    return df


def filter_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply study-scope filters and report counts at each step."""
    n0 = len(df)
    print(f"\nFilter steps (starting n={n0:,}):")

    # 1. Year range
    if "year" in df.columns:
        df = df[df["year"].between(YEAR_MIN, YEAR_MAX)]
        print(f"  After year filter ({YEAR_MIN}–{YEAR_MAX}): {len(df):,}")
    else:
        print("  WARNING: 'year' column not found — skipping year filter.")

    # 2. Maize only
    if "crop_name" in df.columns:
        df = df[df["crop_name"].str.lower().str.contains(PRIMARY_CROP, na=False)]
        print(f"  After crop filter (maize): {len(df):,}")
    else:
        print("  WARNING: 'crop_name' column not found — skipping crop filter.")

    # 3. Point-level observations only (exclude admin centroids)
    # GROW-Africa_Point.xlsx uses SpatialPrecision_km_ (numeric) instead of a
    # string obs_type. Keep rows with precision ≤ 5 km; if already string, use regex.
    if "obs_type" in df.columns:
        if pd.api.types.is_numeric_dtype(df["obs_type"]):
            # SpatialPrecision_km_: lower = more precise; ≤5 km = field/plot-level.
            # NaN means the file had no SpatialPrecision column (e.g. LSMS_cropcut);
            # those are always field-level crop-cut measurements — treat as passing.
            df = df[df["obs_type"].isna() | (df["obs_type"] <= 5)]
            print(f"  After spatial-precision filter (<=5 km): {len(df):,}")
        else:
            point_mask = df["obs_type"].str.lower().str.contains("point|plot|field|gps", na=False)
            df = df[point_mask]
            print(f"  After point-level filter: {len(df):,}")
    else:
        print("  WARNING: 'obs_type' column not found — assuming all rows are point-level.")

    # 4. Must have valid GPS
    if "lat" in df.columns and "lon" in df.columns:
        df = df.dropna(subset=["lat", "lon"])
        df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)]
        print(f"  After GPS validity filter: {len(df):,}")
    else:
        print("  WARNING: lat/lon columns not found — cannot filter by GPS validity.")

    # 5. Must have valid yield
    if "yield_kgha" in df.columns:
        # Convert t/ha to kg/ha if values look like t/ha (median < 20)
        if df["yield_kgha"].median() < 20:
            print("  NOTE: yield values look like t/ha — converting to kg/ha (*1000).")
            df["yield_kgha"] = df["yield_kgha"] * 1000
        df = df[df["yield_kgha"] > 0]
        print(f"  After positive yield filter: {len(df):,}")
    else:
        print("  WARNING: 'yield_kgha' column not found.")

    # 6. Candidate countries only
    if "country" in df.columns:
        df["country"] = df["country"].str.strip().str.title()
        df = df[df["country"].isin(CANDIDATE_COUNTRIES)]
        print(f"  After country filter {CANDIDATE_COUNTRIES}: {len(df):,}")
    else:
        print("  WARNING: 'country' column not found — skipping country filter.")

    return df


def check_sample_counts(df: pd.DataFrame) -> list[str]:
    """Print per-country counts and return countries that meet the minimum threshold."""
    if "country" not in df.columns:
        print("Cannot compute per-country counts — 'country' column missing.")
        return []

    counts = df.groupby("country").size().sort_values(ascending=False)
    print(f"\nSample counts per country (pre cropland-mask filter):")
    for country, n in counts.items():
        flag = "" if n >= MIN_OBSERVATIONS_PER_COUNTRY else "  <-- BELOW THRESHOLD"
        print(f"  {country}: {n}{flag}")

    viable = counts[counts >= MIN_OBSERVATIONS_PER_COUNTRY].index.tolist()
    dropped = [c for c in CANDIDATE_COUNTRIES if c not in viable]
    if dropped:
        print(f"\n  Countries dropped (< {MIN_OBSERVATIONS_PER_COUNTRY} obs): {dropped}")
        print("  Consider adding HarvestStat Africa records or lowering threshold to 60.")
    print(f"\n  Countries retained for LOCO CV: {viable}")
    return viable


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download and load the first .parquet/.csv in data/raw/")
    args = parser.parse_args()

    if args.skip_download:
        candidates = list(RAW_DIR.glob("*.parquet")) + list(RAW_DIR.glob("*.csv"))
        if not candidates:
            sys.exit("No file found in data/raw/ — run without --skip-download first.")
        fname = candidates[0]
        print(f"Loading existing file: {fname.name}")
        df = pd.read_parquet(fname) if fname.suffix == ".parquet" else pd.read_csv(fname)
    else:
        df = download_grow_africa()

    df = normalise_columns(df)
    print(f"\nColumns after normalisation: {list(df.columns)}")

    df = filter_dataset(df)
    viable_countries = check_sample_counts(df)

    # Generate stable field_id for every row (many rows lack one from the source).
    # Format: <country_prefix>_<zero-padded index> — unique within the dataset.
    df = df.reset_index(drop=True)
    df["field_id"] = (
        df["country"].str[:3].str.lower()
        + "_"
        + df.index.astype(str).str.zfill(6)
    )

    out_path = RAW_DIR / "grow_africa_raw.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved filtered dataset: {out_path}  ({len(df):,} rows)")

    counts_df = df.groupby("country").size().reset_index(name="n_observations")
    counts_path = RAW_DIR / "sample_counts.csv"
    counts_df.to_csv(counts_path, index=False)
    print(f"Saved sample counts: {counts_path}")

    if len(viable_countries) < 3:
        print("\nWARNING: Fewer than 3 viable countries. "
              "LOCO CV requires >=3 countries to be meaningful. "
              "Consider:\n"
              "  1. Download HarvestStat Africa from Dryad and merge\n"
              "  2. Lower MIN_OBSERVATIONS_PER_COUNTRY to 60\n"
              "  3. Switch to leave-one-admin-out CV if admin-level labels are richer")


if __name__ == "__main__":
    main()
