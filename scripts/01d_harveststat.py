"""
Phase 1.1b — Download HarvestStat Africa and merge with GROW-Africa labels.

Source: doi:10.5061/dryad.vq83bk42w
Paper:  Lee et al. (2025), Scientific Data, doi:10.1038/s41597-025-05001-z

IMPORTANT — spatial resolution difference:
  GROW-Africa provides GPS coordinates per individual farm plot (jittered 0–5 km).
  HarvestStat Africa provides yield statistics per admin unit (level 1 or 2).
  HarvestStat rows are assigned the CENTROID of their admin polygon as a proxy
  GPS coordinate. This is coarser than GROW-Africa and must be disclosed in the
  paper's limitations section.

  A 'label_source' column distinguishes rows in the merged dataset:
    'grow_africa_point'   — original GPS field observation
    'harveststat_admin'   — admin-centroid proxy (HarvestStat)

Primary use: add Nigeria and Ethiopia coverage, which have 0 rows in
GROW-Africa_Point.xlsx but are well-represented in HarvestStat.

Run AFTER 01_download.py:
    python scripts/01d_harveststat.py

Outputs:
    data/raw/harveststat_raw.csv           (filtered HarvestStat records)
    data/raw/grow_africa_raw.parquet       (overwritten with merged dataset)
    data/raw/sample_counts.csv             (updated counts)

Admin boundary centroids:
  This script uses the FEWS NET admin boundary shapefiles. These are available
  from the FEWS NET Data Warehouse. If the Dryad record includes shapefiles,
  they are used automatically. Otherwise, the script falls back to a lookup
  table of pre-computed country + admin1 centroids for the target countries.
  The fallback is less accurate — replace with actual shapefiles if possible.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DRYAD_DOI = "10.5061/dryad.vq83bk42w"

PRIMARY_CROP = "maize"
YEAR_MIN = 2017
YEAR_MAX = 2022

# Countries we want from HarvestStat that are absent or thin in GROW-Africa.
# Rwanda, Kenya, Tanzania, Malawi have abundant GPS data in GROW-Africa Point;
# Nigeria's GPS data lives in LSMS_cropcut (now correctly included after the
# SpatialPrecision NaN fix in 01_download.py), so HarvestStat is no longer
# needed. Run 01d_harveststat.py only if per-country counts remain below 100
# after running 01_download.py.
HARVESTSTAT_TARGET_COUNTRIES = ["Nigeria"]

# Fallback centroid lookup for admin level 1 regions.
# Used only if no shapefile is available. Coordinates are approximate.
# Extend this table if you add more countries.
ADMIN1_CENTROID_FALLBACK = {
    # Nigeria — 37 states
    ("Nigeria", "Benue"):           (7.34, 8.74),
    ("Nigeria", "Kano"):            (11.99, 8.52),
    ("Nigeria", "Kaduna"):          (10.52, 7.44),
    ("Nigeria", "Niger"):           (10.00, 5.60),
    ("Nigeria", "Plateau"):         (9.22, 9.02),
    ("Nigeria", "Taraba"):          (8.00, 10.78),
    ("Nigeria", "Kebbi"):           (12.45, 4.20),
    ("Nigeria", "Sokoto"):          (13.06, 5.24),
    ("Nigeria", "Bauchi"):          (10.31, 9.84),
    ("Nigeria", "Gombe"):           (10.29, 11.17),
    ("Nigeria", "Adamawa"):         (9.33, 12.40),
    ("Nigeria", "Borno"):           (11.85, 13.15),
    ("Nigeria", "Yobe"):            (12.30, 11.44),
    ("Nigeria", "Zamfara"):         (12.17, 6.23),
    ("Nigeria", "Kogi"):            (7.80, 6.74),
    ("Nigeria", "Nasarawa"):        (8.54, 8.30),
    # Ethiopia — 11 regions
    ("Ethiopia", "Oromia"):         (7.55, 40.00),
    ("Ethiopia", "Amhara"):         (11.34, 37.59),
    ("Ethiopia", "SNNPR"):          (6.90, 37.54),
    ("Ethiopia", "Tigray"):         (14.00, 38.47),
    ("Ethiopia", "Somali"):         (6.60, 44.00),
    ("Ethiopia", "Afar"):           (12.00, 41.00),
    ("Ethiopia", "Benishangul-Gumuz"): (10.75, 35.57),
    ("Ethiopia", "Gambela"):        (8.25, 34.59),
    ("Ethiopia", "Harari"):         (9.31, 42.14),
    ("Ethiopia", "Dire Dawa"):      (9.60, 41.86),
    ("Ethiopia", "Addis Ababa"):    (9.01, 38.76),
}


HARVESTSTAT_CSV_NAME = "hvstat_africa_data_v1.0.csv"
HARVESTSTAT_GPKG_NAME = "hvstat_africa_boundary_v1.0.gpkg"


def download_harveststat() -> tuple[pd.DataFrame, Path | None]:
    """
    Download HarvestStat Africa from Dryad. Returns (dataframe, gpkg_path).

    Dryad requires a browser session for direct file downloads — this script
    checks for local copies first. If not found, prints manual download
    instructions and exits.

    Manual steps:
      1. Visit https://datadryad.org/dataset/doi:10.5061/dryad.vq83bk42w
      2. Download hvstat_africa_data_v1.0.csv  -> data/raw/
      3. Optionally download hvstat_africa_boundary_v1.0.gpkg -> data/raw/
      4. Re-run this script.
    """
    csv_local = RAW_DIR / HARVESTSTAT_CSV_NAME
    gpkg_local = RAW_DIR / HARVESTSTAT_GPKG_NAME

    if not csv_local.exists():
        print(
            f"\nHarvestStat CSV not found at: {csv_local}\n"
            "\nDryad requires a browser login for file downloads. Please:\n"
            "  1. Visit https://datadryad.org/dataset/doi:10.5061/dryad.vq83bk42w\n"
            f"  2. Download {HARVESTSTAT_CSV_NAME}  ->  data/raw/\n"
            f"  3. Optionally download {HARVESTSTAT_GPKG_NAME}  ->  data/raw/\n"
            "  4. Re-run: python scripts/01d_harveststat.py\n"
        )
        sys.exit(1)

    print(f"Loading {csv_local.name} ...")
    df = pd.read_csv(csv_local, low_memory=False)
    print(f"  Loaded {len(df):,} rows")

    gpkg_path = gpkg_local if gpkg_local.exists() else None
    if gpkg_path:
        print(f"  Found boundary file: {gpkg_local.name}")
    else:
        print(f"  No boundary file found — will use fallback centroid table.")

    return df, gpkg_path



def get_admin_centroid(country: str, admin1: str, gpkg_path: Path | None) -> tuple[float, float] | None:
    """
    Return (lon, lat) centroid for the given admin unit.
    Tries geopandas + GeoPackage first, falls back to lookup table.
    """
    if gpkg_path is not None and gpkg_path.exists():
        try:
            import geopandas as gpd
            gdf = gpd.read_file(gpkg_path)
            gdf.columns = [c.lower() for c in gdf.columns]
            country_cols = [c for c in gdf.columns if "country" in c or "adm0" in c]
            admin_cols = [c for c in gdf.columns if "admin1" in c or "adm1" in c or "name_1" in c]
            if country_cols and admin_cols:
                mask = (
                    gdf[country_cols[0]].str.lower().str.contains(country.lower(), na=False)
                    & gdf[admin_cols[0]].str.lower().str.contains(admin1.lower(), na=False)
                )
                matched = gdf[mask]
                if len(matched) > 0:
                    centroid = matched.geometry.centroid.iloc[0]
                    return float(centroid.x), float(centroid.y)
        except Exception as e:
            print(f"  GeoPackage centroid lookup failed ({e}) — using fallback table.")

    # Fallback: exact match on lookup table
    key = (country, admin1)
    if key in ADMIN1_CENTROID_FALLBACK:
        lon, lat = ADMIN1_CENTROID_FALLBACK[key]
        return lon, lat

    # Fuzzy match on lookup table
    for (c, a), (lon, lat) in ADMIN1_CENTROID_FALLBACK.items():
        if c.lower() == country.lower() and admin1.lower() in a.lower():
            return lon, lat

    return None


def filter_harveststat(df: pd.DataFrame) -> pd.DataFrame:
    """Filter HarvestStat to maize, 2017–2022, target countries, yield records only."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    n0 = len(df)
    print(f"\nHarvestStat filter steps (starting n={n0:,}):")

    # Crop filter — column may be 'product' or 'crop*'
    crop_col = next((c for c in df.columns if c == "product" or "crop" in c), None)
    if crop_col:
        df = df[df[crop_col].str.lower().str.contains("maize|corn", na=False)]
        print(f"  After maize filter: {len(df):,}")
    else:
        print("  WARNING: no crop/product column found")

    # Year filter
    year_col = next((c for c in df.columns if "harvest_year" in c or c == "year"), None)
    if year_col:
        df = df[df[year_col].between(YEAR_MIN, YEAR_MAX)]
        print(f"  After year filter ({YEAR_MIN}-{YEAR_MAX}): {len(df):,}")
    else:
        print("  WARNING: no year column found")

    # Country filter — normalise names (e.g. "Tanzania, United Republic of" -> "Tanzania")
    COUNTRY_ALIASES = {
        "Tanzania, United Republic Of": "Tanzania",
        "Tanzania, United Republic of": "Tanzania",
        "Congo, The Democratic Republic Of The": "DRC",
    }
    country_col = next((c for c in df.columns if c == "country"), None)
    if country_col:
        df["country"] = df[country_col].str.strip().str.title().replace(COUNTRY_ALIASES)
        target = HARVESTSTAT_TARGET_COUNTRIES + [c for c in ["Tanzania", "Uganda", "Malawi"]
                                                  if c not in HARVESTSTAT_TARGET_COUNTRIES]
        df = df[df["country"].isin(HARVESTSTAT_TARGET_COUNTRIES)]
        print(f"  After country filter {HARVESTSTAT_TARGET_COUNTRIES}: {len(df):,}")
    else:
        print("  WARNING: no country column found")

    # Yield value column
    value_col = next((c for c in df.columns if c in ("yield", "value", "yield_value")), None)
    if value_col:
        df = df.dropna(subset=[value_col])
        df = df[df[value_col] > 0]
        print(f"  After positive yield filter: {len(df):,}")
    else:
        print("  WARNING: no yield value column found")

    return df, year_col, crop_col, value_col


def build_harveststat_rows(df: pd.DataFrame,
                            year_col: str,
                            value_col: str,
                            gpkg_path: Path | None) -> pd.DataFrame:
    """
    Convert filtered HarvestStat records to the same schema as GROW-Africa.
    Assigns admin centroid as proxy GPS.
    """
    admin1_col = next((c for c in df.columns if "admin1" in c or "adm1" in c), None)
    admin2_col = next((c for c in df.columns if "admin2" in c or "adm2" in c), None)

    rows = []
    no_centroid = 0

    for _, row in df.iterrows():
        country = row["country"]
        admin1 = str(row[admin1_col]) if admin1_col and pd.notna(row.get(admin1_col)) else ""
        admin2 = str(row[admin2_col]) if admin2_col and pd.notna(row.get(admin2_col)) else ""

        # Use finest available admin unit for centroid
        admin_name = admin2 if admin2 else admin1
        result = get_admin_centroid(country, admin_name, gpkg_path)
        if result is None and admin1:
            result = get_admin_centroid(country, admin1, gpkg_path)
        if result is None:
            no_centroid += 1
            continue

        lon, lat = result
        yield_val = float(row[value_col])

        # HarvestStat yields may be in kg/ha or t/ha — check units column
        unit_col = next((c for c in df.columns if "unit" in c), None)
        if unit_col and pd.notna(row.get(unit_col)):
            unit = str(row[unit_col]).lower()
            if "t/ha" in unit or "tonne" in unit or "ton" in unit:
                yield_val *= 1000  # convert to kg/ha

        # If no unit info and yield looks like t/ha (< 20), convert
        if yield_val < 20:
            yield_val *= 1000

        rows.append({
            "field_id": f"hs_{country[:3].lower()}_{admin1[:6].lower().replace(' ', '')}_{int(row[year_col])}",
            "country": country,
            "lat": lat,
            "lon": lon,
            "year": int(row[year_col]),
            "yield_kgha": yield_val,
            "admin1": admin1,
            "admin2": admin2,
            "label_source": "harveststat_admin",
            "_source_file": "harveststat",
        })

    print(f"\n  Rows with centroid assigned: {len(rows)}")
    print(f"  Rows dropped (no centroid in lookup): {no_centroid}")
    if no_centroid > 0:
        print("  Extend ADMIN1_CENTROID_FALLBACK in this script to recover them.")

    return pd.DataFrame(rows)


def merge_with_grow_africa(harveststat_df: pd.DataFrame) -> pd.DataFrame:
    grow_path = RAW_DIR / "grow_africa_raw.parquet"
    if not grow_path.exists():
        sys.exit(f"Run 01_download.py first. Expected: {grow_path}")

    grow_df = pd.read_parquet(grow_path)

    # Tag GROW-Africa rows if not already tagged
    if "label_source" not in grow_df.columns:
        grow_df["label_source"] = "grow_africa_point"
    if "admin1" not in grow_df.columns:
        grow_df["admin1"] = ""
    if "admin2" not in grow_df.columns:
        grow_df["admin2"] = ""

    # Align columns
    all_cols = list(dict.fromkeys(list(grow_df.columns) + list(harveststat_df.columns)))
    grow_df = grow_df.reindex(columns=all_cols)
    harveststat_df = harveststat_df.reindex(columns=all_cols)

    merged = pd.concat([grow_df, harveststat_df], ignore_index=True)
    return merged


def main():
    df_raw, gpkg_path = download_harveststat()

    df_filtered, year_col, crop_col, value_col = filter_harveststat(df_raw)

    if len(df_filtered) == 0:
        sys.exit(
            "No HarvestStat records survived filtering. "
            "Check column names by inspecting data/raw/hvstat_africa_data_v1.0.csv manually."
        )

    # Save raw filtered records for reference
    raw_out = RAW_DIR / "harveststat_raw.csv"
    df_filtered.to_csv(raw_out, index=False)
    print(f"\nSaved filtered HarvestStat records: {raw_out}  ({len(df_filtered):,} rows)")

    hs_rows = build_harveststat_rows(df_filtered, year_col, value_col, gpkg_path)

    if len(hs_rows) == 0:
        sys.exit(
            "No HarvestStat rows could be assigned centroids. "
            "Extend ADMIN1_CENTROID_FALLBACK or provide shapefiles."
        )

    print(f"\nHarvestStat rows by country:")
    print(hs_rows.groupby("country").size().to_string())

    merged = merge_with_grow_africa(hs_rows)

    print(f"\nMerged dataset shape: {merged.shape}")
    print(f"\nSample counts by country and source:")
    print(merged.groupby(["country", "label_source"]).size().to_string())

    # Save merged dataset back to grow_africa_raw.parquet
    out_path = RAW_DIR / "grow_africa_raw.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"\nSaved merged dataset: {out_path}  ({len(merged):,} rows)")

    counts_df = merged.groupby("country").size().reset_index(name="n_observations")
    counts_path = RAW_DIR / "sample_counts.csv"
    counts_df.to_csv(counts_path, index=False)
    print(f"Updated sample counts: {counts_path}")

    print("\n--- Final per-country counts ---")
    for _, r in counts_df.sort_values("n_observations", ascending=False).iterrows():
        flag = "" if r["n_observations"] >= 100 else "  <-- BELOW THRESHOLD"
        print(f"  {r['country']}: {r['n_observations']}{flag}")

    print("\nNOTE FOR PAPER:")
    print("  HarvestStat rows use admin-unit centroids as proxy GPS, not field-level GPS.")
    print("  Report this in Section 3 (Data) and the Limitations section.")
    print("  Use 'label_source' column to ablate HarvestStat-only rows if needed.")


if __name__ == "__main__":
    main()
