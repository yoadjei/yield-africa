"""
Phase 1.2 — Extract Sentinel-2 L2A patches via Google Earth Engine.

For each GPS-labelled field in grow_africa_raw.parquet, this script:
  1. Centres a 64×64 pixel (640 m) patch on the field GPS coordinate
  2. Composites Sentinel-2 L2A to a cloud-masked seasonal median
  3. Exports as GeoTIFF to Google Drive, batched by country

Prerequisites:
  - earthengine-api installed: pip install earthengine-api
  - Authenticated: run `earthengine authenticate` once before this script
  - GEE project created at https://earthengine.google.com
  - Set GEE_PROJECT below to your project ID

After exports complete (may take 1–2 days), download GeoTIFFs from
Google Drive to data/raw/s2_patches/<country>/ and re-run this script
with --verify to confirm all patches downloaded correctly.

Run:
    python scripts/01b_gee_extract.py --country Nigeria
    python scripts/01b_gee_extract.py --country Ethiopia
    ... (one country at a time to stay within GEE free-tier quotas)

    python scripts/01b_gee_extract.py --verify  (after downloading from Drive)

Outputs:
    data/raw/s2_patches/<country>/<field_id>_<year>.tif
    data/raw/export_log.csv  (tracks GEE task IDs and status)
"""

import os
import sys
import time
import json
import argparse
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PATCHES_DIR = RAW_DIR / "s2_patches"
PATCHES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------
GEE_PROJECT = "gee-01-492802"   # set your GEE project ID here
GDRIVE_FOLDER = "yield_africa_s2"     # Google Drive folder name for exports
PATCH_SIZE_PX = 64                     # 64 px × 10 m = 640 m patch
PATCH_SCALE_M = 10                     # Sentinel-2 native resolution

# GEE free-tier allows ~3 000 concurrent tasks. We cap each run at this
# number so we never breach the limit. Re-run the same command after
# tasks complete to submit the next batch.
BATCH_SIZE = 3000

# Save the CSV log every N successful submissions so a crash never loses
# more than this many task IDs.
SAVE_EVERY = 100

# Sentinel-2 L2A bands to export
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# Growing season windows per country (month_start, month_end inclusive)
# Source: FAOSTAT agricultural calendars. Adjust if needed.
SEASON_CALENDAR = {
    "Rwanda":    (3,  8),   # March–August (long rains)
    "Kenya":     (3,  8),   # March–August (long rains)
    "Tanzania":  (3,  8),   # March–August
    "Malawi":    (11, 4),   # November–April (wraps year boundary)
    "Nigeria":   (4,  9),   # April–September
}

CLOUD_PROB_THRESHOLD = 20  # %


def get_ee():
    """Import and initialise Earth Engine, with a clear error message."""
    try:
        import ee
    except ImportError:
        sys.exit(
            "earthengine-api not installed. Run:\n"
            "  pip install earthengine-api\n"
            "Then authenticate:\n"
            "  earthengine authenticate"
        )
    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception as e:
        sys.exit(
            f"GEE initialisation failed: {e}\n"
            "Make sure you have run `earthengine authenticate` and set GEE_PROJECT correctly."
        )
    return ee


def mask_s2_clouds(ee, image):
    """Apply QA60-based cloud mask to a Sentinel-2 L2A image."""
    qa = image.select("QA60")
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(cloud_mask).divide(10000).copyProperties(image, ["system:time_start"])


def get_season_filter(ee, year: int, country: str):
    """Return an EE date filter for the growing season of the given country and year."""
    if country not in SEASON_CALENDAR:
        raise ValueError(f"No season calendar entry for country: {country}. "
                         f"Add it to SEASON_CALENDAR.")
    m_start, m_end = SEASON_CALENDAR[country]

    if m_start <= m_end:
        # Season within single calendar year
        date_start = f"{year}-{m_start:02d}-01"
        # End of the last month
        next_m = m_end + 1
        next_y = year
        if next_m > 12:
            next_m = 1
            next_y += 1
        date_end = f"{next_y}-{next_m:02d}-01"
    else:
        # Season wraps year boundary (e.g. Malawi: Nov year → Apr year+1)
        date_start = f"{year}-{m_start:02d}-01"
        next_m = m_end + 1
        date_end = f"{year + 1}-{next_m:02d}-01"

    return ee.Filter.date(date_start, date_end)


def build_composite(ee, lon: float, lat: float, year: int, country: str):
    """
    Build a cloud-masked Sentinel-2 seasonal median composite
    for a single point location.
    """
    point = ee.Geometry.Point([lon, lat])
    season_filter = get_season_filter(ee, year, country)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filter(season_filter)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_PROB_THRESHOLD))
        .map(lambda img: mask_s2_clouds(ee, img))
        .select(S2_BANDS)
        .median()
    )

    return collection


def submit_export(ee, field_id: str, lon: float, lat: float,
                  year: int, country: str) -> str:
    """
    Submit a GEE export task. Returns the task ID.
    The task exports to Google Drive as a GeoTIFF.
    """
    composite = build_composite(ee, lon, lat, year, country)

    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(PATCH_SIZE_PX * PATCH_SCALE_M / 2).bounds()

    task = ee.batch.Export.image.toDrive(
        image=composite,
        description=f"{field_id}_{year}",
        folder=GDRIVE_FOLDER,
        fileNamePrefix=f"{country}/{field_id}_{year}",
        region=region,
        scale=PATCH_SCALE_M,
        crs="EPSG:4326",
        fileFormat="GeoTIFF",
        maxPixels=1e6,
    )
    task.start()
    return task.id


def export_country(country: str, df: pd.DataFrame, log_path: Path):
    """
    Submit GEE export tasks for all fields in a country, capped at BATCH_SIZE
    per run. The log is saved every SAVE_EVERY submissions so progress is
    never lost. Re-run the same command to pick up where it left off.
    """
    ee = get_ee()

    country_df = df[df["country"] == country].copy().reset_index(drop=True)
    if len(country_df) == 0:
        sys.exit(f"No rows found for country '{country}'. "
                 f"Available countries: {sorted(df['country'].unique())}")

    print(f"\nCountry: {country}  |  Total fields: {len(country_df)}")

    # ------------------------------------------------------------------ #
    # Load existing log — already-submitted keys are skipped              #
    # ------------------------------------------------------------------ #
    if log_path.exists():
        log = pd.read_csv(log_path)
        # Only consider rows for this country to build the skip-set
        already_ok = log[
            (log["country"] == country) & (log["status"] == "submitted")
        ]
        submitted = set(
            (already_ok["field_id"].astype(str)
             + "_"
             + already_ok["year"].astype(str)).tolist()
        )
        print(f"  Already submitted (this country): {len(submitted)}")
    else:
        log = pd.DataFrame(columns=["field_id", "year", "country", "task_id", "status"])
        submitted = set()

    # Rows that still need submitting
    pending = [
        row for _, row in country_df.iterrows()
        if f"{row['field_id']}_{int(row['year'])}" not in submitted
    ]
    total_pending = len(pending)
    print(f"  Pending (not yet submitted): {total_pending}")

    if total_pending == 0:
        print("  Nothing to submit — all fields already in the log.")
        print("  If exports are done, download from Drive and run --verify.")
        return

    # Cap this run at BATCH_SIZE
    this_batch = pending[:BATCH_SIZE]
    remaining_after = total_pending - len(this_batch)
    print(f"  Submitting this run:          {len(this_batch)}  (cap={BATCH_SIZE})")
    if remaining_after:
        print(f"  Will need another run for:    {remaining_after} more fields")
    print()

    # ------------------------------------------------------------------ #
    # Submit                                                              #
    # ------------------------------------------------------------------ #
    QUEUE_LIMIT = 3000
    QUEUE_HEADROOM = 200          # start waiting when queue > QUEUE_LIMIT - HEADROOM
    QUEUE_WAIT_SECS = 120         # how long to sleep before re-checking queue depth
    QUEUE_TARGET = 2500           # resume submitting once queue drops to this level

    def queue_depth():
        """Return number of READY + RUNNING tasks in the GEE queue."""
        try:
            tasks = ee.data.getTaskList()
            return sum(1 for t in tasks if t.get("state") in ("READY", "RUNNING"))
        except Exception:
            return 0  # if we can't check, assume it's fine

    new_rows = []
    for i, row in enumerate(this_batch, start=1):
        key = f"{row['field_id']}_{int(row['year'])}"

        # Wait if queue is nearly full before attempting submission
        depth = queue_depth()
        if depth >= QUEUE_LIMIT - QUEUE_HEADROOM:
            print(f"\n  Queue at {depth}/{QUEUE_LIMIT} — waiting for it to drain "
                  f"to {QUEUE_TARGET} before continuing...")
            while True:
                time.sleep(QUEUE_WAIT_SECS)
                depth = queue_depth()
                print(f"  Queue depth: {depth} — ", end="")
                if depth <= QUEUE_TARGET:
                    print("resuming submissions.")
                    break
                print(f"still waiting (target: {QUEUE_TARGET})...")

        try:
            task_id = submit_export(
                ee,
                str(row["field_id"]),
                float(row["lon"]),
                float(row["lat"]),
                int(row["year"]),
                country,
            )
            new_rows.append({
                "field_id": row["field_id"],
                "year": int(row["year"]),
                "country": country,
                "task_id": task_id,
                "status": "submitted",
            })
            print(f"  [{i}/{len(this_batch)}] Submitted: {key}  task_id={task_id}")
            time.sleep(0.3)  # avoid hammering GEE API
        except Exception as e:
            print(f"  [{i}/{len(this_batch)}] ERROR: {key} — {e}")
            new_rows.append({
                "field_id": row["field_id"],
                "year": int(row["year"]),
                "country": country,
                "task_id": None,
                "status": f"error: {e}",
            })

        # Periodic save so a crash doesn't wipe progress
        if i % SAVE_EVERY == 0 and new_rows:
            log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)
            log.to_csv(log_path, index=False)
            new_rows = []  # reset buffer; already flushed to log
            print(f"  -- checkpoint saved ({i} tasks logged) --")

    # Final flush
    if new_rows:
        log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)
        log.to_csv(log_path, index=False)

    n_ok = log[(log["country"] == country) & (log["status"] == "submitted")].shape[0]
    print(f"\nLog saved: {log_path}")
    print(f"  Total submitted for {country}: {n_ok}")
    if remaining_after:
        print(f"\n  {remaining_after} fields still pending.")
        print(f"  Re-run the same command after current tasks clear:")
        print(f"    python scripts/01b_gee_extract.py --country {country}")
    else:
        print(f"\n  All fields submitted!")

    print("\nMonitor at: https://code.earthengine.google.com/tasks")
    print("After all exports finish, download GeoTIFFs from Google Drive to:")
    print(f"  {PATCHES_DIR}/{country}/")
    print("Then run: python scripts/01b_gee_extract.py --verify")


def verify_patches(df: pd.DataFrame):
    """Check that a GeoTIFF patch exists for every row in the dataset."""
    missing = []
    for _, row in df.iterrows():
        country = row["country"]
        field_id = row["field_id"]
        year = int(row["year"])
        expected = PATCHES_DIR / country / f"{field_id}_{year}.tif"
        if not expected.exists():
            missing.append(str(expected))

    if missing:
        print(f"\nMissing {len(missing)} patches:")
        for p in missing[:20]:
            print(f"  {p}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        missing_path = RAW_DIR / "missing_patches.txt"
        Path(missing_path).write_text("\n".join(missing))
        print(f"\nFull list saved to {missing_path}")
    else:
        total = len(df)
        print(f"\nAll {total} patches present in {PATCHES_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", help="Country name to submit GEE exports for")
    parser.add_argument("--verify", action="store_true",
                        help="Verify all expected patches are downloaded")
    args = parser.parse_args()

    # Prefer the stratified sample (from 01e_sample.py) for GEE exports.
    # Fall back to the full dataset if the sample hasn't been generated yet.
    sampled_path = RAW_DIR / "grow_africa_sampled.parquet"
    full_path = RAW_DIR / "grow_africa_raw.parquet"
    if sampled_path.exists():
        data_path = sampled_path
        print(f"Using stratified sample: {data_path.name}")
    elif full_path.exists():
        data_path = full_path
        print(f"WARNING: sampled parquet not found — using full dataset ({full_path.name}).")
        print("Run 01e_sample.py first for a manageable GEE export workload.")
    else:
        sys.exit(f"Run 01_download.py then 01e_sample.py first. Expected: {sampled_path}")

    df = pd.read_parquet(data_path)

    log_path = RAW_DIR / "export_log.csv"

    if args.verify:
        verify_patches(df)
    elif args.country:
        export_country(args.country, df, log_path)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/01b_gee_extract.py --country Nigeria")
        print("  python scripts/01b_gee_extract.py --verify")


if __name__ == "__main__":
    main()
