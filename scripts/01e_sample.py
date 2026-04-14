"""
Phase 1.5 — Stratified sample of grow_africa_raw.parquet for GEE export.

Target: ~8,000 observations (~1,600 per country), stratified by
yield quantile (5 bins) x year so the yield distribution and temporal
coverage are both preserved within each country.

Why sample:
  - GEE free-tier cap is ~3,000 concurrent tasks
  - Full 48K dataset = ~85 batch runs, weeks of wall time, ~78 GB storage
  - 8K sample is sufficient for a conference paper with LOCO CV

Stratification method:
  - Per country: assign each row to a (yield_quantile, year) cell
  - Sample uniformly from each cell; cells with fewer rows than the quota
    are taken fully and the remainder is drawn from the largest cells
  - Final per-country count is capped at TARGET_PER_COUNTRY

Run:
    python scripts/01e_sample.py

Inputs:
    data/raw/grow_africa_raw.parquet

Outputs:
    data/raw/grow_africa_sampled.parquet  (~8K rows)
    data/raw/sample_report.csv            (per-stratum counts for inspection)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

INPUT_PATH = RAW_DIR / "grow_africa_raw.parquet"
OUTPUT_PATH = RAW_DIR / "grow_africa_sampled.parquet"
REPORT_PATH = RAW_DIR / "sample_report.csv"

TARGET_TOTAL = 8_000
N_YIELD_BINS = 5     # yield quantile bins (quintiles)
RANDOM_SEED = 42


def stratified_sample_country(df_country: pd.DataFrame, target_n: int, seed: int) -> pd.DataFrame:
    """
    Sample `target_n` rows from one country using yield-quantile x year strata.

    Algorithm:
      1. Bin yield into N_YIELD_BINS quintiles (labels 0-4).
      2. Form strata as (yield_bin, year) combinations.
      3. Allocate quota evenly across strata (floor division), with remainder
         distributed to the largest strata.
      4. For each stratum, take min(quota, stratum_size) rows at random.
      5. If total is still below target_n (small strata couldn't fill quota),
         top-up by sampling uniformly from the remaining rows.
    """
    rng = np.random.default_rng(seed)
    df = df_country.copy().reset_index(drop=True)

    if len(df) <= target_n:
        return df  # country is smaller than target — take everything

    # 1. Yield quantile bins (per-country so bins reflect local distribution)
    df["_yield_bin"] = pd.qcut(
        df["yield_kgha"], q=N_YIELD_BINS, labels=False, duplicates="drop"
    )
    years = sorted(df["year"].dropna().unique())
    strata_keys = [(yb, yr) for yb in df["_yield_bin"].dropna().unique() for yr in years]

    # 2. Base quota per stratum
    n_strata = len(strata_keys)
    base_quota = target_n // n_strata
    remainder = target_n - base_quota * n_strata

    # 3. Sort strata by size descending so remainder goes to largest
    strata_sizes = {
        k: len(df[(df["_yield_bin"] == k[0]) & (df["year"] == k[1])])
        for k in strata_keys
    }
    sorted_keys = sorted(strata_keys, key=lambda k: strata_sizes[k], reverse=True)

    selected_indices = []
    for i, key in enumerate(sorted_keys):
        yb, yr = key
        quota = base_quota + (1 if i < remainder else 0)
        stratum_idx = df[(df["_yield_bin"] == yb) & (df["year"] == yr)].index.tolist()
        if len(stratum_idx) == 0:
            continue
        n_take = min(quota, len(stratum_idx))
        chosen = rng.choice(stratum_idx, size=n_take, replace=False)
        selected_indices.extend(chosen.tolist())

    # 4. Top-up if we're short (rare: only if many strata had 0 rows)
    if len(selected_indices) < target_n:
        remaining_pool = list(set(df.index.tolist()) - set(selected_indices))
        shortfall = target_n - len(selected_indices)
        extra = rng.choice(remaining_pool, size=min(shortfall, len(remaining_pool)), replace=False)
        selected_indices.extend(extra.tolist())

    result = df.loc[sorted(selected_indices)].drop(columns=["_yield_bin"])
    return result


def main():
    if not INPUT_PATH.exists():
        sys.exit(f"Input not found: {INPUT_PATH}\nRun 01_download.py first.")

    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df):,} rows from grow_africa_raw.parquet")

    countries = sorted(df["country"].unique())
    n_countries = len(countries)
    target_per_country = TARGET_TOTAL // n_countries
    remainder_countries = TARGET_TOTAL - target_per_country * n_countries

    print(f"\nTarget: {TARGET_TOTAL:,} total  |  {n_countries} countries")
    print(f"Base per country: {target_per_country}  |  remainder slots: {remainder_countries}")

    # Countries sorted by size ascending so remainder goes to the smallest
    # (compensates for countries that can't fill their quota)
    country_sizes = df.groupby("country").size().sort_values()
    frames = []
    report_rows = []

    for i, country in enumerate(country_sizes.index):
        target_n = target_per_country + (1 if i < remainder_countries else 0)
        country_df = df[df["country"] == country]
        sampled = stratified_sample_country(country_df, target_n, seed=RANDOM_SEED + i)
        frames.append(sampled)

        # Report: yield stats before and after
        for yr in sorted(country_df["year"].dropna().unique()):
            n_before = len(country_df[country_df["year"] == yr])
            n_after = len(sampled[sampled["year"] == yr])
            report_rows.append({
                "country": country, "year": int(yr),
                "n_original": n_before, "n_sampled": n_after,
            })

        pct = len(sampled) / len(country_df) * 100
        print(f"  {country:12s}: {len(country_df):6,} -> {len(sampled):5,} ({pct:.1f}%)")

    sampled_df = pd.concat(frames, ignore_index=True)

    # Sanity checks
    assert sampled_df["field_id"].nunique() == len(sampled_df), \
        "Duplicate field_ids in sample — check field_id generation"
    assert sampled_df["yield_kgha"].isna().sum() == 0, \
        "NaN yield_kgha in sample"
    assert sampled_df[["lat", "lon"]].isna().any().any() == False, \
        "NaN GPS in sample"

    sampled_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}  ({len(sampled_df):,} rows)")

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(REPORT_PATH, index=False)
    print(f"Saved: {REPORT_PATH}")

    # Summary table
    print("\nFinal counts per country:")
    counts = sampled_df.groupby("country").agg(
        n=("field_id", "count"),
        yield_p10=("yield_kgha", lambda x: x.quantile(0.1)),
        yield_median=("yield_kgha", "median"),
        yield_p90=("yield_kgha", lambda x: x.quantile(0.9)),
        year_min=("year", "min"),
        year_max=("year", "max"),
    )
    print(counts.to_string())

    print("\nYield distribution preserved? (compare original vs sample medians)")
    orig_medians = df.groupby("country")["yield_kgha"].median().rename("orig_median")
    samp_medians = sampled_df.groupby("country")["yield_kgha"].median().rename("samp_median")
    comparison = pd.concat([orig_medians, samp_medians], axis=1)
    comparison["pct_diff"] = (
        (comparison["samp_median"] - comparison["orig_median"]) / comparison["orig_median"] * 100
    ).round(1)
    print(comparison.to_string())
    print("\nDone. Use grow_africa_sampled.parquet for GEE export (01b_gee_extract.py).")


if __name__ == "__main__":
    main()
