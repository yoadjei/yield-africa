"""
Phase 4b — Sensitivity and ablation analyses.

Produces four additional results files:
  results_ndvi_only.csv        — NDVI-only ablation (1-feature baseline)
  results_loco_no_nigeria.csv  — LOCO excluding Nigeria (removes proxy-label country)
  results_loco_fold_std.csv    — per-fold R²/RMSE std across LOCO countries
  label_shift_kl.csv           — pairwise KL divergence of log-yield distributions

Run:
    python scripts/04b_sensitivity.py
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import entropy
from scipy.special import rel_entr
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


def make_models():
    return {
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=[0.1, 1, 10, 100, 1000])),
        ]),
        "rf": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=100, n_jobs=-1, random_state=42)),
        ]),
        "xgb": Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8,
                n_jobs=-1, random_state=42, verbosity=0)),
        ]),
    }


def impute(X_train, X_test):
    medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    for i in range(X_train.shape[1]):
        X_train[np.isnan(X_train[:, i]), i] = medians[i]
        X_test[np.isnan(X_test[:, i]), i] = medians[i]
    return X_train, X_test


def run_loco(df, feat_cols, target, model_name):
    """Returns (agg_metrics, per_country_rows)."""
    countries = sorted(df["country"].unique())
    y_true_all, y_pred_all = [], []
    rows = []
    for held_out in countries:
        tr = df[df["country"] != held_out]
        te = df[df["country"] == held_out]
        X_tr = tr[feat_cols].values.copy()
        y_tr = tr[target].values.copy()
        X_te = te[feat_cols].values.copy()
        y_te = te[target].values.copy()
        X_tr, X_te = impute(X_tr, X_te)
        ok_tr = ~np.isnan(y_tr)
        ok_te = ~np.isnan(y_te)
        X_tr, y_tr = X_tr[ok_tr], y_tr[ok_tr]
        X_te, y_te = X_te[ok_te], y_te[ok_te]
        if len(X_te) == 0:
            continue
        pipe = make_models()[model_name]
        pipe.fit(X_tr, y_tr)
        y_pr = pipe.predict(X_te)
        m = compute_metrics(y_te, y_pr)
        m["country"] = held_out
        m["n_test"] = int(len(y_te))
        rows.append(m)
        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(y_pr.tolist())
    agg = compute_metrics(np.array(y_true_all), np.array(y_pred_all))
    return agg, rows


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_data():
    master  = pd.read_parquet(PROCESSED / "master_dataset.parquet")
    prithvi = pd.read_parquet(PROCESSED / "embeddings_prithvi.parquet")
    vit     = pd.read_parquet(PROCESSED / "embeddings_vit.parquet")
    df = master.merge(prithvi, on="field_id", how="inner")
    df = df.merge(vit, on="field_id", how="inner")
    print(f"Loaded: {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# 1. NDVI-only ablation
# ---------------------------------------------------------------------------

def ablation_ndvi(df, target):
    print("\n=== 1. NDVI-only ablation ===")
    feat_cols = ["NDVI"]
    rows = []
    for model_name in ["ridge", "rf", "xgb"]:
        agg, country_rows = run_loco(df, feat_cols, target, model_name)
        agg.update(feature="ndvi_only", model=model_name, cv="loco")
        rows.append(agg)
        for cr in country_rows:
            cr.update(feature="ndvi_only", model=model_name, cv="loco")
        r2s = [r["r2"] for r in country_rows]
        rmses = [r["rmse"] for r in country_rows]
        print(f"  {model_name}: R²={agg['r2']:.4f}  "
              f"fold_std_r2={np.std(r2s):.4f}  fold_std_rmse={np.std(rmses):.1f}")
    out = pd.DataFrame(rows)[["feature", "model", "cv", "rmse", "mae", "r2"]]
    out.to_csv(PROCESSED / "results_ndvi_only.csv", index=False)
    print(f"Saved: results_ndvi_only.csv")
    return out


# ---------------------------------------------------------------------------
# 2. Nigeria-excluded LOCO
# ---------------------------------------------------------------------------

def sensitivity_no_nigeria(df, target):
    print("\n=== 2. LOCO excluding Nigeria ===")
    df_no_ng = df[df["country"] != "Nigeria"].copy()
    feat_sets = {
        "spectral": (
            [c for c in df.columns if c.startswith("S2_")]
            + ["NDVI", "EVI", "LSWI", "NDWI"]
            + [c for c in ["chirps_total", "chirps_mean", "chirps_cv"] if c in df.columns]
        ),
        "prithvi": [c for c in df.columns if c.startswith("prithvi_")],
        "vit":     [c for c in df.columns if c.startswith("vit_")],
    }
    rows = []
    for feat_name, cols in feat_sets.items():
        for model_name in ["ridge", "rf", "xgb"]:
            agg, country_rows = run_loco(df_no_ng, cols, target, model_name)
            agg.update(feature=feat_name, model=model_name, cv="loco_no_nigeria")
            rows.append(agg)
            r2s = [r["r2"] for r in country_rows]
            print(f"  {feat_name}/{model_name}: R²={agg['r2']:.4f}  "
                  f"per-country R²={[round(r,3) for r in r2s]}")
    out = pd.DataFrame(rows)[["feature", "model", "cv", "rmse", "mae", "r2"]]
    out.to_csv(PROCESSED / "results_loco_no_nigeria.csv", index=False)
    print(f"Saved: results_loco_no_nigeria.csv")
    return out


# ---------------------------------------------------------------------------
# 3. Per-fold variability (std of per-country R² / RMSE)
# ---------------------------------------------------------------------------

def fold_variability(df, target):
    print("\n=== 3. Per-fold R² variability (LOCO) ===")
    feat_sets = {
        "spectral": (
            [c for c in df.columns if c.startswith("S2_")]
            + ["NDVI", "EVI", "LSWI", "NDWI"]
            + [c for c in ["chirps_total", "chirps_mean", "chirps_cv"] if c in df.columns]
        ),
        "prithvi": [c for c in df.columns if c.startswith("prithvi_")],
        "vit":     [c for c in df.columns if c.startswith("vit_")],
    }
    rows = []
    for feat_name, cols in feat_sets.items():
        for model_name in ["ridge", "rf", "xgb"]:
            _, country_rows = run_loco(df, cols, target, model_name)
            r2s   = np.array([r["r2"]   for r in country_rows])
            rmses = np.array([r["rmse"] for r in country_rows])
            row = {
                "feature": feat_name,
                "model": model_name,
                "mean_r2":    round(float(r2s.mean()),  4),
                "std_r2":     round(float(r2s.std()),   4),
                "min_r2":     round(float(r2s.min()),   4),
                "max_r2":     round(float(r2s.max()),   4),
                "mean_rmse":  round(float(rmses.mean()), 1),
                "std_rmse":   round(float(rmses.std()),  1),
            }
            rows.append(row)
            print(f"  {feat_name}/{model_name}: "
                  f"mean_R²={row['mean_r2']:.3f} ± {row['std_r2']:.3f}  "
                  f"range=[{row['min_r2']:.3f}, {row['max_r2']:.3f}]")
    out = pd.DataFrame(rows)
    out.to_csv(PROCESSED / "results_loco_fold_std.csv", index=False)
    print(f"Saved: results_loco_fold_std.csv")
    return out


# ---------------------------------------------------------------------------
# 4. Label-shift KL divergence
# ---------------------------------------------------------------------------

def label_shift_kl(df, target):
    print("\n=== 4. Label-shift KL divergence (log-yield) ===")
    countries = sorted(df["country"].unique())
    # estimate with Gaussian KL: KL(P||Q) = log(σQ/σP) + (σP²+(μP-μQ)²)/(2σQ²) - 1/2
    stats = {}
    for c in countries:
        vals = df.loc[df["country"] == c, target].dropna().values
        stats[c] = {"mu": float(vals.mean()), "sigma": float(vals.std()), "n": len(vals)}
        print(f"  {c}: μ={stats[c]['mu']:.4f}  σ={stats[c]['sigma']:.4f}  n={stats[c]['n']}")

    rows = []
    for src in countries:
        for tgt in countries:
            if src == tgt:
                continue
            mu_p, sig_p = stats[src]["mu"], stats[src]["sigma"]
            mu_q, sig_q = stats[tgt]["mu"], stats[tgt]["sigma"]
            # gaussian KL(P||Q)
            kl = (np.log(sig_q / sig_p)
                  + (sig_p**2 + (mu_p - mu_q)**2) / (2 * sig_q**2)
                  - 0.5)
            rows.append({
                "source_country": src,
                "target_country": tgt,
                "kl_divergence": round(float(kl), 4),
                "mean_diff_logscale": round(float(mu_p - mu_q), 4),
            })
    out = pd.DataFrame(rows)
    out.to_csv(PROCESSED / "label_shift_kl.csv", index=False)
    print(f"\nPairwise KL (source → target, Gaussian approx):")
    pivot = out.pivot(index="source_country", columns="target_country", values="kl_divergence")
    print(pivot.round(3).to_string())
    print(f"\nSaved: label_shift_kl.csv")
    return out, stats


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    df = load_data()
    target = "yield_log" if "yield_log" in df.columns else "yield_kgha"
    print(f"Target: {target}")

    ablation_ndvi(df, target)
    sensitivity_no_nigeria(df, target)
    fold_variability(df, target)
    label_shift_kl(df, target)

    print("\nDone. Update paper with numbers from:")
    print("  data/processed/results_ndvi_only.csv")
    print("  data/processed/results_loco_no_nigeria.csv")
    print("  data/processed/results_loco_fold_std.csv")
    print("  data/processed/label_shift_kl.csv")


if __name__ == "__main__":
    main()
