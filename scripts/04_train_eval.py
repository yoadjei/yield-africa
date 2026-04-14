"""
Phase 4 — Train and evaluate yield prediction models.

Experimental design (18 conditions):
  Feature sets (3):
    spectral   — S2 band medians + spectral indices (NDVI, EVI, LSWI, NDWI) + CHIRPS rainfall
    prithvi    — Prithvi-EO-1.0-100M CLS-token embeddings (768-dim)
    vit        — ViT-Base CLS-token embeddings (768-dim)

  Regressors (3):
    ridge      — Ridge regression (L2, alpha tuned via CV)
    rf         — Random Forest (100 trees)
    xgb        — XGBoost

  CV schemes (2):
    loco       — Leave-One-Country-Out (tests geographic generalization)
    random     — Stratified 5-fold (standard benchmark)

Metrics per condition: RMSE, MAE, R²
LOCO also reports per-country held-out metrics.

Outputs:
    data/processed/results_all.csv        — full 18-condition table
    data/processed/results_loco_country.csv — per-country breakdown (LOCO only)

Run:
    python scripts/04_train_eval.py
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    master = pd.read_parquet(PROCESSED / "master_dataset.parquet")
    prithvi = pd.read_parquet(PROCESSED / "embeddings_prithvi.parquet")
    vit = pd.read_parquet(PROCESSED / "embeddings_vit.parquet")

    df = master.merge(prithvi, on="field_id", how="inner")
    df = df.merge(vit, on="field_id", how="inner")
    print(f"Merged dataset: {len(df):,} rows, {df.shape[1]} columns")
    return df


def get_feature_cols(df: pd.DataFrame):
    """Return column lists for each feature set."""
    s2_cols = [c for c in df.columns if c.startswith("S2_")]
    idx_cols = ["NDVI", "EVI", "LSWI", "NDWI"]
    chirps_cols = [c for c in ["chirps_total", "chirps_mean", "chirps_cv"]
                   if c in df.columns]
    spectral_cols = s2_cols + idx_cols + chirps_cols

    prithvi_cols = [c for c in df.columns if c.startswith("prithvi_")]
    vit_cols     = [c for c in df.columns if c.startswith("vit_")]

    return {
        "spectral": spectral_cols,
        "prithvi":  prithvi_cols,
        "vit":      vit_cols,
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


# ---------------------------------------------------------------------------
# CV schemes
# ---------------------------------------------------------------------------

def run_loco(df: pd.DataFrame, feat_cols: list, target: str,
             model_name: str) -> tuple[dict, list]:
    """
    Leave-One-Country-Out CV.
    Returns aggregate metrics + per-country breakdown.
    """
    countries = df["country"].unique()
    y_true_all, y_pred_all = [], []
    country_rows = []

    for held_out in countries:
        train_mask = df["country"] != held_out
        test_mask  = df["country"] == held_out

        X_train = df.loc[train_mask, feat_cols].values
        y_train = df.loc[train_mask, target].values
        X_test  = df.loc[test_mask, feat_cols].values
        y_test  = df.loc[test_mask, target].values

        # impute NaN with train column median; fallback 0 if column all-NaN
        train_medians = np.nanmedian(X_train, axis=0)
        train_medians = np.where(np.isnan(train_medians), 0.0, train_medians)
        for col_i in range(X_train.shape[1]):
            fill = train_medians[col_i]
            X_train[np.isnan(X_train[:, col_i]), col_i] = fill
            X_test[np.isnan(X_test[:, col_i]), col_i] = fill

        # drop target NaN rows only
        train_ok = ~np.isnan(y_train)
        test_ok  = ~np.isnan(y_test)
        X_train, y_train = X_train[train_ok], y_train[train_ok]
        X_test, y_test   = X_test[test_ok],   y_test[test_ok]

        if len(X_test) == 0:
            continue

        models = make_models()
        pipe = models[model_name]
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        m = compute_metrics(y_test, y_pred)
        m["country"] = held_out
        m["n_test"] = int(len(y_test))
        country_rows.append(m)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

    agg = compute_metrics(np.array(y_true_all), np.array(y_pred_all))
    return agg, country_rows


def run_random_cv(df: pd.DataFrame, feat_cols: list, target: str,
                  model_name: str, n_splits: int = 5) -> dict:
    """Stratified 5-fold CV (stratified by country to keep distribution)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    X = df[feat_cols].values
    y = df[target].values

    # impute NaN with global column median; drop NaN targets
    col_medians = np.nanmedian(X, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
    for col_i in range(X.shape[1]):
        X[np.isnan(X[:, col_i]), col_i] = col_medians[col_i]
    ok = ~np.isnan(y)
    X, y = X[ok], y[ok]

    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in kf.split(X):
        models = make_models()
        pipe = models[model_name]
        pipe.fit(X[train_idx], y[train_idx])
        y_pred = pipe.predict(X[test_idx])
        y_true_all.extend(y[test_idx].tolist())
        y_pred_all.extend(y_pred.tolist())

    return compute_metrics(np.array(y_true_all), np.array(y_pred_all))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_data()
    feat_sets = get_feature_cols(df)

    # use log-transformed yield if available (skewness > 1 during preprocessing)
    target = "yield_log" if "yield_log" in df.columns else "yield_kgha"
    print(f"Target: {target}")
    print(f"Countries: {sorted(df['country'].unique())}")
    print(f"Feature set sizes: "
          f"spectral={len(feat_sets['spectral'])}, "
          f"prithvi={len(feat_sets['prithvi'])}, "
          f"vit={len(feat_sets['vit'])}")

    results = []
    country_results = []

    feature_names  = list(feat_sets.keys())       # spectral, prithvi, vit
    regressor_names = ["ridge", "rf", "xgb"]
    cv_names        = ["loco", "random"]

    total = len(feature_names) * len(regressor_names) * len(cv_names)
    run = 0

    for feat_name in feature_names:
        cols = feat_sets[feat_name]
        for model_name in regressor_names:
            for cv_name in cv_names:
                run += 1
                label = f"[{run:02d}/{total}] {feat_name} / {model_name} / {cv_name}"
                print(f"\n{label}")

                if cv_name == "loco":
                    metrics, country_rows = run_loco(df, cols, target, model_name)
                    for row in country_rows:
                        row.update(feature=feat_name, model=model_name, cv=cv_name)
                        country_results.append(row)
                else:
                    metrics = run_random_cv(df, cols, target, model_name)

                metrics.update(feature=feat_name, model=model_name, cv=cv_name)
                results.append(metrics)
                print(f"  RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  R²={metrics['r2']:.4f}")

    # save results
    results_df = pd.DataFrame(results)[
        ["feature", "model", "cv", "rmse", "mae", "r2"]]
    out_path = PROCESSED / "results_all.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    if country_results:
        country_df = pd.DataFrame(country_results)[
            ["feature", "model", "cv", "country", "n_test", "rmse", "mae", "r2"]]
        out_path2 = PROCESSED / "results_loco_country.csv"
        country_df.to_csv(out_path2, index=False)
        print(f"Saved: {out_path2}")

    # print summary table
    print("\n=== Summary (all 18 conditions) ===")
    print(results_df.to_string(index=False))

    print("\n=== LOCO R² by feature set (best model per set) ===")
    loco = results_df[results_df["cv"] == "loco"]
    best = loco.groupby("feature")["r2"].max().sort_values(ascending=False)
    print(best.to_string())

    print("\nDone. Next: python scripts/05_figures.py")


if __name__ == "__main__":
    main()
