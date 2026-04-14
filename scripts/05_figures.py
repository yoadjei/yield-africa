"""
Phase 5 — Generate paper figures.

Figures produced:
  fig1_loco_r2_heatmap.pdf      — R² heatmap: feature x model (LOCO)
  fig2_random_vs_loco.pdf       — grouped bar: random vs LOCO R² per feature set
  fig3_loco_country_rmse.pdf    — per-country RMSE under LOCO + naive baseline + n_test labels
  fig4_generalization_gap.pdf   — generalization gap (random R² − LOCO R²) per condition
  fig5_naive_baseline.pdf       — model RMSE vs naive mean-predictor RMSE per country
  fig6_pred_vs_actual.pdf       — predicted vs actual scatter, best condition (prithvi/ridge/loco)
  fig7_kl_divergence.pdf        — pairwise KL divergence heatmap on log-yield distributions
  fig8_fold_r2_errorbars.pdf    — per-fold R² mean ± std by feature set and model

Run:
    python scripts/05_figures.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seaborn as sns
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

PALETTE = {
    "spectral": "#1f77b4",   # vivid blue
    "prithvi":  "#e84545",   # vivid red-orange
    "vit":      "#2ca02c",   # vivid green
}
MODEL_COLORS = ["#1f77b4", "#ff7f0e", "#9467bd"]   # blue / orange / purple
MODEL_ORDER   = ["ridge", "rf", "xgb"]
FEATURE_ORDER = ["spectral", "prithvi", "vit"]
MODEL_LABELS  = {"ridge": "Ridge", "rf": "Random Forest", "xgb": "XGBoost"}
FEAT_LABELS   = {"spectral": "Spectral", "prithvi": "Prithvi-EO", "vit": "ViT-Base"}

COUNTRY_COLORS = {
    "Kenya":    "#e41a1c",
    "Malawi":   "#377eb8",
    "Nigeria":  "#4daf4a",
    "Rwanda":   "#984ea3",
    "Tanzania": "#ff7f00",
}


def load():
    results = pd.read_csv(PROCESSED / "results_all.csv")
    country = pd.read_csv(PROCESSED / "results_loco_country.csv")
    return results, country


def load_master():
    master = pd.read_parquet(PROCESSED / "master_dataset.parquet")
    prithvi = pd.read_parquet(PROCESSED / "embeddings_prithvi.parquet")
    df = master.merge(prithvi, on="field_id", how="inner")
    target = "yield_log" if "yield_log" in df.columns else "yield_kgha"
    return df, target


def compute_naive_baselines(master: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    For each held-out country, naive predictor = mean of all other countries' target.
    Returns DataFrame with columns: country, naive_rmse, naive_r2, n_test.
    """
    rows = []
    countries = master["country"].unique()
    for held_out in countries:
        train = master[master["country"] != held_out][target].dropna().values
        test  = master[master["country"] == held_out][target].dropna().values
        naive_pred = np.full(len(test), train.mean())
        rmse = float(np.sqrt(np.mean((test - naive_pred) ** 2)))
        r2   = float(r2_score(test, naive_pred))
        rows.append({"country": held_out, "naive_rmse": rmse,
                     "naive_r2": r2, "n_test": len(test)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fig 1 — LOCO R² heatmap
# ---------------------------------------------------------------------------

def fig1_heatmap(results: pd.DataFrame):
    loco = results[results["cv"] == "loco"].copy()
    pivot = loco.pivot(index="feature", columns="model", values="r2")
    pivot = pivot.loc[FEATURE_ORDER, MODEL_ORDER]
    pivot.index   = [FEAT_LABELS[f] for f in pivot.index]
    pivot.columns = [MODEL_LABELS[m] for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn",
        center=0, vmin=-0.15, vmax=0.15,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 11, "weight": "bold"},
    )
    ax.set_title("LOCO Cross-Country R²\n(Leave-One-Country-Out)", fontsize=12)
    ax.set_xlabel("Regressor", fontsize=10)
    ax.set_ylabel("Feature Set", fontsize=10)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    out = FIGS / "fig1_loco_r2_heatmap.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Fig 2 — Random vs LOCO R² grouped bar
# ---------------------------------------------------------------------------

def fig2_random_vs_loco(results: pd.DataFrame):
    loco   = results[results["cv"] == "loco"][["feature", "model", "r2"]].rename(columns={"r2": "loco_r2"})
    random = results[results["cv"] == "random"][["feature", "model", "r2"]].rename(columns={"r2": "rand_r2"})
    df = loco.merge(random, on=["feature", "model"])

    x = np.arange(len(FEATURE_ORDER))
    width = 0.13
    offsets = np.linspace(-width, width, len(MODEL_ORDER))

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for mi, model in enumerate(MODEL_ORDER):
        sub = df[df["model"] == model].set_index("feature").loc[FEATURE_ORDER]
        col = MODEL_COLORS[mi]
        for fi, feat in enumerate(FEATURE_ORDER):
            rand_val = sub.loc[feat, "rand_r2"]
            loco_val = sub.loc[feat, "loco_r2"]
            pos = x[fi] + offsets[mi]
            ax.bar(pos - width * 0.25, rand_val, width * 0.45,
                   color=col, alpha=0.9, label=MODEL_LABELS[model] if fi == 0 else "")
            ax.bar(pos + width * 0.25, loco_val, width * 0.45,
                   color=col, alpha=0.45, hatch="///",
                   label=MODEL_LABELS[model] + " (LOCO)" if fi == 0 else "")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([FEAT_LABELS[f] for f in FEATURE_ORDER], fontsize=11)
    ax.set_ylabel("R²", fontsize=11)
    ax.set_title("Within-Country (Random) vs Cross-Country (LOCO) R²", fontsize=11)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=MODEL_COLORS[i], label=MODEL_LABELS[m])
        for i, m in enumerate(MODEL_ORDER)
    ]
    legend_handles += [
        Patch(facecolor="grey", alpha=0.9, label="Solid = Random CV"),
        Patch(facecolor="grey", alpha=0.45, hatch="///", label="Hatched = LOCO CV"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right", ncol=2)
    plt.tight_layout()
    out = FIGS / "fig2_random_vs_loco.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Fig 3 — Per-country RMSE under LOCO + n_test labels
# ---------------------------------------------------------------------------

def fig3_country_rmse(country: pd.DataFrame):
    avg_rmse = country.groupby(["feature", "model"])["rmse"].mean()
    best_models = avg_rmse.groupby("feature").idxmin().apply(lambda x: x[1])

    rows = []
    for feat in FEATURE_ORDER:
        best_m = best_models[feat]
        sub = country[(country["feature"] == feat) & (country["model"] == best_m)]
        for _, r in sub.iterrows():
            rows.append({"feature": feat, "country": r["country"],
                         "rmse": r["rmse"], "model": best_m,
                         "n_test": r["n_test"]})
    df = pd.DataFrame(rows)

    countries = sorted(df["country"].unique())
    x = np.arange(len(countries))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars_drawn = {}
    for fi, feat in enumerate(FEATURE_ORDER):
        sub = df[df["feature"] == feat].set_index("country").reindex(countries)
        bars = ax.bar(x + (fi - 1) * width, sub["rmse"], width,
                      label=f"{FEAT_LABELS[feat]} ({MODEL_LABELS[sub['model'].iloc[0]]})",
                      color=PALETTE[feat], alpha=0.9, edgecolor="white", linewidth=0.4)
        bars_drawn[feat] = (bars, sub)

    n_test_by_country = df.groupby("country")["n_test"].first()
    xlabels = [f"{c}\nn={n_test_by_country[c]:,}" for c in countries]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel("RMSE (kg/ha)", fontsize=11)
    ax.set_title("Per-Country LOCO RMSE  (best model per feature set)", fontsize=11, pad=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = FIGS / "fig3_loco_country_rmse.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Fig 4 — Generalization gap
# ---------------------------------------------------------------------------

def fig4_generalization_gap(results: pd.DataFrame):
    loco   = results[results["cv"] == "loco"][["feature", "model", "r2"]].rename(columns={"r2": "loco"})
    random = results[results["cv"] == "random"][["feature", "model", "r2"]].rename(columns={"r2": "rand"})
    df = loco.merge(random, on=["feature", "model"])
    df["gap"] = df["rand"] - df["loco"]

    x = np.arange(len(FEATURE_ORDER))
    width = 0.22

    fig, ax = plt.subplots(figsize=(7, 4))
    for mi, model in enumerate(MODEL_ORDER):
        sub = df[df["model"] == model].set_index("feature").reindex(FEATURE_ORDER)
        ax.bar(x + (mi - 1) * width, sub["gap"], width,
               label=MODEL_LABELS[model],
               color=MODEL_COLORS[mi], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([FEAT_LABELS[f] for f in FEATURE_ORDER], fontsize=11)
    ax.set_ylabel("Generalization Gap\n(Random R² − LOCO R²)", fontsize=10)
    ax.set_title("Generalization Gap by Feature Set and Regressor", fontsize=11)
    ax.legend(fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    out = FIGS / "fig4_generalization_gap.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Fig 5 — Model RMSE vs naive mean-predictor baseline per country
# ---------------------------------------------------------------------------

def fig5_naive_baseline(country: pd.DataFrame, naive: pd.DataFrame):
    """
    For each country: bar = best-model RMSE (averaged across feature sets),
    line = naive mean-predictor RMSE.
    Shows concretely that all models are worse than predicting the training mean
    for held-out countries.
    """
    # best model per feature
    avg_rmse = country.groupby(["feature", "model"])["rmse"].mean()
    best_models = avg_rmse.groupby("feature").idxmin().apply(lambda x: x[1])

    rows = []
    for feat in FEATURE_ORDER:
        best_m = best_models[feat]
        sub = country[(country["feature"] == feat) & (country["model"] == best_m)]
        for _, r in sub.iterrows():
            rows.append({"feature": feat, "country": r["country"], "rmse": r["rmse"]})
    df = pd.DataFrame(rows)

    countries = sorted(df["country"].unique())
    x = np.arange(len(countries))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    for fi, feat in enumerate(FEATURE_ORDER):
        sub = df[df["feature"] == feat].set_index("country").reindex(countries)
        ax.bar(x + (fi - 1) * width, sub["rmse"], width,
               label=FEAT_LABELS[feat], color=PALETTE[feat], alpha=0.85)

    # naive baseline as scatter + line
    naive_ordered = naive.set_index("country").reindex(countries)
    ax.plot(x, naive_ordered["naive_rmse"], "D--", color="black",
            linewidth=1.5, markersize=7, label="Naive (train-mean predictor)", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(countries, fontsize=10)
    ax.set_ylabel("RMSE (kg/ha)", fontsize=11)
    ax.set_title("LOCO RMSE vs Naive Baseline per Country\n"
                 "(diamond = predict held-out country with training-set mean)", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = FIGS / "fig5_naive_baseline.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Fig 6 — Predicted vs actual scatter (best condition: prithvi / ridge / LOCO)
# ---------------------------------------------------------------------------

def fig6_pred_vs_actual(master: pd.DataFrame, target: str):
    """
    Re-runs prithvi/ridge/LOCO (fast: ~6k rows, Ridge only) to get predictions,
    then plots predicted vs actual coloured by country.
    """
    prithvi_cols = [c for c in master.columns if c.startswith("prithvi_")]

    countries = sorted(master["country"].unique())
    all_true, all_pred, all_country = [], [], []

    for held_out in countries:
        train_mask = master["country"] != held_out
        test_mask  = master["country"] == held_out

        X_train = master.loc[train_mask, prithvi_cols].values.copy()
        y_train = master.loc[train_mask, target].values.copy()
        X_test  = master.loc[test_mask,  prithvi_cols].values.copy()
        y_test  = master.loc[test_mask,  target].values.copy()

        # impute NaN
        medians = np.nanmedian(X_train, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        for ci in range(X_train.shape[1]):
            X_train[np.isnan(X_train[:, ci]), ci] = medians[ci]
            X_test[np.isnan(X_test[:, ci]), ci]   = medians[ci]

        train_ok = ~np.isnan(y_train)
        test_ok  = ~np.isnan(y_test)
        X_train, y_train = X_train[train_ok], y_train[train_ok]
        X_test, y_test   = X_test[test_ok],   y_test[test_ok]

        if len(X_test) == 0:
            continue

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RidgeCV(alphas=[0.1, 1, 10, 100, 1000])),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())
        all_country.extend([held_out] * len(y_test))

    df = pd.DataFrame({"true": all_true, "pred": all_pred, "country": all_country})
    overall_r2 = r2_score(df["true"], df["pred"])

    fig, axes = plt.subplots(1, len(countries), figsize=(4 * len(countries), 4), sharey=False)

    for ax, ctry in zip(axes, countries):
        sub = df[df["country"] == ctry]
        col = COUNTRY_COLORS.get(ctry, "steelblue")
        ax.scatter(sub["true"], sub["pred"], alpha=0.3, s=8, color=col)

        lo = min(sub["true"].min(), sub["pred"].min())
        hi = max(sub["true"].max(), sub["pred"].max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)

        r2 = r2_score(sub["true"], sub["pred"])
        ax.set_title(f"{ctry}\nR²={r2:.3f}, n={len(sub)}", fontsize=9)
        ax.set_xlabel("Actual", fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel("Predicted", fontsize=8)

    fig.suptitle(
        f"Predicted vs Actual — Prithvi-EO / Ridge / LOCO  (overall R²={overall_r2:.3f})",
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    out = FIGS / "fig6_pred_vs_actual.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Fig 7 — KL divergence heatmap (pairwise, log-yield)
# ---------------------------------------------------------------------------

def fig7_kl_heatmap():
    kl_path = PROCESSED / "label_shift_kl.csv"
    if not kl_path.exists():
        print("label_shift_kl.csv not found — run scripts/04b_sensitivity.py first")
        return
    kl = pd.read_csv(kl_path)
    pivot = kl.pivot(index="src", columns="tgt", values="kl")
    # fill diagonal with 0
    for c in pivot.index:
        if c in pivot.columns:
            pivot.loc[c, c] = 0.0
    countries = sorted(pivot.index)
    pivot = pivot.reindex(index=countries, columns=countries)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    mask = pivot.isna()
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, mask=mask,
        annot_kws={"size": 10},
    )
    ax.set_title("Pairwise KL Divergence of Log-Yield Distributions\nKL(row ‖ col)", fontsize=11)
    ax.set_xlabel("Target Country", fontsize=10)
    ax.set_ylabel("Source Country", fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    out = FIGS / "fig7_kl_divergence.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Fig 8 — Per-fold R² mean ± std error bars
# ---------------------------------------------------------------------------

def fig8_fold_errorbars():
    fold_path = PROCESSED / "results_loco_fold_std.csv"
    if not fold_path.exists():
        print("results_loco_fold_std.csv not found — run scripts/04b_sensitivity.py first")
        return
    fold = pd.read_csv(fold_path)

    x = np.arange(len(FEATURE_ORDER))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mi, model in enumerate(MODEL_ORDER):
        sub = fold[fold["model"] == model].set_index("feature").reindex(FEATURE_ORDER)
        means = sub["mean"].values
        stds  = sub["std"].values
        col = MODEL_COLORS[mi]
        ax.bar(x + (mi - 1) * width, means, width,
               yerr=stds, capsize=4,
               label=MODEL_LABELS[model], color=col, alpha=0.85,
               error_kw={"elinewidth": 1.2, "ecolor": "black"})

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([FEAT_LABELS[f] for f in FEATURE_ORDER], fontsize=11)
    ax.set_ylabel("R² (mean ± std across LOCO folds)", fontsize=10)
    ax.set_title("Per-Fold R² Variability — Conditions Not Statistically Separable", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = FIGS / "fig8_fold_r2_errorbars.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Fig 9 — NDVI-only ablation vs all feature sets (LOCO R²)
# ---------------------------------------------------------------------------

def fig9_ndvi_ablation(results: pd.DataFrame):
    ndvi_path = PROCESSED / "results_ndvi_only.csv"
    if not ndvi_path.exists():
        print("results_ndvi_only.csv not found — run scripts/04b_sensitivity.py first")
        return
    ndvi = pd.read_csv(ndvi_path)

    # build combined table: ndvi_only + spectral + prithvi + vit (LOCO)
    loco = results[results["cv"] == "loco"][["feature", "model", "r2"]].copy()
    ndvi_rows = ndvi[["model", "r2"]].copy()
    ndvi_rows["feature"] = "ndvi_only"
    combined = pd.concat([loco, ndvi_rows[["feature", "model", "r2"]]], ignore_index=True)

    feat_order = ["ndvi_only", "spectral", "prithvi", "vit"]
    feat_labels = {
        "ndvi_only": "NDVI\nOnly",
        "spectral":  "Spectral\n(10-band)",
        "prithvi":   "Prithvi-EO\n(768-dim)",
        "vit":       "ViT-Base\n(768-dim)",
    }
    x = np.arange(len(feat_order))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mi, model in enumerate(MODEL_ORDER):
        sub = combined[combined["model"] == model].set_index("feature").reindex(feat_order)
        col = MODEL_COLORS[mi]
        ax.bar(x + (mi - 1) * width, sub["r2"], width,
               label=MODEL_LABELS[model], color=col, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([feat_labels[f] for f in feat_order], fontsize=10)
    ax.set_ylabel("LOCO R²", fontsize=11)
    ax.set_title("Feature Ablation: NDVI-Only vs Spectral vs Foundation Embeddings\n"
                 "(all conditions negative — richer features do not improve cross-country generalisation)",
                 fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = FIGS / "fig9_ndvi_ablation.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Fig 10 — Yield distribution violin per country (label shift visual)
# ---------------------------------------------------------------------------

def fig10_yield_distributions(master: pd.DataFrame):
    countries = ["Kenya", "Malawi", "Nigeria", "Rwanda", "Tanzania"]
    data = master[master["country"].isin(countries)][["country", "yield_kgha"]].dropna()
    pooled_mean = data["yield_kgha"].mean()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    parts = ax.violinplot(
        [data[data["country"] == c]["yield_kgha"].values for c in countries],
        positions=range(len(countries)),
        showmedians=True,
        showextrema=False,   # suppress whiskers — cleaner
        widths=0.7,
    )

    colors = [COUNTRY_COLORS[c] for c in countries]
    for body, col in zip(parts["bodies"], colors):
        body.set_facecolor(col)
        body.set_alpha(0.80)
        body.set_edgecolor("white")
        body.set_linewidth(0.8)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)

    ax.axhline(pooled_mean, color="#444444", linewidth=1.2,
               linestyle="--", label=f"Pooled mean  {pooled_mean:.0f} kg/ha", zorder=3)

    # stats table as x-tick labels: "Country\nn=X  μ=Y"
    xlabels = []
    for c in countries:
        vals = data[data["country"] == c]["yield_kgha"]
        xlabels.append(f"{c}\nn={len(vals):,}   μ={vals.mean():.0f}")

    ax.set_xticks(range(len(countries)))
    ax.set_xticklabels(xlabels, fontsize=9.5)
    ax.set_ylabel("Yield (kg/ha)", fontsize=11)
    ax.set_title("Yield Distributions per Country — Label Shift Across LOCO Folds",
                 fontsize=11, pad=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = FIGS / "fig10_yield_distributions.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results, country = load()
    print(f"Loaded {len(results)} condition results, {len(country)} country-level rows")

    fig1_heatmap(results)
    fig2_random_vs_loco(results)
    fig3_country_rmse(country)
    fig4_generalization_gap(results)

    print("Computing naive baselines and loading master dataset...")
    master, target = load_master()
    naive = compute_naive_baselines(master, target)
    print(f"Naive baselines:\n{naive.to_string(index=False)}")

    fig5_naive_baseline(country, naive)
    fig6_pred_vs_actual(master, target)

    fig7_kl_heatmap()
    fig8_fold_errorbars()
    fig9_ndvi_ablation(results)
    fig10_yield_distributions(master)

    print(f"\nAll figures saved to {FIGS}/")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
