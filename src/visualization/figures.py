"""Generate every figure in reports/figures/ from real data, saved models
and result tables - nothing is hand-drawn or copied from the coursework PDF.

    python make_figures.py --stage eda        # fig01-06  (needs cleaned data)
    python make_figures.py --stage results    # fig07-11, 15, 16  (needs benchmark run)
    python make_figures.py --stage ablation   # fig17     (needs ablation runs)
    python make_figures.py --stage explain    # fig18-19  (needs a trained model)
    python make_figures.py --stage bigdata    # fig12-14
    python make_figures.py --stage all
"""
from __future__ import annotations

import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.evaluation.metrics import evaluate_predictions  # noqa: E402,F401
from src.explainability.analysis import (  # noqa: E402
    error_analysis_tables, partial_dependence_usd, permutation_importance_table,
)
from src.training_utils import load_training_frame, model_path  # noqa: E402
from src.utils.config import resolve_path  # noqa: E402
from src.utils.io import ensure_dir, read_table  # noqa: E402
from src.visualization.plots import save_figure, set_publication_style  # noqa: E402

NAMES = {"ridge": "Ridge", "random_forest": "Random Forest", "xgboost": "XGBoost",
         "hist_gb": "HistGradientBoosting", "mlp": "MLP"}
COLORS = {"ridge": "#4C72B0", "random_forest": "#DD8452", "xgboost": "#C44E52",
          "hist_gb": "#55A868", "mlp": "#8172B2"}


def _fig_dir():
    return ensure_dir(resolve_path("figure_dir"))


def _tab_dir():
    return ensure_dir(resolve_path("table_dir"))


def _save(fig, name):
    save_figure(fig, _fig_dir() / name)
    print("saved", name)


def _benchmark(suffix: str = "") -> pd.DataFrame:
    return pd.read_csv(_tab_dir() / f"benchmark_results{suffix}.csv")


def _load_model(name: str, feature_set: str = "full"):
    with open(model_path(name, feature_set), "rb") as f:
        return pickle.load(f)


def _boost_name(available) -> str:
    return "xgboost" if "xgboost" in available else "hist_gb"


# ----------------------------------------------------------------- EDA
def eda_figures() -> None:
    set_publication_style()
    df = read_table(resolve_path("processed_dataset"))
    rev = df["ttm_revenue"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].hist(rev, bins=60, color="#4C72B0", edgecolor="white")
    ax[0].set(title="TTM revenue - raw scale", xlabel="Revenue (USD)", ylabel="Listings")
    ax[1].hist(np.log1p(rev), bins=60, color="#DD8452", edgecolor="white")
    ax[1].set(title="TTM revenue - log(1 + revenue)", xlabel="log(1 + revenue)", ylabel="Listings")
    ax[0].text(0.97, 0.9, f"skew = {rev.skew():.2f}", transform=ax[0].transAxes, ha="right")
    ax[1].text(0.03, 0.9, f"skew = {np.log1p(rev).skew():.2f}", transform=ax[1].transAxes)
    _save(fig, "fig01_revenue_distribution.png")

    top = df.groupby("country")["ttm_revenue"].median().sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(top.index, top.values, color="teal", edgecolor="white")
    ax.set(title="Median TTM revenue by country (top 12)", ylabel="Median revenue (USD)")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    _save(fig, "fig02_revenue_by_country.png")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(df["ttm_occupancy"], bins=50, color="#8172B2", edgecolor="white")
    ax.set(title="TTM occupancy rate distribution", xlabel="Occupancy rate", ylabel="Listings")
    _save(fig, "fig03_occupancy_distribution.png")

    cols = ["ttm_revenue", "ttm_occupancy", "ttm_avg_rate", "guests", "bedrooms", "beds", "baths",
            "num_reviews", "rating_overall", "cleaning_fee", "min_nights"]
    corr = df[cols].dropna().corr()
    fig, ax = plt.subplots(figsize=(9.5, 8))
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)), annot=True, fmt=".2f",
                cmap="coolwarm", vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.75}, ax=ax)
    ax.grid(False)
    ax.set_title("Pearson correlation - key numeric features")
    _save(fig, "fig04_correlation_heatmap.png")

    room = df.groupby("room_type")["ttm_revenue"].median().sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(room.index, room.values, color="#DD8452", edgecolor="white")
    ax.set(title="Median TTM revenue by room type", xlabel="Median revenue (USD)")
    _save(fig, "fig05_revenue_by_roomtype.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    groups = [df.loc[df["superhost"] == 0, "ttm_revenue"], df.loc[df["superhost"] == 1, "ttm_revenue"]]
    bp = ax.boxplot(groups, patch_artist=True, notch=True, showfliers=False,
                    medianprops=dict(color="#C44E52", linewidth=2))
    for patch, c in zip(bp["boxes"], ["#9ecae1", "#4C72B0"]):
        patch.set_facecolor(c)
    ax.set_xticklabels(["Not superhost", "Superhost"])
    ax.set(title="TTM revenue by superhost status (outliers hidden)", ylabel="Revenue (USD)")
    _save(fig, "fig06_superhost_revenue.png")


# ------------------------------------------------------------- results
def results_figures() -> None:
    set_publication_style()
    bench = _benchmark()
    models = list(bench["model"])
    pipes = {m: _load_model(m) for m in models}
    label = [NAMES[m] for m in models]
    colors = [COLORS[m] for m in models]

    # fig07 - random forest impurity importance
    if "random_forest" in pipes:
        p = pipes["random_forest"]
        names = [n.split("__", 1)[1] for n in p.named_steps["preprocess"].get_feature_names_out()]
        imp = pd.Series(p.named_steps["regressor"].feature_importances_, index=names).nlargest(15)[::-1]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.barh(imp.index, imp.values, color=COLORS["random_forest"])
        ax.set(title="Random Forest - top 15 features (mean decrease in impurity)",
               xlabel="Importance")
        _save(fig, "fig07_random_forest_importance.png")
        imp[::-1].rename("importance").to_csv(_tab_dir() / "random_forest_importance.csv")

    # fig09 - metric comparison
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for a, (col, title, fmt) in zip(ax, [("MAE", "MAE (USD) - lower is better", "${:,.0f}"),
                                         ("RMSE", "RMSE (USD) - lower is better", "${:,.0f}"),
                                         ("R2_test", "Test R\u00b2 - higher is better", "{:.3f}")]):
        bars = a.bar(label, bench[col], color=colors, edgecolor="white")
        a.set_title(title, fontsize=11)
        a.bar_label(bars, labels=[fmt.format(v) for v in bench[col]], fontsize=8, padding=2)
        plt.setp(a.get_xticklabels(), rotation=25, ha="right", fontsize=8)
        if col == "R2_test":
            a.set_ylim(bench[col].min() - 0.05, 1.0)
    fig.suptitle("Model comparison on the held-out test set", fontweight="bold")
    _save(fig, "fig09_model_comparison.png")

    # fig10 - best model predictions
    best = bench.sort_values("R2_test", ascending=False).iloc[0]["model"]
    _, X_test, _, y_test = load_training_frame()
    pred_log = pipes[best].predict(X_test)
    actual, pred = np.expm1(y_test), np.expm1(pred_log)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(actual), size=min(5000, len(actual)), replace=False)
    lim = float(np.quantile(actual, 0.995))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ax[0].scatter(actual[idx], pred[idx], s=5, alpha=0.3, color=COLORS[best])
    ax[0].plot([0, lim], [0, lim], "--", color="#C44E52", label="Perfect prediction")
    ax[0].set(xlim=(0, lim), ylim=(0, lim), xlabel="Actual revenue (USD)",
              ylabel="Predicted revenue (USD)", title=f"{NAMES[best]}: predicted vs actual (test)")
    ax[0].legend()
    ax[1].scatter(pred_log[idx], (y_test - pred_log)[idx], s=5, alpha=0.3, color="#4C72B0")
    ax[1].axhline(0, ls="--", color="#C44E52")
    ax[1].set(xlabel="Predicted log(1 + revenue)", ylabel="Residual (log scale)",
              title=f"{NAMES[best]}: residuals")
    _save(fig, "fig10_best_model_pred_vs_actual.png")

    # fig11 - generalisation
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    w = 0.27
    for i, (col, lab, c) in enumerate([("R2_train", "Train R\u00b2", "#4C72B0"),
                                       ("R2_test", "Test R\u00b2", "#DD8452"),
                                       ("R2_cv", "5-fold CV R\u00b2", "#55A868")]):
        ax.bar(x + (i - 1) * w, bench[col], w, label=lab, color=c, edgecolor="white")
    ax.set_xticks(x, label)
    ax.set(ylim=(max(0.0, bench[["R2_train", "R2_test", "R2_cv"]].min().min() - 0.03), 1.0),
           ylabel="R\u00b2 (y-axis truncated)", title="Train / test / CV R\u00b2 - generalisation check")
    ax.legend(loc="lower right")
    _save(fig, "fig11_generalisation.png")

    # fig15 - CV stability
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ypos = np.arange(len(models))
    ax.errorbar(bench["R2_cv"], ypos, xerr=bench["R2_cv_std"], fmt="o", capsize=4,
                color="black", ecolor="black")
    for yp, m, v, s in zip(ypos, models, bench["R2_cv"], bench["R2_cv_std"]):
        ax.text(v, yp + 0.18, f"{v:.4f} \u00b1 {s:.4f}", ha="center", fontsize=8)
    ax.set_yticks(ypos, label)
    ax.set(xlabel="5-fold CV R\u00b2 (mean \u00b1 std)", title="Cross-validation stability")
    _save(fig, "fig15_cv_stability.png")

    # fig16 - runtime vs accuracy
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for _, r in bench.iterrows():
        ax.scatter(r["time_seconds"], r["R2_test"], s=160, color=COLORS[r["model"]], edgecolor="white", zorder=3)
        ax.annotate(NAMES[r["model"]], (r["time_seconds"], r["R2_test"]), xytext=(8, 6),
                    textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set(xlabel="GridSearchCV training time (s, log scale; 1 CPU core)", ylabel="Test R\u00b2",
           title="Runtime vs accuracy trade-off")
    _save(fig, "fig16_runtime_vs_accuracy.png")


# ------------------------------------------------------------ ablation
def ablation_figure() -> None:
    set_publication_style()
    frames = []
    for suffix, name in [("", "Full"), ("_no_performance", "No realised price/occupancy"),
                         ("_pre_booking", "Pre-booking only")]:
        try:
            f = _benchmark(suffix)
        except FileNotFoundError:
            continue
        f["set"] = name
        frames.append(f)
    df = pd.concat(frames)
    models = [m for m in ["ridge", "hist_gb", "xgboost"] if m in set(df["model"])]
    sets = [s for s in ["Full", "No realised price/occupancy", "Pre-booking only"] if s in set(df["set"])]
    palette = dict(zip(["Full", "No realised price/occupancy", "Pre-booking only"],
                       ["#4C72B0", "#DD8452", "#C44E52"]))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    w = 0.8 / len(sets)
    for j, (col, title, fmt) in enumerate([("R2_test", "Test R\u00b2", "{:.3f}"), ("MAE", "Test MAE (USD)", "${:,.0f}")]):
        for i, s in enumerate(sets):
            vals = [df[(df["model"] == m) & (df["set"] == s)][col].iloc[0] for m in models]
            bars = ax[j].bar(np.arange(len(models)) + (i - (len(sets) - 1) / 2) * w, vals, w,
                             label=s, color=palette[s], edgecolor="white")
            ax[j].bar_label(bars, labels=[fmt.format(v) for v in vals], fontsize=8, padding=2)
        ax[j].set_xticks(np.arange(len(models)), [NAMES[m] for m in models])
        ax[j].set_title(title)
    ax[0].set_ylim(0, 1.3)
    ax[0].legend(fontsize=8, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Ablation: how much does the model rely on realised performance?", fontweight="bold")
    _save(fig, "fig17_ablation.png")


# ------------------------------------------------------------- explain
def explain_figures() -> None:
    set_publication_style()
    bench = _benchmark()
    name = _boost_name(set(bench["model"]))
    pipe = _load_model(name)
    _, X_test, _, y_test, meta = load_training_frame(return_meta=True)

    perm = permutation_importance_table(pipe, X_test, y_test)
    perm.to_csv(_tab_dir() / "permutation_importance.csv", index=False)
    top = perm.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["feature"], top["r2_drop_mean"], xerr=top["r2_drop_std"], color=COLORS[name], capsize=3)
    ax.set(title=f"{NAMES[name]} - permutation importance (raw input columns)",
           xlabel="Drop in test R\u00b2 when the column is shuffled")
    _save(fig, "fig08_permutation_importance.png")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for a, (col, lab) in zip(ax, [("ttm_occupancy", "Occupancy rate"), ("ttm_avg_rate", "Average nightly rate (USD)"),
                                  ("bedrooms", "Bedrooms")]):
        pdp = partial_dependence_usd(pipe, X_test, col)
        pdp.to_csv(_tab_dir() / f"pdp_{col}.csv", index=False)
        a.plot(pdp[col], pdp["avg_predicted_revenue_usd"], color=COLORS[name], lw=2.5)
        a.set(xlabel=lab)
        a.set_ylim(bottom=0)
    ax[0].set_ylabel("Avg predicted revenue (USD)")
    fig.suptitle(f"Partial dependence ({NAMES[name]}): predicted TTM revenue as one input is varied",
                 fontweight="bold")
    _save(fig, "fig18_partial_dependence.png")

    pred = np.expm1(pipe.predict(X_test))
    by_decile, by_country = error_analysis_tables(np.expm1(y_test), pred, meta["country"].values)
    by_decile.to_csv(_tab_dir() / "error_by_revenue_decile.csv", index=False)
    by_country.to_csv(_tab_dir() / "error_by_country.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].bar(by_decile["revenue_decile"].astype(str), by_decile["MAE"], color=COLORS[name], edgecolor="white")
    ax[0].set(title="MAE by actual-revenue decile (D1 = lowest)", ylabel="MAE (USD)", xlabel="Revenue decile")
    ax2 = ax[0].twinx()
    ax2.plot(by_decile["revenue_decile"].astype(str), by_decile["median_ape"] * 100, "o-", color="#C44E52")
    ax2.set_ylabel("Median abs. % error", color="#C44E52")
    ax2.grid(False)
    ax[1].bar(by_country["country"], by_country["median_ape"] * 100, color="#8172B2", edgecolor="white")
    ax[1].set(title="Median abs. % error by country (top 10 markets)", ylabel="Median abs. % error")
    plt.setp(ax[1].get_xticklabels(), rotation=40, ha="right")
    _save(fig, "fig19_error_analysis.png")


# ------------------------------------------------------------- bigdata
def bigdata_figures() -> None:
    from src.spark.batch_processing import run_batch_processing
    from src.spark.cpu_parallelism import load_or_reported
    from src.spark.stream_processing import run_streaming_simulation

    set_publication_style()
    df = read_table(resolve_path("processed_dataset"))

    batch = run_batch_processing(df.copy(), chunk_size=10_000)
    batch.to_csv(_tab_dir() / "batch_processing.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(batch["batch"], batch["mean_revenue"], "o-", color="#4C72B0")
    ax.axhline(batch["mean_revenue"].mean(), ls="--", color="#C44E52",
               label=f"Mean of batch means: ${batch['mean_revenue'].mean():,.0f}")
    ax.set(xlabel="Batch (10,000 listings each)", ylabel="Mean revenue (USD)",
           title="Batch processing: revenue by batch (data-ordering check)")
    ax.legend()
    _save(fig, "fig12_batch_processing.png")

    rates = read_table(resolve_path("raw_past_rates"))
    stream = run_streaming_simulation(rates)
    stream.to_csv(_tab_dir() / "streaming_monthly.csv", index=False)
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    ax[0].plot(stream["period"], stream["total_revenue"] / 1e6, "o-", color="#55A868")
    ax[0].set(ylabel="Total revenue (M USD)", title="Streaming simulation: monthly revenue and occupancy")
    ax[1].plot(stream["period"], stream["avg_occupancy"], "s-", color="#DD8452")
    ax[1].set(ylabel="Mean occupancy", xlabel="Month")
    plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right")
    _save(fig, "fig13_streaming_trend.png")

    cpu = load_or_reported()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.barh(cpu["config"], cpu["time_s"], color=["#DD8452", "#4C72B0", "#55A868"][: len(cpu)])
    ax.bar_label(bars, labels=[f"{v:.1f}s" for v in cpu["time_s"]], padding=3)
    ax.set(xlabel="Random Forest training time (s)",
           title=f"CPU parallelism - source: {cpu['source'].iloc[0]}")
    _save(fig, "fig14_cpu_parallelism.png")

    mem64 = df.memory_usage(deep=True).sum() / 1024 ** 2
    df32 = df.copy()
    for c in df32.select_dtypes("float64").columns:
        df32[c] = df32[c].astype("float32")
    mem32 = df32.memory_usage(deep=True).sum() / 1024 ** 2
    pd.DataFrame([{"float64_mb": round(mem64, 1), "float32_mb": round(mem32, 1),
                   "reduction_pct": round((1 - mem32 / mem64) * 100, 1)}]).to_csv(
        _tab_dir() / "memory_efficiency.csv", index=False)
    print(f"memory float64={mem64:.1f}MB float32={mem32:.1f}MB")
