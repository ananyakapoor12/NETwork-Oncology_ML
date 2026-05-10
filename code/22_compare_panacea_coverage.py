"""Compare all ML models against PANACEA's native PEN-diff ranking using the paper's coverage@m% protocol."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS  = PROJECT_ROOT / "outputs"
FIGURES  = PROJECT_ROOT / "figures"

N_BUCKETS  = 5
TEST_FRAC  = 0.2
SEED       = 42
COVERAGE_M = [1, 10, 20, 50]    # top-m% thresholds, matching PANACEA paper

FULL_FEATURES = [
    "pen_diff", "dist_diff", "ppr_diff",
    "bucket_pen", "bucket_dist", "bucket_ppr",
    "pos_in_bucket_pen",
]


# ════════════════════════════════════════════════════════════════════════════
# TRAIN / TEST SPLIT
# ════════════════════════════════════════════════════════════════════════════

def split_positives(df: pd.DataFrame):
    """Return (train_pos_idx, test_pos_idx) using the same SEED as the training pipeline."""
    rng = np.random.default_rng(SEED)
    pos_idx = df.index[df["is_known"] == 1].to_numpy().copy()
    rng.shuffle(pos_idx)
    n_test = int(len(pos_idx) * TEST_FRAC)
    return pos_idx[n_test:], pos_idx[:n_test]    # train, test


def load_train_df(cancer_dir: Path) -> pd.DataFrame:
    train_path = cancer_dir / "train_rank_final.csv"
    if not train_path.exists():
        train_path = cancer_dir / "train_rank.csv"
    print(f"  Loading training data from {train_path.name}...")
    return pd.read_csv(train_path).sort_values("group_id").reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ════════════════════════════════════════════════════════════════════════════

def train_lgbm(train_df: pd.DataFrame) -> lgb.LGBMRanker:
    groups = train_df.groupby("group_id").size().to_numpy()
    print(f"  Training LightGBM LambdaMART ({len(groups):,} groups)...", end="", flush=True)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        num_leaves=63,
        learning_rate=0.05,
        n_estimators=300,
        random_state=SEED,
        verbosity=-1,
        force_col_wise=True,
    )
    model.fit(train_df[FULL_FEATURES], train_df["label"].astype(int), group=groups)
    print(" done.")
    return model


def train_random_forest(train_df: pd.DataFrame) -> RandomForestClassifier:
    print(f"  Training Random Forest (500 trees)...", end="", flush=True)
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        n_jobs=-1,
        random_state=SEED,
    )
    rf.fit(train_df[FULL_FEATURES], train_df["label"].astype(int))
    print(" done.")
    return rf


def train_catboost(train_df: pd.DataFrame):
    try:
        from catboost import CatBoostRanker, Pool
    except ImportError:
        print("  [SKIP] CatBoost not installed — skipping.")
        return None

    groups = train_df["group_id"].values
    pool = Pool(
        data=train_df[FULL_FEATURES].values.astype(np.float32),
        label=train_df["label"].astype(int).values,
        group_id=groups,
        feature_names=FULL_FEATURES,
    )
    print(f"  Training CatBoost YetiRank (1000 iterations)...", end="", flush=True)
    model = CatBoostRanker(
        loss_function="YetiRank",
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_seed=SEED,
        verbose=0,
    )
    model.fit(pool)
    print(" done.")
    return model


# ════════════════════════════════════════════════════════════════════════════
# COVERAGE METRIC  (PANACEA's evaluation protocol — Table 4 in their paper)
# ════════════════════════════════════════════════════════════════════════════

def coverage_at_m(bucket_df: pd.DataFrame, score_col: str, m_pct: float) -> float:
    """
    Within a single bucket, sort ALL candidates by score_col descending and
    measure: coverage@m% = fraction of known targets in the top-m% of candidates.
    """
    total_known = int(bucket_df["is_known"].sum())
    if total_known == 0 or len(bucket_df) == 0:
        return float("nan")
    n_top = max(1, int(len(bucket_df) * m_pct / 100))
    top_m = bucket_df.nlargest(n_top, score_col)
    return int(top_m["is_known"].sum()) / total_known


def compute_coverage_table(df: pd.DataFrame, score_col: str, label: str) -> pd.DataFrame:
    rows = []
    for b in range(N_BUCKETS):
        bdf = df[df["bucket_pen"] == b]
        row = {
            "Ranker": label,
            "Bucket": b,
            "N_candidates": len(bdf),
            "N_known": int(bdf["is_known"].sum()),
        }
        for m in COVERAGE_M:
            cov = coverage_at_m(bdf, score_col, m)
            row[f"Coverage@{m}%"] = f"{100*cov:.1f}%" if not np.isnan(cov) else "—"
        rows.append(row)
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY — weighted-mean coverage across explored buckets
# ════════════════════════════════════════════════════════════════════════════

def summarise(cov_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ranker, grp in cov_df.groupby("Ranker", sort=False):
        explored = grp[grp["N_known"] > 0]
        row = {"Ranker": ranker}
        for m in COVERAGE_M:
            col = f"Coverage@{m}%"
            vals    = explored[col].str.rstrip("%").replace("—", "nan").astype(float)
            weights = explored["N_known"].values
            row[col] = f"{np.average(vals.values, weights=weights):.1f}%" \
                       if weights.sum() > 0 else "—"
        rows.append(row)
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# PRINT HELPERS
# ════════════════════════════════════════════════════════════════════════════

def print_bucket_table(all_cov: pd.DataFrame):
    for b in range(N_BUCKETS):
        b_rows = all_cov[all_cov["Bucket"] == b]
        n_known = b_rows["N_known"].iloc[0]
        n_cand  = b_rows["N_candidates"].iloc[0]
        status  = "EXPLORED" if n_known > 0 else "UNEXPLORED"
        print(f"\n  Bucket {b}  [{status}]  {n_cand:,} candidates  |  {n_known} known targets")
        print(f"  {'Ranker':<42}  " + "  ".join(f"@{m:>2}%" for m in COVERAGE_M))
        print(f"  {'─'*42}  " + "  ".join("─"*5 for _ in COVERAGE_M))
        for _, row in b_rows.iterrows():
            vals = "  ".join(f"{row[f'Coverage@{m}%']:>5}" for m in COVERAGE_M)
            print(f"  {row['Ranker']:<42}  {vals}")


def print_summary_table(summary: pd.DataFrame):
    print(f"\n  {'─'*70}")
    print(f"  Weighted-mean coverage across explored buckets (weight = N_known):")
    print(f"  {'─'*70}")
    print(f"  {'Ranker':<42}  " + "  ".join(f"@{m:>2}%" for m in COVERAGE_M))
    print(f"  {'─'*42}  " + "  ".join("─"*5 for _ in COVERAGE_M))
    for _, row in summary.iterrows():
        vals = "  ".join(f"{row[f'Coverage@{m}%']:>5}" for m in COVERAGE_M)
        print(f"  {row['Ranker']:<42}  {vals}")


# ════════════════════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════════════════════

_RANKER_STYLE = {
    "PEN-diff (PANACEA native)":    ("#d62728", "--"),
    "PPR-diff (PANACEA baseline)":  ("#ff7f0e", "--"),
    "dist-diff (PANACEA baseline)": ("#bcbd22", "--"),
    "Random Forest":                ("#2ca02c", "-"),
    "LightGBM LambdaMART":          ("#1f77b4", "-"),
    "CatBoost YetiRank":            ("#9467bd", "-"),
}


def _parse(val: str) -> float:
    return float("nan") if val == "—" else float(str(val).rstrip("%"))


def plot_results(all_cov: pd.DataFrame, summary: pd.DataFrame,
                 fig_dir: Path, cancer: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not installed — no plots saved.")
        return

    explored = sorted(all_cov[all_cov["N_known"] > 0]["Bucket"].unique())
    ranker_order = summary["Ranker"].tolist()

    def color(r):  return _RANKER_STYLE.get(r, ("#555555", "-"))[0]
    def ls(r):     return _RANKER_STYLE.get(r, ("#555555", "-"))[1]

    # ── Figure 1: Coverage curves (line plot per bucket + weighted mean) ──
    n_plots = len(explored) + 1
    ncols   = min(3, n_plots)
    nrows   = (n_plots + ncols - 1) // ncols
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows),
                                squeeze=False)
    axes1 = axes1.flatten()

    for ax_i, b in enumerate(explored):
        ax   = axes1[ax_i]
        bdf  = all_cov[all_cov["Bucket"] == b]
        n_k  = bdf["N_known"].iloc[0]
        n_c  = bdf["N_candidates"].iloc[0]
        for _, row in bdf.iterrows():
            r    = row["Ranker"]
            covs = [_parse(row[f"Coverage@{m}%"]) for m in COVERAGE_M]
            ax.plot(COVERAGE_M, covs, marker="o", label=r,
                    color=color(r), linestyle=ls(r), linewidth=2)
        ax.set_title(f"Bucket {b}  ({n_c:,} candidates | {n_k} known)", fontsize=9)
        ax.set_xlabel("Top-m% of bucket")
        ax.set_ylabel("Coverage of known targets (%)")
        ax.set_xticks(COVERAGE_M)
        ax.set_xticklabels([f"{m}%" for m in COVERAGE_M])
        ax.set_ylim(-5, 108)
        ax.grid(True, alpha=0.3)

    # Summary subplot
    ax = axes1[len(explored)]
    for _, row in summary.iterrows():
        r    = row["Ranker"]
        covs = [_parse(row[f"Coverage@{m}%"]) for m in COVERAGE_M]
        ax.plot(COVERAGE_M, covs, marker="o", label=r,
                color=color(r), linestyle=ls(r), linewidth=2)
    ax.set_title("Weighted Mean (across explored buckets)", fontsize=9)
    ax.set_xlabel("Top-m% of bucket")
    ax.set_ylabel("Coverage of known targets (%)")
    ax.set_xticks(COVERAGE_M)
    ax.set_xticklabels([f"{m}%" for m in COVERAGE_M])
    ax.set_ylim(-5, 108)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    for i in range(len(explored) + 1, len(axes1)):
        axes1[i].set_visible(False)

    fig1.suptitle(
        f"Coverage@m% — All Rankers vs PANACEA PEN-diff  ({cancer.capitalize()} Cancer)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out1 = fig_dir / f"fig_panacea_vs_models_coverage_curves_{cancer}.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"    {out1}")

    # ── Figure 2: Bar chart — Coverage@1% per bucket (one subplot per bucket) ──
    SHORT = {
        "PEN-diff (PANACEA native)":    "PEN-diff\n(PANACEA)",
        "PPR-diff (PANACEA baseline)":  "PPR-diff\n(PANACEA)",
        "dist-diff (PANACEA baseline)": "dist-diff\n(PANACEA)",
        "Random Forest":                "Random\nForest",
        "LightGBM LambdaMART":          "LightGBM\nLambdaMART",
        "CatBoost YetiRank":            "CatBoost\nYetiRank",
    }

    n_plots2 = len(explored) + 1
    ncols2   = min(3, n_plots2)
    nrows2   = (n_plots2 + ncols2 - 1) // ncols2
    fig2, axes2 = plt.subplots(nrows2, ncols2,
                                figsize=(5.5 * ncols2, 4.5 * nrows2), squeeze=False)
    axes2 = axes2.flatten()

    xtick_labels = [SHORT.get(r, r) for r in ranker_order]
    x_pos        = np.arange(len(ranker_order))
    bar_colors   = [color(r) for r in ranker_order]

    def _draw_bars(ax, vals, title):
        bars = ax.bar(x_pos, vals, color=bar_colors, alpha=0.85, width=0.6)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1.0,
                        f"{v:.0f}%", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(xtick_labels, fontsize=8)
        ax.set_ylim(0, 118)
        ax.set_ylabel("Coverage@1% (%)")
        ax.set_title(title, fontsize=9)
        ax.axhline(100, color="gray", linewidth=0.8, linestyle=":")
        ax.grid(True, axis="y", alpha=0.3)

    for ax_i, b in enumerate(explored):
        bdf   = all_cov[all_cov["Bucket"] == b]
        n_k   = bdf["N_known"].iloc[0]
        n_c   = bdf["N_candidates"].iloc[0]
        vals  = [_parse(bdf.loc[bdf["Ranker"] == r, "Coverage@1%"].iloc[0])
                 if (bdf["Ranker"] == r).any() else float("nan")
                 for r in ranker_order]
        _draw_bars(axes2[ax_i], vals,
                   f"Bucket {b}  ({n_c:,} candidates | {n_k} known)")

    # Summary subplot
    sum_vals = [_parse(summary.loc[summary["Ranker"] == r, "Coverage@1%"].iloc[0])
                if (summary["Ranker"] == r).any() else float("nan")
                for r in ranker_order]
    _draw_bars(axes2[len(explored)], sum_vals,
               "Weighted Mean (across explored buckets)")

    for i in range(len(explored) + 1, len(axes2)):
        axes2[i].set_visible(False)

    fig2.suptitle(
        f"Coverage@1% — All Rankers vs PANACEA PEN-diff  ({cancer.capitalize()} Cancer)\n"
        "Fraction of known targets recovered in the top-1% of each bucket's ranked list",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out2 = fig_dir / f"fig_panacea_vs_models_coverage_bar_1pct_{cancer}.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"    {out2}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True, choices=["breast", "prostate"])
    args = ap.parse_args()
    cancer = args.cancer

    print(f"\n{'='*70}")
    print(f"  ALL MODELS vs PANACEA PEN-diff RANKING — {cancer.upper()}")
    print(f"  Bucket-level coverage (PANACEA evaluation protocol, Table 4)")
    print(f"{'='*70}\n")

    cancer_dir = OUTPUTS / cancer
    pairs_path = cancer_dir / "pairs_with_buckets.csv"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Missing: {pairs_path}")

    print("  Loading pairs_with_buckets.csv...")
    df = pd.read_csv(pairs_path)
    print(f"  {len(df):,} gene pairs  |  {int(df['is_known'].sum()):,} known targets\n")

    # ── Train all models on training split ──
    train_df = load_train_df(cancer_dir)
    print()

    lgbm_model = train_lgbm(train_df)
    rf_model   = train_random_forest(train_df)
    cb_model   = train_catboost(train_df)
    print()

    # ── Score ALL pairs with each model ──
    print("  Scoring all pairs...")
    df["score_lgbm"] = lgbm_model.predict(df[FULL_FEATURES])
    df["score_rf"]   = rf_model.predict_proba(df[FULL_FEATURES])[:, 1]
    if cb_model is not None:
        from catboost import Pool
        pool_all = Pool(
            data=df[FULL_FEATURES].values.astype(np.float32),
            feature_names=FULL_FEATURES,
        )
        df["score_cb"] = cb_model.predict(pool_all)
    print("  Done.\n")

    # ════════════════════════════════════════════════════════════════════════
    # COMPUTE COVERAGE FOR EACH RANKER
    # ════════════════════════════════════════════════════════════════════════

    print(f"  {'─'*70}")
    print(f"  Bucket-Level Coverage  (% of known targets in top-m% of bucket)")
    print(f"  {'─'*70}")

    # PANACEA score baselines (sort descending — higher = better)
    tables = [
        compute_coverage_table(df, "pen_diff",  "PEN-diff (PANACEA native)"),
        compute_coverage_table(df, "ppr_diff",  "PPR-diff (PANACEA baseline)"),
        compute_coverage_table(df, "dist_diff", "dist-diff (PANACEA baseline)"),
        compute_coverage_table(df, "score_rf",  "Random Forest"),
        compute_coverage_table(df, "score_lgbm","LightGBM LambdaMART"),
    ]
    if cb_model is not None:
        tables.append(compute_coverage_table(df, "score_cb", "CatBoost YetiRank"))

    all_cov = pd.concat(tables, ignore_index=True)

    print_bucket_table(all_cov)

    summary = summarise(all_cov)
    print_summary_table(summary)

    # ── Plots ──
    print(f"\n  Saving plots...")
    FIGURES.mkdir(exist_ok=True)
    plot_results(all_cov, summary, FIGURES, cancer)

    # ── Save CSVs ──
    out_cov     = cancer_dir / "panacea_vs_models_coverage.csv"
    out_summary = cancer_dir / "panacea_vs_models_summary.csv"
    all_cov.to_csv(out_cov,     index=False)
    summary.to_csv(out_summary, index=False)

    print(f"    {out_cov}")
    print(f"    {out_summary}\n")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
