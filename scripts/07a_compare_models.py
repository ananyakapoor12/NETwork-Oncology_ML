"""
Compare all trained ranking models on the test set for a given cancer type.

Evaluates: LightGBM LambdaRank, XGBoost LambdaRank, CatBoost YetiRank.
Metrics:   Recall@K  (K = 1, 3, 5, 10)
           NDCG@K    (K = 5, 10)

Also plots:
  • Bar chart of Recall@K and NDCG@K across models (per cancer type)
  • Per-bucket Recall@10 comparison (shows intra-bucket ranking quality)
  • Explored vs unexplored bucket summary with top-ranked novel candidates
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS  = PROJECT_ROOT / "outputs"
MODELS   = PROJECT_ROOT / "models"
FIGURES  = PROJECT_ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

FEATURES = [
    "pen_diff", "dist_diff", "ppr_diff",
    "bucket_pen", "bucket_dist", "bucket_ppr",
    "pos_in_bucket_pen",
]


# ─────────────────────────── metric helpers ───────────────────────────────── #

def recall_at_k(df: pd.DataFrame, k: int) -> float:
    hits, total = 0, 0
    for _, g in df.groupby("group_id"):
        if (g.sort_values("pred", ascending=False).head(k)["label"] == 1).any():
            hits += 1
        total += 1
    return hits / total if total > 0 else 0.0


def ndcg_at_k(df: pd.DataFrame, k: int) -> float:
    vals = []
    for _, g in df.groupby("group_id"):
        g2 = g.sort_values("pred", ascending=False).reset_index(drop=True)
        pos = g2[g2["label"] == 1].index
        if len(pos) == 0:
            continue
        rank = pos[0] + 1
        dcg  = (1.0 / np.log2(rank + 1)) if rank <= k else 0.0
        vals.append(dcg / (1.0 / np.log2(2)))
    return float(np.mean(vals)) if vals else 0.0


def per_bucket_recall(df: pd.DataFrame, k: int, bucket_col: str = "bucket_pen") -> dict:
    """Return {bucket_id: recall@k} for each bucket present in df."""
    result = {}
    for bid, bdf in df.groupby(bucket_col):
        result[int(bid)] = recall_at_k(bdf, k) # type: ignore
    return result


# ─────────────────────────── model loading ────────────────────────────────── #

def score_with_lgbm(test_df: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    import joblib
    model_path = model_dir / "lgbm_ranker.joblib"
    if not model_path.exists():
        return None # type: ignore
    ranker = joblib.load(model_path)
    scored = test_df.copy()
    scored["pred"] = ranker.predict(test_df[FEATURES])
    return scored


def score_with_xgb(test_df: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    try:
        import xgboost as xgb
    except ImportError:
        print("[WARN] xgboost not installed – skipping XGBoost comparison.")
        return None # type: ignore
    model_path = model_dir / "xgb_ranker.ubj"
    if not model_path.exists():
        print(f"[WARN] {model_path} not found – run 04b_train_xgboost_ranker.py first.")
        return None # type: ignore
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    dmat = xgb.DMatrix(test_df[FEATURES].values.astype("float32"), feature_names=FEATURES)
    scored = test_df.copy()
    scored["pred"] = booster.predict(dmat)
    return scored


def score_with_catboost(test_df: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    try:
        from catboost import CatBoostRanker, Pool
    except ImportError:
        print("[WARN] catboost not installed – skipping CatBoost comparison.")
        return None # type: ignore
    model_path = model_dir / "catboost_ranker.cbm"
    if not model_path.exists():
        print(f"[WARN] {model_path} not found – run 04c_train_catboost_ranker.py first.")
        return None # type: ignore
    model = CatBoostRanker()
    model.load_model(str(model_path))
    pool = Pool(
        data=test_df[FEATURES].values.astype("float32"),
        label=test_df["label"].values,
        group_id=test_df["group_id"].values,
        feature_names=FEATURES,
    )
    scored = test_df.copy()
    scored["pred"] = model.predict(pool)
    return scored


# ─────────────────────────── plotting helpers ─────────────────────────────── #

def plot_metric_comparison(results: dict, cancer: str, out_dir: Path):
    """
    results = {"LightGBM": {...metrics...}, "XGBoost": {...}, "CatBoost": {...}}
    """
    metric_keys = ["Recall@1", "Recall@3", "Recall@5", "Recall@10", "NDCG@5", "NDCG@10"]
    models = list(results.keys())
    x = np.arange(len(metric_keys))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, model in enumerate(models):
        vals = [results[model].get(m, 0.0) for m in metric_keys]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=7
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison – {cancer.capitalize()} Cancer")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"fig_model_comparison_{cancer}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {path}")


def plot_per_bucket_recall(bucket_results: dict, cancer: str, out_dir: Path, k: int = 10):
    """
    bucket_results = {"LightGBM": {0: 0.8, 1: 0.6, ...}, ...}
    """
    all_buckets = sorted({b for v in bucket_results.values() for b in v})
    models = list(bucket_results.keys())
    x = np.arange(len(all_buckets))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        vals = [bucket_results[model].get(b, 0.0) for b in all_buckets]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Bucket {b}" for b in all_buckets], rotation=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(f"Recall@{k}")
    ax.set_title(f"Per-Bucket Recall@{k} – {cancer.capitalize()} Cancer")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"fig_per_bucket_recall_{cancer}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {path}")


def plot_explored_vs_unexplored(policy_df: pd.DataFrame, cancer: str, out_dir: Path):
    """
    Show the PANACEA delta histogram (explored vs unexplored buckets) and
    the W_norm allocation weight for each bucket.
    """
    buckets = policy_df["bucket"].values
    known_counts = policy_df["known_count"].values
    w_norm = policy_df["W_norm"].values
    x = np.arange(len(buckets))

    colors = ["#2196F3" if k > 0 else "#FF9800" for k in known_counts]
    labels = [
        f"Explored\n(n={k})" if k > 0 else "Unexplored\n(novel)"
        for k in known_counts
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    bars = ax1.bar(x, known_counts, color=colors, edgecolor="white")
    for bar, lbl in zip(bars, labels):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            lbl, ha="center", va="bottom", fontsize=8
        )
    ax1.set_ylabel("No. Known Target Combinations")
    ax1.set_title(
        f"Explored vs Unexplored Histogram Buckets – {cancer.capitalize()} Cancer\n"
        "(Blue = known combinations present; Orange = novel / unexplored)"
    )
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, w_norm, color=colors, edgecolor="white")
    for xi, w in zip(x, w_norm):
        ax2.text(xi, w + 0.005, f"{w:.2f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x)
    bkt_pen = policy_df["range"].values if "range" in policy_df else [str(b) for b in buckets]
    ax2.set_xticklabels([f"B{b}\n{r}" for b, r in zip(buckets, bkt_pen)], fontsize=7)
    ax2.set_ylabel("Allocation Weight W_norm")
    ax2.set_title("Bucket Sampling Weight (higher = more exploration focus)")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = out_dir / f"fig_explored_vs_unexplored_{cancer}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {path}")


# ─────────────────────────── main ─────────────────────────────────────────── #

def evaluate_cancer(cancer: str):
    cancer_dir = OUTPUTS / cancer
    model_dir  = MODELS / cancer
    test_path  = cancer_dir / "test_rank.csv"
    policy_path = cancer_dir / "bucket_policy.csv"

    if not test_path.exists():
        print(f"[SKIP] {cancer}: test_rank.csv not found.")
        return

    test_df = pd.read_csv(test_path).sort_values("group_id").reset_index(drop=True)

    scored_models = {}
    for name, fn in [
        ("LightGBM", score_with_lgbm),
        ("XGBoost",  score_with_xgb),
        ("CatBoost", score_with_catboost),
    ]:
        scored = fn(test_df, model_dir)
        if scored is not None:
            scored_models[name] = scored

    if not scored_models:
        print(f"[SKIP] {cancer}: no models found.")
        return

    # ── Metric table ────────────────────────────────────────────────────── #
    rows = []
    bucket_results = {}
    for name, scored in scored_models.items():
        metrics = {
            "Model": name,
            "Cancer": cancer,
            "Recall@1":  recall_at_k(scored, 1),
            "Recall@3":  recall_at_k(scored, 3),
            "Recall@5":  recall_at_k(scored, 5),
            "Recall@10": recall_at_k(scored, 10),
            "NDCG@5":    ndcg_at_k(scored, 5),
            "NDCG@10":   ndcg_at_k(scored, 10),
        }
        rows.append(metrics)
        bucket_results[name] = per_bucket_recall(scored, k=10)
        print(f"\n[{name}] {cancer.capitalize()}: "
              f"R@1={metrics['Recall@1']:.4f}  R@3={metrics['Recall@3']:.4f}  "
              f"R@5={metrics['Recall@5']:.4f}  R@10={metrics['Recall@10']:.4f}  "
              f"NDCG@5={metrics['NDCG@5']:.4f}  NDCG@10={metrics['NDCG@10']:.4f}")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(MODELS / cancer / "model_comparison.csv", index=False)
    print(f"\n[OK] Saved comparison table to: {MODELS / cancer / 'model_comparison.csv'}")

    # ── Plots ───────────────────────────────────────────────────────────── #
    results_dict = {
        name: {k: v for k, v in row.items() if k not in ("Model", "Cancer")}
        for name, row in zip(scored_models.keys(), rows)
    }
    plot_metric_comparison(results_dict, cancer, FIGURES)
    plot_per_bucket_recall(bucket_results, cancer, FIGURES)

    if policy_path.exists():
        policy_df = pd.read_csv(policy_path)
        plot_explored_vs_unexplored(policy_df, cancer, FIGURES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cancer", required=True,
        choices=["breast", "prostate", "both"],
        help="Which cancer type(s) to evaluate."
    )
    args = ap.parse_args()

    cancers = ["breast", "prostate"] if args.cancer == "both" else [args.cancer]
    for cancer in cancers:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {cancer.upper()}")
        print(f"{'='*60}")
        evaluate_cancer(cancer)


if __name__ == "__main__":
    main()
