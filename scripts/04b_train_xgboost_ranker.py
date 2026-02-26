"""
Train an XGBoost learning-to-rank model (rank:ndcg objective) on the same
bucket-guided training data used for the LightGBM ranker.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
MODELS = PROJECT_ROOT / "models"

FEATURES = [
    "pen_diff", "dist_diff", "ppr_diff",
    "bucket_pen", "bucket_dist", "bucket_ppr",
    "pos_in_bucket_pen",
]


def load_rank_csv(path: Path):
    df = pd.read_csv(path)
    df = df.sort_values("group_id").reset_index(drop=True)
    group_sizes = df.groupby("group_id").size().to_numpy()
    X = df[FEATURES].values.astype(np.float32)
    y = df["label"].astype(int).values
    return df, X, y, group_sizes


def recall_at_k_by_group(df_scored: pd.DataFrame, k: int) -> float:
    hits = 0
    total = 0
    for _, g in df_scored.groupby("group_id"):
        g_sorted = g.sort_values("pred", ascending=False)
        if (g_sorted.head(k)["label"] == 1).any():
            hits += 1
        total += 1
    return hits / total if total > 0 else 0.0


def ndcg_at_k_by_group(df_scored: pd.DataFrame, k: int) -> float:
    """Compute mean NDCG@k across groups (each group has at most 1 positive)."""
    scores = []
    for _, g in df_scored.groupby("group_id"):
        g_sorted = g.sort_values("pred", ascending=False).reset_index(drop=True)
        # find position of positive (0-indexed)
        pos_idx = g_sorted[g_sorted["label"] == 1].index
        if len(pos_idx) == 0:
            continue
        rank = pos_idx[0] + 1  # 1-indexed
        if rank <= k:
            dcg = 1.0 / np.log2(rank + 1)
        else:
            dcg = 0.0
        idcg = 1.0 / np.log2(2)  # ideal: positive at rank 1
        scores.append(dcg / idcg)
    return float(np.mean(scores)) if scores else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True, choices=["breast", "prostate"])
    ap.add_argument("--n_estimators", type=int, default=1000)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--colsample_bytree", type=float, default=0.9)
    ap.add_argument("--early_stopping_rounds", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cancer_dir = OUTPUTS / args.cancer
    train_path = cancer_dir / "train_rank_final.csv"
    test_path = cancer_dir / "test_rank.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Missing train/test_rank.csv. Run scripts/03_prepare_training_data.py first."
        )

    print(f"[XGBoost] Loading data for {args.cancer} ...")
    train_df, X_train, y_train, group_train = load_rank_csv(train_path)
    test_df, X_test, y_test, group_test = load_rank_csv(test_path)

    # XGBoost DMatrix with qid (group sizes → cumulative qid array)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURES)
    dtrain.set_group(group_train)

    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURES)
    dtest.set_group(group_test)

    params = {
        "objective": "rank:ndcg",
        "eval_metric": ["ndcg@5", "ndcg@10"],
        "eta": args.learning_rate,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "seed": args.seed,
        "nthread": -1,
        # lambdarank normalisation – ndcg-based gradients
        "lambdarank_normalization": "true",
    }

    print(f"[XGBoost] Training (up to {args.n_estimators} rounds) ...")
    evals_result = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=args.n_estimators,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=args.early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=100,
    )

    # Predict on test
    preds = booster.predict(dtest)
    scored = test_df.copy()
    scored["pred"] = preds

    r1 = recall_at_k_by_group(scored, 1)
    r3 = recall_at_k_by_group(scored, 3)
    r5 = recall_at_k_by_group(scored, 5)
    r10 = recall_at_k_by_group(scored, 10)
    n5 = ndcg_at_k_by_group(scored, 5)
    n10 = ndcg_at_k_by_group(scored, 10)

    print("\n=== XGBoost Evaluation (test set) ===")
    print(f"Recall@1  : {r1:.4f}")
    print(f"Recall@3  : {r3:.4f}")
    print(f"Recall@5  : {r5:.4f}")
    print(f"Recall@10 : {r10:.4f}")
    print(f"NDCG@5    : {n5:.4f}")
    print(f"NDCG@10   : {n10:.4f}")

    # Save model + metrics
    out_dir = MODELS / args.cancer
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "xgb_ranker.ubj"
    booster.save_model(str(model_path))
    joblib.dump(booster, out_dir / "xgb_ranker.joblib")

    # Feature importance
    imp = booster.get_score(importance_type="gain")
    imp_df = (
        pd.DataFrame({"feature": list(imp.keys()), "gain_importance": list(imp.values())})
        .sort_values("gain_importance", ascending=False)
    )
    imp_df.to_csv(out_dir / "xgb_feature_importance.csv", index=False)

    # Save scored test for comparison script
    scored.to_csv(out_dir / "xgb_test_scored.csv", index=False)

    print(f"\n[OK] Saved XGBoost model to: {model_path}")
    print(f"[OK] Feature importance: {out_dir / 'xgb_feature_importance.csv'}")


if __name__ == "__main__":
    main()
