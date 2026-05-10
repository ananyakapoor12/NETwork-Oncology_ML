"""Train a LightGBM LambdaMART ranker on bucket-guided training pairs."""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
MODELS = PROJECT_ROOT / "models"
MODELS.mkdir(exist_ok=True)

FEATURES = [
    "pen_diff", "dist_diff", "ppr_diff",
    "bucket_pen", "bucket_dist", "bucket_ppr",
    "pos_in_bucket_pen"
]

def load_rank_csv(path: Path):
    df = pd.read_csv(path)
    # sort by group_id so we can build group sizes
    df = df.sort_values("group_id").reset_index(drop=True)
    group_sizes = df.groupby("group_id").size().to_numpy()
    X = df[FEATURES]
    y = df["label"].astype(int)
    return df, X, y, group_sizes

def recall_at_k_by_group(df_scored: pd.DataFrame, k: int) -> float:
    # Each group has exactly one positive (by construction). Recall@K = fraction where positive is in top K.
    hits = 0
    total = 0
    for gid, g in df_scored.groupby("group_id"):
        g_sorted = g.sort_values("pred", ascending=False)
        topk = g_sorted.head(k)
        # hit if any positive in topk
        if (topk["label"] == 1).any():
            hits += 1
        total += 1
    return hits / total if total > 0 else 0.0

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True, choices=["breast", "prostate"])
    ap.add_argument("--num_leaves", type=int, default=63)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--n_estimators", type=int, default=2000)
    ap.add_argument("--early_stopping_rounds", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cancer_dir = OUTPUTS / args.cancer
    train_path = cancer_dir / "train_rank_final.csv"
    test_path = cancer_dir / "test_rank.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing train/test_rank.csv. Run scripts/03_prepare_training_data.py first.")

    train_df, X_train, y_train, group_train = load_rank_csv(train_path)
    test_df,  X_test,  y_test,  group_test  = load_rank_csv(test_path)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=args.seed,
        n_jobs=-1
    )

    ranker.fit(
        X_train, y_train,
        group=group_train,
        eval_set=[(X_test, y_test)],
        eval_group=[group_test],
        eval_at=[1, 3, 5, 10],
        callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=True)]
    )

    # Save model
    out_dir = MODELS / args.cancer
    out_dir.mkdir(parents=True, exist_ok=True)

    booster = ranker.booster_
    model_path = out_dir / "lgbm_ranker.txt"
    booster.save_model(str(model_path))

    # Also save full sklearn wrapper (optional)
    joblib.dump(ranker, out_dir / "lgbm_ranker.joblib")

    # Evaluate recall@K on the test groups
    preds = ranker.predict(X_test)
    scored = test_df.copy()
    scored["pred"] = preds

    r1 = recall_at_k_by_group(scored, 1)
    r3 = recall_at_k_by_group(scored, 3)
    r5 = recall_at_k_by_group(scored, 5)
    r10 = recall_at_k_by_group(scored, 10)

    print("\n=== Evaluation (test mini-queries) ===")
    print(f"Recall@1  : {r1:.4f}")
    print(f"Recall@3  : {r3:.4f}")
    print(f"Recall@5  : {r5:.4f}")
    print(f"Recall@10 : {r10:.4f}")

    # Feature importance
    imp = pd.DataFrame({
        "feature": FEATURES,
        "gain_importance": booster.feature_importance(importance_type="gain"),
        "split_importance": booster.feature_importance(importance_type="split"),
    }).sort_values("gain_importance", ascending=False)
    imp.to_csv(out_dir / "feature_importance.csv", index=False)

    print(f"\n[OK] Saved model to: {model_path}")
    print(f"[OK] Saved feature importance to: {out_dir / 'feature_importance.csv'}")

if __name__ == "__main__":
    main()
