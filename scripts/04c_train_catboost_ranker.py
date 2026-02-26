"""
Train a CatBoost learning-to-rank model (YetiRank objective) on the same
bucket-guided training data used for the LightGBM and XGBoost rankers.

YetiRank directly optimises NDCG, making it well aligned with the therapeutic
prioritisation objective.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
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
    X = df[FEATURES].values.astype(np.float32)
    y = df["label"].astype(int).values
    groups = df["group_id"].values
    return df, X, y, groups


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
    scores = []
    for _, g in df_scored.groupby("group_id"):
        g_sorted = g.sort_values("pred", ascending=False).reset_index(drop=True)
        pos_idx = g_sorted[g_sorted["label"] == 1].index
        if len(pos_idx) == 0:
            continue
        rank = pos_idx[0] + 1
        dcg = (1.0 / np.log2(rank + 1)) if rank <= k else 0.0
        idcg = 1.0 / np.log2(2)
        scores.append(dcg / idcg)
    return float(np.mean(scores)) if scores else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True, choices=["breast", "prostate"])
    ap.add_argument("--n_estimators", type=int, default=1000)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--l2_leaf_reg", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cancer_dir = OUTPUTS / args.cancer
    train_path = cancer_dir / "train_rank_final.csv"
    test_path = cancer_dir / "test_rank.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Missing train/test_rank.csv. Run scripts/03_prepare_training_data.py first."
        )

    print(f"[CatBoost] Loading data for {args.cancer} ...")
    train_df, X_train, y_train, groups_train = load_rank_csv(train_path)
    test_df, X_test, y_test, groups_test = load_rank_csv(test_path)

    train_pool = Pool(
        data=X_train,
        label=y_train,
        group_id=groups_train,
        feature_names=FEATURES,
    )
    test_pool = Pool(
        data=X_test,
        label=y_test,
        group_id=groups_test,
        feature_names=FEATURES,
    )

    model = CatBoostRanker(
        loss_function="YetiRank",   # directly optimises NDCG
        eval_metric="NDCG:top=10",
        iterations=args.n_estimators,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=args.seed,
        early_stopping_rounds=50,
        verbose=100,
    )

    print(f"[CatBoost] Training (up to {args.n_estimators} iterations) ...")
    model.fit(train_pool, eval_set=test_pool)

    # Predict & evaluate
    preds = model.predict(test_pool)
    scored = test_df.copy()
    scored["pred"] = preds

    r1 = recall_at_k_by_group(scored, 1)
    r3 = recall_at_k_by_group(scored, 3)
    r5 = recall_at_k_by_group(scored, 5)
    r10 = recall_at_k_by_group(scored, 10)
    n5 = ndcg_at_k_by_group(scored, 5)
    n10 = ndcg_at_k_by_group(scored, 10)

    print("\n=== CatBoost Evaluation (test set) ===")
    print(f"Recall@1  : {r1:.4f}")
    print(f"Recall@3  : {r3:.4f}")
    print(f"Recall@5  : {r5:.4f}")
    print(f"Recall@10 : {r10:.4f}")
    print(f"NDCG@5    : {n5:.4f}")
    print(f"NDCG@10   : {n10:.4f}")

    # Save model
    out_dir = MODELS / args.cancer
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "catboost_ranker.cbm"
    model.save_model(str(model_path))
    joblib.dump(model, out_dir / "catboost_ranker.joblib")

    # Feature importance
    imp_df = pd.DataFrame({
        "feature": FEATURES,
        "gain_importance": model.get_feature_importance(train_pool, type="PredictionValuesChange"), # type: ignore
    }).sort_values("gain_importance", ascending=False)
    imp_df.to_csv(out_dir / "catboost_feature_importance.csv", index=False)

    # Save scored test for comparison script
    scored.to_csv(out_dir / "catboost_test_scored.csv", index=False)

    print(f"\n[OK] Saved CatBoost model to: {model_path}")
    print(f"[OK] Feature importance: {out_dir / 'catboost_feature_importance.csv'}")


if __name__ == "__main__":
    main()
