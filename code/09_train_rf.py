"""Train a Random Forest classifier used as a ranker via predict_proba scores."""
from pathlib import Path
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
MODELS = PROJECT_ROOT / "models"

FEATURES = [
    "pen_diff", "dist_diff", "ppr_diff",
    "bucket_pen", "bucket_dist", "bucket_ppr",
    "pos_in_bucket_pen"
]


def recall_at_k(df_scored, k):
    """Compute Recall@K for ranking groups."""
    hits = 0
    for gid, g in df_scored.groupby("group_id"):
        g_sorted = g.sort_values("pred", ascending=False)
        if (g_sorted.head(k)["label"] == 1).any():
            hits += 1
    return hits / df_scored["group_id"].nunique()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True, choices=["breast", "prostate"])
    args = ap.parse_args()
    
    cancer_dir = OUTPUTS / args.cancer
    train_path = cancer_dir / "train_rank_final.csv"
    if not train_path.exists():
        train_path = cancer_dir / "train_rank.csv"
    test_path = cancer_dir / "test_rank.csv"
    
    print(f"\n{'='*70}")
    print(f"  RANDOM FOREST RANKER - {args.cancer.upper()}")
    print(f"{'='*70}\n")
    
    # Load data
    train_df = pd.read_csv(train_path).sort_values("group_id")
    test_df = pd.read_csv(test_path).sort_values("group_id")
    
    X_train = train_df[FEATURES]
    y_train = train_df["label"].astype(int)
    X_test = test_df[FEATURES]
    
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}\n")
    print(f"  Training Random Forest...")
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    preds = rf.predict_proba(X_test)[:, 1]
    
    # Evaluate
    test_df_copy = test_df.copy()
    test_df_copy["pred"] = preds
    
    print(f"\n=== Evaluation (test mini-queries) ===")
    print(f"Recall@1  : {recall_at_k(test_df_copy, 1):.4f}")
    print(f"Recall@3  : {recall_at_k(test_df_copy, 3):.4f}")
    print(f"Recall@5  : {recall_at_k(test_df_copy, 5):.4f}")
    print(f"Recall@10 : {recall_at_k(test_df_copy, 10):.4f}\n")
    
    # Save model
    out_dir = MODELS / args.cancer
    out_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(rf, out_dir / "rf_ranker.joblib")
    
    # Feature importance
    pd.DataFrame({
        "feature": FEATURES,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False).to_csv(
        out_dir / "rf_feature_importance.csv", index=False
    )
    
    print(f"[OK] Saved model to: {out_dir / 'rf_ranker.joblib'}")
    print(f"[OK] Saved feature importance to: {out_dir / 'rf_feature_importance.csv'}\n")


if __name__ == "__main__":
    main()