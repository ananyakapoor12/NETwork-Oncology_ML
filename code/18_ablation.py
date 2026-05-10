"""Measure each feature group's contribution by systematically ablating inputs."""
from pathlib import Path
import pandas as pd 
import numpy as np 
import lightgbm as lgb 
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"

FULL_FEATURES = [
    "pen_diff", "dist_diff", "ppr_diff",
    "bucket_pen", "bucket_dist", "bucket_ppr",
    "pos_in_bucket_pen"
]


def recall_at_k(df_scored, k):
    """Compute Recall@K."""
    hits = 0
    for gid, g in df_scored.groupby("group_id"):
        g_sorted = g.sort_values("pred", ascending=False)
        if (g_sorted.head(k)["label"] == 1).any():
            hits += 1
    return hits / df_scored["group_id"].nunique()


def train_and_evaluate(train_df, test_df, features, name):
    """Train LightGBM with given features and evaluate."""
    print(f"  Training: {name}...", end='', flush=True)
    
    train_df = train_df.sort_values("group_id").reset_index(drop=True)
    test_df = test_df.sort_values("group_id").reset_index(drop=True)
    
    group_train = train_df.groupby("group_id").size().to_numpy().copy()
    
    X_train = train_df[features].copy()
    y_train = train_df["label"].astype(int).copy()
    X_test = test_df[features].copy()
    
    # Train LightGBM (faster settings)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        num_leaves=31,  # Reduced from 63
        learning_rate=0.1,  # Increased from 0.05
        n_estimators=100,  # Reduced from 500
        random_state=42,
        verbosity=-1,
        force_col_wise=True  # Faster
    )
    
    model.fit(X_train, y_train, group=group_train)
    preds = model.predict(X_test)
    
    test_copy = test_df.copy()
    test_copy["pred"] = preds
    
    r1 = recall_at_k(test_copy, 1)
    
    print(f" Recall@1: {r1:.4f}")
    
    return r1


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True, choices=["breast", "prostate"])
    args = ap.parse_args()
    
    print(f"\n{'='*70}")
    print(f"  ABLATION STUDY - {args.cancer.upper()}")
    print(f"{'='*70}\n")
    
    cancer_dir = OUTPUTS / args.cancer
    
    # Load data
    train_path = cancer_dir / "train_rank_final.csv"
    if not train_path.exists():
        train_path = cancer_dir / "train_rank.csv"
    
    print(f"  Loading data from {train_path.name}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(cancer_dir / "test_rank.csv")
    print(f"  Train: {len(train_df):,} rows, Test: {len(test_df):,} rows\n")
    
    results = []
    
    # ══════════════════════════════════════════════════════════════
    # BASELINE
    # ══════════════════════════════════════════════════════════════
    
    print("BASELINE:")
    r1 = train_and_evaluate(train_df, test_df, FULL_FEATURES, "Full model (all 7 features)")
    results.append({"Ablation": "Baseline (all features)", "Recall@1": f"{r1:.4f}", "Delta": "—"})
    baseline = r1
    
    # ══════════════════════════════════════════════════════════════
    # FEATURE ABLATION (remove one at a time)
    # ══════════════════════════════════════════════════════════════
    
    print(f"\nFEATURE ABLATION (remove one feature at a time):")
    
    for feature in FULL_FEATURES:
        ablated = [f for f in FULL_FEATURES if f != feature]
        name = f"Without {feature}"
        r1 = train_and_evaluate(train_df, test_df, ablated, name)
        delta = r1 - baseline
        results.append({
            "Ablation": f"Remove {feature}",
            "Recall@1": f"{r1:.4f}",
            "Delta": f"{delta:+.4f}"
        })
    
    # ══════════════════════════════════════════════════════════════
    # GROUP ABLATION
    # ══════════════════════════════════════════════════════════════
    
    print(f"\nGROUP ABLATION:")
    
    # Remove all bucket features
    no_buckets = ["pen_diff", "dist_diff", "ppr_diff"]
    r1 = train_and_evaluate(train_df, test_df, no_buckets, "Without ALL bucket features")
    delta = r1 - baseline
    results.append({
        "Ablation": "Remove all bucket features",
        "Recall@1": f"{r1:.4f}",
        "Delta": f"{delta:+.4f}"
    })
    
    # Only buckets
    only_buckets = ["bucket_pen", "bucket_dist", "bucket_ppr", "pos_in_bucket_pen"]
    r1 = train_and_evaluate(train_df, test_df, only_buckets, "Only bucket features")
    delta = r1 - baseline
    results.append({
        "Ablation": "Only bucket features",
        "Recall@1": f"{r1:.4f}",
        "Delta": f"{delta:+.4f}"
    })
    
    # Only PEN features
    only_pen = ["pen_diff", "bucket_pen", "pos_in_bucket_pen"]
    r1 = train_and_evaluate(train_df, test_df, only_pen, "Only PEN features")
    delta = r1 - baseline
    results.append({
        "Ablation": "Only PEN features",
        "Recall@1": f"{r1:.4f}",
        "Delta": f"{delta:+.4f}"
    })
    
    # ══════════════════════════════════════════════════════════════
    # HARD NEGATIVE ABLATION
    # ══════════════════════════════════════════════════════════════
    
    if (cancer_dir / "train_rank.csv").exists() and train_path.name == "train_rank_final.csv":
        print(f"\nHARD NEGATIVE ABLATION:")
        
        train_regular = pd.read_csv(cancer_dir / "train_rank.csv")
        r1 = train_and_evaluate(train_regular, test_df, FULL_FEATURES, "Without hard negatives")
        delta = r1 - baseline
        results.append({
            "Ablation": "No hard negatives",
            "Recall@1": f"{r1:.4f}",
            "Delta": f"{delta:+.4f}"
        })
    
    # ══════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════════════════════
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(cancer_dir / "ablation_study.csv", index=False)
    
    print(f"\n{'='*70}")
    print("ABLATION SUMMARY:")
    print(f"{'='*70}\n")
    print(results_df.to_string(index=False))
    print(f"\n{'='*70}")
    print(f"\nSaved to: {cancer_dir / 'ablation_study.csv'}\n")


if __name__ == "__main__":
    main()