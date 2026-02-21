#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
MODELS  = PROJECT_ROOT / "models"

FEATURES = [
    "pen_diff", "dist_diff", "ppr_diff",
    "bucket_pen", "bucket_dist", "bucket_ppr",
    "pos_in_bucket_pen"
]

EVAL_K = [1, 3, 5, 10]


# ─────────────────────────────────────────────
# Metric Helpers
# ─────────────────────────────────────────────

def recall_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Recall@K: fraction of positives in top-K."""
    total_pos = labels.sum()
    if total_pos == 0:
        return 0.0
    order = np.argsort(scores)[::-1]
    topk_labels = labels[order[:k]]
    return topk_labels.sum() / total_pos


def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """NDCG@K using sklearn."""
    if labels.sum() == 0:
        return 0.0
    try:
        return ndcg_score(labels.reshape(1, -1), scores.reshape(1, -1), k=k)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def assign_buckets(df: pd.DataFrame, meta: Dict) -> pd.DataFrame:
    """Assign bucket indices using meta edges."""
    df = df.copy()
    
    for score, key in [("pen_diff", "pen"), ("dist_diff", "dist"), ("ppr_diff", "ppr")]:
        edges = np.array(meta["edges"][key], dtype=float)
        df[f"bucket_{key}"] = pd.cut(df[score], bins=edges, labels=False, include_lowest=True) # type: ignore
    
    grp = df.groupby("bucket_pen")["pen_diff"]
    df["pos_in_bucket_pen"] = (
        (df["pen_diff"] - grp.transform("min"))
        / (grp.transform("max") - grp.transform("min") + 1e-12)
    )
    
    return df.dropna(subset=["bucket_pen", "bucket_dist", "bucket_ppr"])


def load_and_split(cancer_dir: Path, meta: Dict, seed: int = 42) -> Tuple:
    """Load pairs, identify explored buckets, split train/test."""
    df = pd.read_csv(cancer_dir / "pairs_k2_standardized.csv")
    df = assign_buckets(df, meta)
    df["bucket_pen"] = df["bucket_pen"].astype(int)
    df["bucket_dist"] = df["bucket_dist"].astype(int)
    df["bucket_ppr"] = df["bucket_ppr"].astype(int)
    
    # Identify explored buckets
    known_per_bucket = df.groupby("bucket_pen")["is_known"].sum()
    explored_ids = set(known_per_bucket[known_per_bucket > 0].index.astype(int).tolist())
    
    df_explored = df[df["bucket_pen"].isin(explored_ids)].copy()
    
    # Train/test split stratified by bucket
    rng = np.random.default_rng(seed)
    test_idx = []
    for b, grp in df_explored.groupby("bucket_pen"):
        pos_idx = grp.index[grp["is_known"] == 1].tolist()
        if len(pos_idx) == 0:
            continue
        n_test = max(1, int(len(pos_idx) * 0.2))
        chosen = rng.choice(pos_idx, size=n_test, replace=False)
        test_idx.extend(chosen.tolist())
    
    train_df = df_explored.drop(index=test_idx).reset_index(drop=True)
    test_df = df_explored.loc[test_idx].reset_index(drop=True)
    
    return train_df, test_df, explored_ids


# ─────────────────────────────────────────────
# Model 1: LightGBM LambdaMART
# ─────────────────────────────────────────────

def build_lgbm_groups(df: pd.DataFrame, neg_per_pos: int, rng, features: list) -> Tuple:
    """Build LGBMRanker training format: (X, y, group_sizes). Optimized for large datasets."""
    # Pre-allocate arrays for better performance
    pos_indices = df.index[df["is_known"] == 1].tolist()
    n_pos = len(pos_indices)
    
    if n_pos == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Estimate size and pre-allocate
    estimated_size = n_pos * (neg_per_pos + 1)
    rows = np.zeros((estimated_size, len(features)), dtype=np.float32)
    labels = np.zeros(estimated_size, dtype=np.int32)
    groups = []
    
    # Pre-extract features as numpy array for faster access
    df_features = df[features].values.astype(np.float32)
    df_is_known = df["is_known"].values
    df_bucket = df["bucket_pen"].values
    
    # Build neg index by bucket using numpy for speed
    neg_by_bucket = {}
    for b in np.unique(df_bucket): # type: ignore
        mask = (df_bucket == b) & (df_is_known == 0)
        neg_by_bucket[int(b)] = np.where(mask)[0]
    
    row_idx = 0
    for pi in pos_indices:
        b = int(df_bucket[pi])
        pool = neg_by_bucket.get(b)
        if pool is None or len(pool) == 0:
            continue
        
        k = min(neg_per_pos, len(pool))
        chosen = rng.choice(pool, size=k, replace=False)
        
        # Add positive
        rows[row_idx] = df_features[pi]
        labels[row_idx] = 1
        row_idx += 1
        
        # Add negatives
        for ni in chosen:
            rows[row_idx] = df_features[ni]
            labels[row_idx] = 0
            row_idx += 1
        
        groups.append(k + 1)
    
    # Trim to actual size
    rows = rows[:row_idx]
    labels = labels[:row_idx]
    
    return rows, labels, np.array(groups)


def train_lgbm(train_df: pd.DataFrame, rng, features: list, neg_per_pos: int = 8) -> lgb.LGBMRanker:
    """Train LightGBM LambdaMART ranker."""
    X, y, groups = build_lgbm_groups(train_df, neg_per_pos=neg_per_pos, rng=rng, features=features)
    
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        num_leaves=63,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    ranker.fit(X, y, group=groups)
    return ranker


# ─────────────────────────────────────────────
# Model 2: XGBoost Pairwise Ranker
# ─────────────────────────────────────────────

def build_xgb_pairs(df: pd.DataFrame, neg_per_pos: int, rng, features: list) -> Tuple:
    """Build pairwise training data for XGBoost rank:pairwise."""
    X_pos, X_neg = [], []
    neg_by_bucket = {b: g.index[g["is_known"] == 0].tolist()
                     for b, g in df.groupby("bucket_pen")}
    
    for pi in df.index[df["is_known"] == 1]:
        b = int(df.at[pi, "bucket_pen"]) # type: ignore
        pool = neg_by_bucket.get(b, [])
        if not pool:
            continue
        k = min(neg_per_pos, len(pool))
        chosen = rng.choice(pool, size=k, replace=False).tolist()
        x_pos = df.loc[pi, features].values.astype(float)
        for ni in chosen:
            x_neg = df.loc[ni, features].values.astype(float)
            X_pos.append(x_pos)
            X_neg.append(x_neg)
    
    X_pos = np.array(X_pos)
    X_neg = np.array(X_neg)
    # Pairwise: diff features, label=1 (pos is better)
    X = np.vstack([X_pos - X_neg, X_neg - X_pos])
    y = np.array([1] * len(X_pos) + [0] * len(X_neg))
    return X, y


def train_xgb(train_df: pd.DataFrame, rng, features: list) -> xgb.XGBClassifier:
    """Train XGBoost pairwise ranker."""
    X, y = build_xgb_pairs(train_df, neg_per_pos=8, rng=rng, features=features)
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


def score_xgb(model: xgb.XGBClassifier, X_query: np.ndarray, X_all: np.ndarray) -> np.ndarray:
    """Score each candidate vs mean of all others (pairwise aggregation)."""
    if len(X_all) == 0:
        return np.zeros(len(X_query))
    mean_other = X_all.mean(axis=0, keepdims=True)
    diffs = X_query - mean_other
    probs = model.predict_proba(diffs)[:, 1]
    return probs


# ─────────────────────────────────────────────
# Model 3: Neural MLP Ranker
# ─────────────────────────────────────────────

class NeuralRanker:
    """Pairwise MLP ranker: learns s(x) such that s(pos) > s(neg)."""
    
    def __init__(self, hidden=(128, 64), lr=1e-3, epochs=80, batch=256, seed=42):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch = batch
        self.seed = seed
        self.scaler = StandardScaler()
        self.mlp: MLPRegressor | None = None
    
    def _build_pairs(self, df: pd.DataFrame, rng, features: list, neg_per_pos: int = 8) -> Tuple:
        X_pos, X_neg = [], []
        neg_by_bucket = {b: g.index[g["is_known"] == 0].tolist()
                         for b, g in df.groupby("bucket_pen")}
        
        for pi in df.index[df["is_known"] == 1]:
            b = int(df.at[pi, "bucket_pen"]) # type: ignore
            pool = neg_by_bucket.get(b, [])
            if not pool:
                continue
            k = min(neg_per_pos, len(pool))
            chosen = rng.choice(pool, size=k, replace=False).tolist()
            xp = df.loc[pi, features].values.astype(float)
            for ni in chosen:
                X_pos.append(xp)
                X_neg.append(df.loc[ni, features].values.astype(float))
        
        return np.array(X_pos), np.array(X_neg)
    
    def fit(self, train_df: pd.DataFrame, rng, features: list):
        X_pos, X_neg = self._build_pairs(train_df, rng, features)
        if len(X_pos) == 0:
            raise ValueError("No training pairs built.")
        
        X_all = np.vstack([X_pos, X_neg])
        X_all = self.scaler.fit_transform(X_all)
        X_pos_s = X_all[:len(X_pos)]
        X_neg_s = X_all[len(X_pos):]
        
        # Train regression: +1 for pos, -1 for neg
        X_train = np.vstack([X_pos_s, X_neg_s])
        y_train = np.concatenate([np.ones(len(X_pos_s)), -np.ones(len(X_neg_s))])
        
        self.mlp = MLPRegressor(
            hidden_layer_sizes=self.hidden,
            activation="relu",
            learning_rate_init=self.lr,
            max_iter=self.epochs,
            batch_size=self.batch,
            random_state=self.seed,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=False
        )
        self.mlp.fit(X_train, y_train)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self.scaler.transform(X)
        return self.mlp.predict(X_s) # type: ignore


# ─────────────────────────────────────────────
# Per-Bucket Evaluation
# ─────────────────────────────────────────────

def evaluate_per_bucket(df: pd.DataFrame, score_fn, model_name: str,
                        explored_ids: Set[int], eval_k: list) -> pd.DataFrame:
    """Evaluate model per explored bucket."""
    df = df.copy()
    df["score"] = score_fn(df)
    
    bucket_rows = []
    for b in sorted(explored_ids):
        sub = df[df["bucket_pen"] == b]
        if len(sub) == 0:
            continue
        
        labels = sub["is_known"].values
        scores = sub["score"].values
        n_known = int(labels.sum())
        
        row = {
            "model": model_name,
            "bucket": b,
            "n_candidates": len(sub),
            "n_known": n_known,
        }
        
        for k in eval_k:
            row[f"recall@{k}"] = recall_at_k(labels, scores, k)
            row[f"ndcg@{k}"] = ndcg_at_k(labels, scores, k)
        
        bucket_rows.append(row)
    
    return pd.DataFrame(bucket_rows)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True, choices=["breast", "prostate"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    cancer_dir = OUTPUTS / args.cancer
    meta_path = cancer_dir / "bucket_policy_meta.json"
    
    if not meta_path.exists():
        raise FileNotFoundError(f"Run 02_bucket_policy.py first. Missing: {meta_path}")
    
    meta = json.loads(meta_path.read_text())
    rng = np.random.default_rng(args.seed)
    
    print(f"\n{'='*70}")
    print(f"  MULTI-MODEL BUCKET RANKER — {args.cancer.upper()} CANCER")
    print(f"{'='*70}\n")
    
    # ── Load & split ──────────────────────────────────────────
    train_df, test_df, explored_ids = load_and_split(cancer_dir, meta, seed=args.seed)
    
    print(f"  Explored buckets    : {sorted(explored_ids)}")
    print(f"  Train pairs         : {len(train_df):,}  (positives: {train_df['is_known'].sum()})")
    print(f"  Test pairs          : {len(test_df):,}   (positives: {test_df['is_known'].sum()})")
    print()
    
    # For evaluation: use TEST SET ONLY (not train+test)
    # This gives meaningful Recall@K metrics
    df_eval = test_df.copy()
    
    all_bucket_metrics = []
    all_rankings = {}
    
    # ── Model 1: LightGBM ─────────────────────────────────────
    print("  [1/3] Training LightGBM LambdaMART ...")
    
    # Adjust neg_per_pos based on dataset size to avoid memory issues
    n_pos_train = int(train_df["is_known"].sum())
    if len(train_df) > 1_000_000:
        neg_per_pos = 4  # Reduce for very large datasets
        print(f"        (Using neg_per_pos=4 for large dataset)")
    else:
        neg_per_pos = 8
    
    lgbm_model = train_lgbm(train_df, rng, FEATURES, neg_per_pos=neg_per_pos)
    print("        ✓ Done")
    
    lgbm_metrics = evaluate_per_bucket(
        df_eval,
        lambda df: lgbm_model.predict(df[FEATURES]),
        "LightGBM-LambdaMART",
        explored_ids,
        EVAL_K
    )
    all_bucket_metrics.append(lgbm_metrics)
    
    # Save rankings (use full test set for output)
    df_ranked = df_eval.copy()
    df_ranked["score"] = lgbm_model.predict(df_eval[FEATURES])
    df_ranked["rank_in_bucket"] = (
        df_ranked.groupby("bucket_pen")["score"]
        .rank(ascending=False, method="first").astype(int)
    )
    all_rankings["lgbm"] = df_ranked
    
    print(f"        Mean Recall@10: {lgbm_metrics['recall@10'].mean():.4f}  |  "
          f"Mean NDCG@10: {lgbm_metrics['ndcg@10'].mean():.4f}")
    
    # ── Model 2: XGBoost ──────────────────────────────────────
    print("  [2/3] Training XGBoost Pairwise Ranker ...", end=" ")
    xgb_model = train_xgb(train_df, rng, FEATURES)
    print("✓")
    
    X_all = train_df[FEATURES].values  # Use training set for XGBoost reference
    xgb_metrics = evaluate_per_bucket(
        df_eval,
        lambda df: score_xgb(xgb_model, df[FEATURES].values, X_all),
        "XGBoost-Pairwise",
        explored_ids,
        EVAL_K
    )
    all_bucket_metrics.append(xgb_metrics)
    
    df_ranked = df_eval.copy()
    df_ranked["score"] = score_xgb(xgb_model, df_eval[FEATURES].values, X_all)
    df_ranked["rank_in_bucket"] = (
        df_ranked.groupby("bucket_pen")["score"]
        .rank(ascending=False, method="first").astype(int)
    )
    all_rankings["xgb"] = df_ranked
    
    print(f"        Mean Recall@10: {xgb_metrics['recall@10'].mean():.4f}  |  "
          f"Mean NDCG@10: {xgb_metrics['ndcg@10'].mean():.4f}")
    
    # ── Model 3: Neural MLP ───────────────────────────────────
    print("  [3/3] Training Neural MLP Ranker ...", end=" ")
    neural_model = NeuralRanker(hidden=(128, 64), lr=1e-3, epochs=80, seed=args.seed)
    neural_model.fit(train_df, rng, FEATURES)
    print("✓")
    
    neural_metrics = evaluate_per_bucket(
        df_eval,
        lambda df: neural_model.predict(df[FEATURES].values),
        "Neural-MLP",
        explored_ids,
        EVAL_K
    )
    all_bucket_metrics.append(neural_metrics)
    
    df_ranked = df_eval.copy()
    df_ranked["score"] = neural_model.predict(df_eval[FEATURES].values)
    df_ranked["rank_in_bucket"] = (
        df_ranked.groupby("bucket_pen")["score"]
        .rank(ascending=False, method="first").astype(int)
    )
    all_rankings["neural"] = df_ranked
    
    print(f"        Mean Recall@10: {neural_metrics['recall@10'].mean():.4f}  |  "
          f"Mean NDCG@10: {neural_metrics['ndcg@10'].mean():.4f}")
    
    # ── Save outputs ──────────────────────────────────────────
    print()
    
    # Combined comparison
    comparison_df = pd.concat(all_bucket_metrics, ignore_index=True)
    comparison_df.to_csv(cancer_dir / "explored_model_comparison.csv", index=False)
    
    # Per-model rankings
    for name, ranked_df in all_rankings.items():
        out_cols = [c for c in ["gene_u", "gene_v", "bucket_pen", "pen_diff",
                                 "is_known", "score", "rank_in_bucket"]
                    if c in ranked_df.columns]
        ranked_df[out_cols].to_csv(cancer_dir / f"explored_rankings_{name}.csv", index=False)
    
    # Summary: best model per bucket
    summary_rows = []
    for b in sorted(explored_ids):
        b_data = comparison_df[comparison_df["bucket"] == b]
        if len(b_data) == 0:
            continue
        best_row = b_data.loc[b_data["recall@10"].idxmax()]
        summary_rows.append({
            "bucket": b,
            "n_candidates": int(best_row["n_candidates"]),
            "n_known": int(best_row["n_known"]),
            "best_model": best_row["model"],
            "best_recall@10": round(best_row["recall@10"], 4),
            "best_ndcg@10": round(best_row["ndcg@10"], 4),
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(cancer_dir / "explored_bucket_summary.csv", index=False)
    
    # Save models
    model_dir = MODELS / args.cancer
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(lgbm_model, model_dir / "lgbm_ranker_multibucket.joblib")
    joblib.dump(xgb_model, model_dir / "xgb_ranker_multibucket.joblib")
    joblib.dump(neural_model, model_dir / "neural_ranker_multibucket.joblib")
    
    # ── Print summary ─────────────────────────────────────────
    print("  ── Per-Bucket Best Model Summary ────────────────────────")
    print(summary_df.to_string(index=False))
    
    print(f"\n Saved outputs:")
    print(f"     • explored_model_comparison.csv   (per-bucket metrics)")
    print(f"     • explored_bucket_summary.csv     (best model per bucket)")
    print(f"     • explored_rankings_*.csv         (3 files with scores)")
    print(f"     • Models saved to {model_dir}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()