"""Compare LightGBM against heuristic baselines: random, degree centrality, and single-score rankings."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"

N_BUCKETS   = 5
NEG_PER_POS = 8
TEST_FRAC   = 0.2
SEED        = 42

FULL_FEATURES = [
    "pen_diff", "dist_diff", "ppr_diff",
    "bucket_pen", "bucket_dist", "bucket_ppr",
    "pos_in_bucket_pen",
]


# ════════════════════════════════════════════════════════════════════════════
# METRIC HELPERS
# ════════════════════════════════════════════════════════════════════════════

def recall_at_k(df_scored: pd.DataFrame, k: int) -> float:
    hits = 0
    n_groups = df_scored["group_id"].nunique()
    for _, g in df_scored.groupby("group_id"):
        if (g.nlargest(k, "pred")["label"] == 1).any():
            hits += 1
    return hits / n_groups if n_groups > 0 else 0.0


def ndcg_at_k(df_scored: pd.DataFrame, k: int) -> float:
    """Normalised Discounted Cumulative Gain at K."""
    scores = []
    for _, g in df_scored.groupby("group_id"):
        ranked = g.nlargest(k, "pred")["label"].values
        dcg  = sum(r / np.log2(i + 2) for i, r in enumerate(ranked))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(int(g["label"].sum()), k)))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def evaluate(df_scored: pd.DataFrame, name: str) -> dict:
    r1   = recall_at_k(df_scored, 1)
    r5   = recall_at_k(df_scored, 5)
    r10  = recall_at_k(df_scored, 10)
    n10  = ndcg_at_k(df_scored,  10)
    print(f"    {name:<42}  R@1={r1:.4f}  R@5={r5:.4f}  R@10={r10:.4f}  NDCG@10={n10:.4f}")
    return {"Baseline": name,
            "Recall@1": round(r1, 4),
            "Recall@5": round(r5, 4),
            "Recall@10": round(r10, 4),
            "NDCG@10": round(n10, 4)}


# ════════════════════════════════════════════════════════════════════════════
# REBUILD TEST SET
# ════════════════════════════════════════════════════════════════════════════

def build_test_set(cancer_dir: Path) -> pd.DataFrame:
    """
    Rebuild the held-out test ranking groups from pairs_with_buckets.csv,
    using the same SEED and split fraction as the training pipeline.

    Returns a DataFrame with columns:
        group_id, label, gene_u, gene_v,
        pen_diff, dist_diff, ppr_diff,
        bucket_pen, bucket_dist, bucket_ppr,
        pos_in_bucket_pen
    """
    pairs_path = cancer_dir / "pairs_with_buckets.csv"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Missing: {pairs_path}")

    df = pd.read_csv(pairs_path)

    rng = np.random.default_rng(SEED)
    pos_idx = df.index[df["is_known"] == 1].to_numpy().copy()
    rng.shuffle(pos_idx)
    n_test   = int(len(pos_idx) * TEST_FRAC)
    test_pos = pos_idx[:n_test]

    # Negatives per bucket (same ratio as training split)
    neg_by_bucket = {
        b: df.index[(df["is_known"] == 0) & (df["bucket_pen"] == b)].to_numpy().copy()
        for b in range(N_BUCKETS)
    }

    rows = []
    for gid, pi in enumerate(test_pos):
        b = int(df.at[pi, "bucket_pen"])
        neg_pool = neg_by_bucket.get(b, np.array([], dtype=int))
        if len(neg_pool) == 0:
            continue
        replace = len(neg_pool) < NEG_PER_POS
        negs = rng.choice(neg_pool, size=NEG_PER_POS, replace=replace)
        for idx in [pi] + negs.tolist():
            r = df.loc[idx]
            rows.append({
                "group_id": gid,
                "label": int(r["is_known"]),
                "gene_u": r.get("gene_u", ""),
                "gene_v": r.get("gene_v", ""),
                "pen_diff":          float(r["pen_diff"]),
                "dist_diff":         float(r["dist_diff"]),
                "ppr_diff":          float(r["ppr_diff"]),
                "bucket_pen":        int(r["bucket_pen"]),
                "bucket_dist":       int(r["bucket_dist"]),
                "bucket_ppr":        int(r["bucket_ppr"]),
                "pos_in_bucket_pen": float(r["pos_in_bucket_pen"]),
            })

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# BASELINE 0 — Full LightGBM model  (reference line)
# ════════════════════════════════════════════════════════════════════════════

def baseline_lgbm(cancer_dir: Path, test_df: pd.DataFrame) -> dict:
    train_path = cancer_dir / "train_rank_final.csv"
    if not train_path.exists():
        train_path = cancer_dir / "train_rank.csv"

    train_df = pd.read_csv(train_path)
    train_df = train_df.sort_values("group_id").reset_index(drop=True)
    groups   = train_df.groupby("group_id").size().to_numpy()

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

    scored = test_df.copy()
    scored["pred"] = model.predict(test_df[FULL_FEATURES])
    return evaluate(scored, "LightGBM LambdaMART (full model)")


# ════════════════════════════════════════════════════════════════════════════
# BASELINE 1 — Random Ranking
# ════════════════════════════════════════════════════════════════════════════

def baseline_random(test_df: pd.DataFrame) -> dict:
    rng    = np.random.default_rng(SEED + 1)   # different seed from split
    scored = test_df.copy()
    scored["pred"] = rng.random(len(scored))
    return evaluate(scored, "Random Ranking")


# ════════════════════════════════════════════════════════════════════════════
# BASELINE 2 — Degree Centrality
# ════════════════════════════════════════════════════════════════════════════

def baseline_degree_centrality(cancer_dir: Path, test_df: pd.DataFrame) -> dict:
    """
    Rank pairs by the sum of their two nodes' degrees in the full PPI graph
    (all pairs in pairs_with_buckets.csv, not just positives).
    """
    pairs_path = cancer_dir / "pairs_with_buckets.csv"
    all_pairs  = pd.read_csv(pairs_path, usecols=["gene_u", "gene_v"])

    # Count how many times each gene appears across all pairs
    degree = {}
    for _, row in all_pairs.iterrows():
        degree[row["gene_u"]] = degree.get(row["gene_u"], 0) + 1
        degree[row["gene_v"]] = degree.get(row["gene_v"], 0) + 1

    scored = test_df.copy()
    if "gene_u" in scored.columns and scored["gene_u"].ne("").all():
        scored["pred"] = (
            scored["gene_u"].map(degree).fillna(0)
            + scored["gene_v"].map(degree).fillna(0)
        )
    else:
        # Fallback: no gene names in test set — use pen_diff proxy
        # (should not happen if pairs_with_buckets.csv has gene_u/gene_v)
        print("    [WARNING] gene names not available — using pen_diff as degree proxy")
        scored["pred"] = scored["pen_diff"]

    return evaluate(scored, "Degree Centrality")


# ════════════════════════════════════════════════════════════════════════════
# BASELINE 3–5 — Single-Score Ranking (no model, direct score)
# ════════════════════════════════════════════════════════════════════════════

def baseline_single_score(test_df: pd.DataFrame, score_col: str, label: str) -> dict:
    scored = test_df.copy()
    scored["pred"] = scored[score_col]
    return evaluate(scored, label)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True, choices=["breast", "prostate"])
    args = ap.parse_args()
    cancer = args.cancer

    print(f"\n{'='*70}")
    print(f"  BASELINE COMPARISON — {cancer.upper()}")
    print(f"  Evaluating heuristic and single-score baselines vs LightGBM")
    print(f"{'='*70}\n")

    cancer_dir = OUTPUTS / cancer

    print("  Building test set...")
    test_df = build_test_set(cancer_dir)
    n_groups = test_df["group_id"].nunique()
    print(f"  Test groups: {n_groups:,}  |  Test rows: {len(test_df):,}\n")

    print(f"  {'Baseline':<44}  R@1      R@5      R@10     NDCG@10")
    print(f"  {'─'*44}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")

    results = []

    print("\n  [0] LightGBM reference model")
    results.append(baseline_lgbm(cancer_dir, test_df))

    # ── Baseline 1: Random ──
    print("\n  [1] Random Ranking")
    results.append(baseline_random(test_df))

    # ── Baseline 2: Degree Centrality ──
    print("\n  [2] Degree Centrality")
    results.append(baseline_degree_centrality(cancer_dir, test_df))

    # ── Baseline 3–5: Single Scores ──
    print("\n  [3–5] Single-Score Baselines")
    results.append(baseline_single_score(test_df, "pen_diff",  "Single-Score: PEN-diff only"))
    results.append(baseline_single_score(test_df, "dist_diff", "Single-Score: dist-diff only"))
    results.append(baseline_single_score(test_df, "ppr_diff",  "Single-Score: PPR-diff only"))

    # ── Save ──
    results_df = pd.DataFrame(results)
    out_path = cancer_dir / "baseline_comparison.csv"
    results_df.to_csv(out_path, index=False)

    print(f"\n{'='*70}")
    print(f"  SUMMARY — {cancer.upper()}")
    print(f"{'='*70}\n")
    print(results_df.to_string(index=False))
    print(f"\n  Saved to: {out_path}\n")


if __name__ == "__main__":
    main()
