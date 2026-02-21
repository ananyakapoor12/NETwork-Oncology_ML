from __future__ import annotations
import argparse
import json
from pathlib import Path
import warnings
from typing import Dict, Set

import numpy as np
import pandas as pd
import joblib

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


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def assign_buckets(df: pd.DataFrame, meta: Dict) -> pd.DataFrame:
    """Assign bucket indices and pos_in_bucket_pen using meta edges."""
    df = df.copy()
    
    # Assign buckets - fixed to avoid type warnings
    for score, key in [("pen_diff", "pen"), ("dist_diff", "dist"), ("ppr_diff", "ppr")]:
        edges = np.array(meta["edges"][key], dtype=float)
        # Use pd.cut with explicit typing to avoid Pylance warnings
        bucket_series = pd.cut(df[score], bins=edges, labels=False, include_lowest=True) # type: ignore
        df[f"bucket_{key}"] = bucket_series
    
    # Position within PEN bucket
    grp = df.groupby("bucket_pen")["pen_diff"]
    df["pos_in_bucket_pen"] = (
        (df["pen_diff"] - grp.transform("min")) 
        / (grp.transform("max") - grp.transform("min") + 1e-12)
    )
    
    return df.dropna(subset=["bucket_pen", "bucket_dist", "bucket_ppr"])


def load_best_model(cancer: str):
    """Load the best available trained ranker (prefer LightGBM)."""
    model_dir = MODELS / cancer
    
    # Try in order of preference
    for fname in ["lgbm_ranker.joblib", "lgbm_ranker_v2.joblib"]:
        p = model_dir / fname
        if p.exists():
            print(f"  ✓ Using model: {p.name}")
            return joblib.load(p), "LightGBM"
    
    raise FileNotFoundError(
        f"No trained model found in {model_dir}.\n"
        f"Run scripts/04_train_ranker.py first."
    )


def compute_novelty_weight(df: pd.DataFrame, unexplored_ids: Set[int], meta: Dict) -> Dict[int, float]:
    """
    Novelty weight for each unexplored bucket based on:
      - Mean PEN-diff (higher = more oncogene-specific)
      - Bucket size (more candidates = more reliable)
    """
    weights: Dict[int, float] = {}
    for b in unexplored_ids:
        sub = df[df["bucket_pen"] == b]
        if len(sub) == 0:
            weights[b] = 0.0
            continue
        
        mean_pen = float(sub["pen_diff"].mean())
        n_cands  = len(sub)
        
        # Bonus for larger buckets (more candidates to explore)
        size_bonus = np.log1p(n_cands) / 10.0
        
        # Positive PEN-diff is better (oncogene-specific influence)
        pen_bonus = 1.5 if mean_pen > 0 else 1.0
        
        weights[b] = mean_pen * pen_bonus * (1.0 + size_bonus)
    
    # Normalize to sum=1
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    
    return weights


def assign_confidence_tier(composite_score: float, q33: float, q67: float,
                            pen_diff: float) -> str:
    """
    Assign High/Medium/Low confidence based on composite score percentile.
    Also requires positive PEN-diff for High tier.
    """
    if composite_score >= q67 and pen_diff > 0.15:
        return "High"
    elif composite_score >= q33:
        return "Medium"
    else:
        return "Low"


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True, choices=["breast", "prostate"])
    ap.add_argument("--top_n", type=int, default=100,
                    help="Number of top candidates to output as high-priority shortlist")
    args = ap.parse_args()
    
    cancer_dir = OUTPUTS / args.cancer
    meta_path  = cancer_dir / "bucket_policy_meta.json"
    
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Run 02_bucket_policy.py first. Missing: {meta_path}"
        )
    
    meta = json.loads(meta_path.read_text())
    
    print(f"\n{'='*70}")
    print(f"  🔬 UNEXPLORED BUCKET CANDIDATE RANKER — {args.cancer.upper()} CANCER")
    print(f"{'='*70}\n")
    
    # ── Load data ──────────────────────────────────────────────
    pairs_path = cancer_dir / "pairs_k2_standardized.csv"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Missing: {pairs_path}. Run 01_prepare_data.py first.")
    
    df = pd.read_csv(pairs_path)
    df = assign_buckets(df, meta)
    df["bucket_pen"]  = df["bucket_pen"].astype(int)
    df["bucket_dist"] = df["bucket_dist"].astype(int)
    df["bucket_ppr"]  = df["bucket_ppr"].astype(int)
    
    # ── Identify explored vs unexplored ───────────────────────
    known_per_bucket = df.groupby("bucket_pen")["is_known"].sum()
    all_buckets      = set(df["bucket_pen"].unique())
    explored_ids     = set(known_per_bucket[known_per_bucket > 0].index.astype(int).tolist())
    unexplored_ids   = all_buckets - explored_ids
    
    print(f"  Total buckets         : {len(all_buckets)}")
    print(f"  Explored buckets      : {sorted(explored_ids)}  ({len(explored_ids)} buckets)")
    print(f"  Unexplored buckets    : {sorted(unexplored_ids)}  ({len(unexplored_ids)} buckets)")
    
    if len(unexplored_ids) == 0:
        print("\n  ⚠️  All buckets are explored. No novel candidates to rank.")
        print("      (This is unusual — double-check bucket_policy.csv)\n")
        return
    
    df_unexplored = df[df["bucket_pen"].isin(unexplored_ids)].copy()
    print(f"  Candidates in unexplored regions: {len(df_unexplored):,}\n")
    
    if len(df_unexplored) == 0:
        print("  ⚠️  No candidates found in unexplored buckets.\n")
        return
    
    # ── Load model ────────────────────────────────────────────
    model, model_name = load_best_model(args.cancer)
    
    # ── Score unexplored candidates ───────────────────────────
    print(f"  Scoring {len(df_unexplored):,} candidates using {model_name} ...", end=" ")
    df_unexplored["model_score"] = model.predict(df_unexplored[FEATURES])
    print("✓")
    
    # ── Compute novelty weights ───────────────────────────────
    novelty_weights = compute_novelty_weight(df_unexplored, unexplored_ids, meta)
    df_unexplored["novelty_weight"] = df_unexplored["bucket_pen"].map(novelty_weights)
    
    # ── Composite score ────────────────────────────────────────
    # Normalize model scores to [0,1]
    score_min = df_unexplored["model_score"].min()
    score_max = df_unexplored["model_score"].max()
    df_unexplored["model_score_norm"] = (
        (df_unexplored["model_score"] - score_min) 
        / (score_max - score_min + 1e-12)
    )
    
    # 70% model, 30% novelty
    df_unexplored["composite_score"] = (
        0.7 * df_unexplored["model_score_norm"] +
        0.3 * df_unexplored["novelty_weight"]
    )
    
    # ── Confidence tiers ───────────────────────────────────────
    q33 = df_unexplored["composite_score"].quantile(0.33)
    q67 = df_unexplored["composite_score"].quantile(0.67)
    
    df_unexplored["confidence"] = df_unexplored.apply(
        lambda r: assign_confidence_tier(r["composite_score"], q33, q67, r["pen_diff"]),
        axis=1
    )
    
    # ── Global and intra-bucket ranks ──────────────────────────
    df_unexplored = df_unexplored.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df_unexplored["global_rank"] = range(1, len(df_unexplored) + 1)
    
    df_unexplored["rank_in_bucket"] = (
        df_unexplored.groupby("bucket_pen")["composite_score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    
    # ── Per-bucket summary ─────────────────────────────────────
    bucket_summary_rows = []
    for b in sorted(unexplored_ids):
        sub = df_unexplored[df_unexplored["bucket_pen"] == b]
        if len(sub) == 0:
            continue
        
        # PEN-diff range from meta
        edges = meta["edges"]["pen"]
        pen_min = edges[b] if b < len(edges) else None
        pen_max = edges[b + 1] if (b + 1) < len(edges) else None
        pen_range = f"[{pen_min:.4f}, {pen_max:.4f}]" if pen_min is not None else "N/A"
        
        bucket_summary_rows.append({
            "bucket": b,
            "pen_diff_range": pen_range,
            "n_candidates": len(sub),
            "novelty_weight": round(novelty_weights.get(b, 0), 4),
            "mean_pen_diff": round(sub["pen_diff"].mean(), 4),
            "mean_composite_score": round(sub["composite_score"].mean(), 4),
            "top_composite_score": round(sub["composite_score"].max(), 4),
            "high_confidence_count": int((sub["confidence"] == "High").sum()),
            "medium_confidence_count": int((sub["confidence"] == "Medium").sum()),
            "low_confidence_count": int((sub["confidence"] == "Low").sum()),
        })
    
    bucket_summary = pd.DataFrame(bucket_summary_rows)
    
    # ── Output ─────────────────────────────────────────────────
    output_cols = [
        "gene_u", "gene_v", "bucket_pen", 
        "pen_diff", "dist_diff", "ppr_diff",
        "model_score", "composite_score", "novelty_weight",
        "confidence", "global_rank", "rank_in_bucket"
    ]
    avail_cols = [c for c in output_cols if c in df_unexplored.columns]
    
    # Full ranked list
    df_unexplored[avail_cols].to_csv(
        cancer_dir / "unexplored_ranked_all.csv", index=False
    )
    
    # Top N shortlist
    top_n = df_unexplored[df_unexplored["confidence"] == "High"].head(args.top_n)
    if len(top_n) < args.top_n:
        # Fill with Medium if not enough High
        needed = args.top_n - len(top_n)
        medium = df_unexplored[df_unexplored["confidence"] == "Medium"].head(needed)
        top_n = pd.concat([top_n, medium])
    
    top_n[avail_cols].to_csv(
        cancer_dir / "unexplored_top_candidates.csv", index=False
    )
    
    # Bucket summary
    bucket_summary.to_csv(
        cancer_dir / "unexplored_by_bucket.csv", index=False
    )
    
    # ── Print summary ──────────────────────────────────────────
    print("\n  ── Unexplored Bucket Summary ───────────────────────────────────")
    print(bucket_summary.to_string(index=False))
    
    print(f"\n  ── Confidence Distribution ─────────────────────────────────────")
    tier_counts = df_unexplored["confidence"].value_counts()
    for tier in ["High", "Medium", "Low"]:
        count = tier_counts.get(tier, 0)
        pct   = 100 * count / len(df_unexplored) if len(df_unexplored) > 0 else 0
        print(f"     {tier:7s} : {count:6,}  ({pct:5.1f}%)")
    
    print(f"\n  ── Top-10 High-Confidence Novel Candidates ─────────────────────")
    display_cols = [c for c in ["gene_u", "gene_v", "bucket_pen", "pen_diff",
                                 "composite_score", "confidence", "global_rank"]
                    if c in df_unexplored.columns]
    print(df_unexplored[display_cols].head(10).to_string(index=False))
    
    print(f"\n  ✅ Saved outputs:")
    print(f"     • unexplored_ranked_all.csv       ({len(df_unexplored):,} candidates)")
    print(f"     • unexplored_top_candidates.csv   (top {len(top_n)} high-confidence)")
    print(f"     • unexplored_by_bucket.csv        ({len(bucket_summary)} buckets)")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()