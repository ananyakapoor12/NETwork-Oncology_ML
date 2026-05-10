"""Assign PANACEA's published PEN/PPR/dist histogram bucket boundaries to all gene pairs."""
from pathlib import Path
import json
import numpy as np # type: ignore
import pandas as pd # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"

CANCERS = ["breast", "prostate"]

# ═══════════════════════════════════════════════════════════════════════════
# PANACEA'S PUBLISHED BUCKET BOUNDARIES (ALL THREE SCORES)
# Extracted directly from PANACEA histogram output images
# ═══════════════════════════════════════════════════════════════════════════

PANACEA_BUCKET_EDGES = {
    "breast": {
        # From Breast_Cancer_oncogenes_PEN-diff_percentage_plot_des.png
        "pen": [-0.0045, 0.4301, 0.8647, 1.2993, 1.7338, 2.1684],
        
        # From Breast_Cancer_oncogenes_Distance-diff_percentage_plot_des.png
        "dist": [-2.9549, -1.5958, -0.2368, 1.1222, 2.4812, 3.8403],
        
        # From Breast_Cancer_oncogenes_ppr-diff_percentage_plot_des.png
        "ppr": [-0.0137, -0.0109, -0.0081, -0.0053, -0.0024, 0.0004],
    },
    "prostate": {
        # From Prostate_Cancer_oncogenes_PEN-diff_percentage_plot_des.png
        "pen": [-0.5126, -0.1847, 0.1432, 0.4712, 0.7991, 1.1271],
        
        # From Prostate_Cancer_oncogenes_Distance-diff_percentage_plot_des.png
        "dist": [-2.7135, -1.5697, -0.4259, 0.7179, 1.8617, 3.0055],
        
        # From Prostate_Cancer_oncogenes_ppr-diff_percentage_plot_des.png
        "ppr": [-0.0233, -0.0183, -0.0137, -0.0044, -0.0009, 0.0003]
    }
}

N_BUCKETS = 5


def compute_bucket_statistics(df: pd.DataFrame, score_col: str, bucket_col: str, 
                               edges: list, n_buckets: int = 5):
    """
    Compute PANACEA-style statistics for each bucket.
    
    Args:
        df: DataFrame with gene pairs
        score_col: Name of score column (e.g., 'pen_diff')
        bucket_col: Name of bucket column (e.g., 'bucket_pen')
        edges: Bucket edges from PANACEA
        n_buckets: Number of buckets (always 5 for PANACEA)
    
    Returns:
        DataFrame with bucket statistics
    """
    bucket_rows = []
    total_known = int(df["is_known"].sum())
    
    for b in range(n_buckets):
        bucket_df = df[df[bucket_col] == b]
        
        n_pairs = len(bucket_df)
        n_known = int(bucket_df["is_known"].sum())
        
        edge_min = edges[b]
        edge_max = edges[b + 1]
        
        # Compute coverage at different thresholds (PANACEA style)
        candidates = bucket_df[score_col].sort_values(ascending=False)
        known_scores = bucket_df[bucket_df["is_known"] == 1][score_col].sort_values(ascending=False)
        
        coverages = {}
        for pct in [1, 10, 20, 50]:
            if len(candidates) == 0 or len(known_scores) == 0:
                coverages[f"coverage_{pct}"] = 0.0
            else:
                idx = max(0, int(len(candidates) * pct / 100) - 1)
                threshold = candidates.iloc[idx]
                coverage = float((known_scores >= threshold).sum() / len(known_scores))
                coverages[f"coverage_{pct}"] = coverage
        
        bucket_rows.append({
            "bucket": b,
            "range": f"[{edge_min:.4f}, {edge_max:.4f}]",
            "candidate_count": n_pairs,
            "known_count": n_known,
            "percentage_of_all_known": f"{100 * n_known / total_known:.1f}%" if total_known > 0 else "0%",
            **coverages
        })
    
    return pd.DataFrame(bucket_rows)


def main():
    """
    Create bucket policy using PANACEA's exact histogram boundaries.
    """
    
    for cancer in CANCERS:
        print(f"\n{'='*70}")
        print(f"  PANACEA BUCKET POLICY - {cancer.upper()}")
        print(f"  Using PANACEA's Published Histogram Boundaries")
        print(f"{'='*70}\n")
        
        cancer_dir = OUTPUTS / cancer
        cancer_dir.mkdir(parents=True, exist_ok=True)
        
        # Load standardized pairs
        pairs_path = cancer_dir / "pairs_k2_standardized.csv"
        if not pairs_path.exists():
            raise FileNotFoundError(f"Missing: {pairs_path}")
        
        df = pd.read_csv(pairs_path)
        print(f"  Loaded {len(df):,} gene pairs")
        
        # Get PANACEA's edges for this cancer
        edges = PANACEA_BUCKET_EDGES[cancer]
        
        print(f"\n  PANACEA Bucket Edges (from histogram output):")
        print(f"    PEN-diff:      {edges['pen']}")
        print(f"    Distance-diff: {edges['dist']}")
        print(f"    PPR-diff:      {edges['ppr']}\n")
        
        # ══════════════════════════════════════════════════════════════
        # Assign buckets using PANACEA's EXACT edges
        # ══════════════════════════════════════════════════════════════
        
        for score_name, edge_key in [("pen_diff", "pen"), 
                                       ("dist_diff", "dist"), 
                                       ("ppr_diff", "ppr")]:
            
            bucket_edges = np.array(edges[edge_key], dtype=float)
            
            # Use pd.cut with PANACEA's exact boundaries
            df[f"bucket_{edge_key}"] = pd.cut(
                df[score_name],
                bins=bucket_edges, # type: ignore
                labels=False,
                include_lowest=True,
                duplicates='drop'  # Handle any duplicate edges
            ) # type: ignore
        
        # ══════════════════════════════════════════════════════════════
        # Drop NaN (values outside PANACEA's range)
        # ══════════════════════════════════════════════════════════════
        
        n_before = len(df)
        df = df.dropna(subset=["bucket_pen", "bucket_dist", "bucket_ppr"]).copy()
        n_dropped = n_before - len(df)
        
        if n_dropped > 0:
            pct = 100 * n_dropped / n_before
            print(f" {n_dropped:,} pairs ({pct:.2f}%) fall outside PANACEA's bucket ranges")
            print(f"      Keeping {len(df):,} pairs within PANACEA's defined ranges\n")
        
        # Convert to int
        df["bucket_pen"] = df["bucket_pen"].astype(int)
        df["bucket_dist"] = df["bucket_dist"].astype(int)
        df["bucket_ppr"] = df["bucket_ppr"].astype(int)
        
        # ══════════════════════════════════════════════════════════════
        # Compute position within PEN bucket
        # ══════════════════════════════════════════════════════════════
        
        grp = df.groupby("bucket_pen")["pen_diff"]
        df["pos_in_bucket_pen"] = (
            (df["pen_diff"] - grp.transform("min"))
            / (grp.transform("max") - grp.transform("min") + 1e-12)
        )
        
        # Save pairs with bucket assignments
        df.to_csv(cancer_dir / "pairs_with_buckets.csv", index=False)
        
        # ══════════════════════════════════════════════════════════════
        # Analyze PEN-diff bucket statistics (primary focus)
        # ══════════════════════════════════════════════════════════════
        
        print(f"  PEN-diff Bucket Statistics:")
        print(f"  {'─'*70}")
        
        bucket_summary = []
        
        for bucket_id in range(N_BUCKETS):
            bucket_df = df[df["bucket_pen"] == bucket_id]
            n_pairs = len(bucket_df)
            n_known = int(bucket_df["is_known"].sum())
            
            pen_min = edges["pen"][bucket_id]
            pen_max = edges["pen"][bucket_id + 1]
            
            status = "EXPLORED" if n_known > 0 else "UNEXPLORED"
            
            bucket_summary.append({
                "bucket": bucket_id,
                "pen_range": f"[{pen_min:.4f}, {pen_max:.4f}]",
                "n_pairs": n_pairs,
                "n_known": n_known,
                "percentage_known": f"{100 * n_known / n_pairs:.1f}%" if n_pairs > 0 else "0%",
                "status": status,
            })
            
            print(f"  Bucket {bucket_id}: {pen_min:8.4f} to {pen_max:8.4f}  "
                  f"| {n_pairs:8,} pairs | {n_known:5} known | {status}")
        
        # Identify explored vs unexplored
        explored = [b["bucket"] for b in bucket_summary if b["n_known"] > 0]
        unexplored = [b["bucket"] for b in bucket_summary if b["n_known"] == 0]
        
        print(f"\n  Explored buckets   : {explored}")
        print(f"  Unexplored buckets : {unexplored}")
        
        # ══════════════════════════════════════════════════════════════
        # Compute detailed statistics for all three scores
        # ══════════════════════════════════════════════════════════════
        
        bkt_pen_stats = compute_bucket_statistics(df, "pen_diff", "bucket_pen", edges["pen"])
        bkt_dist_stats = compute_bucket_statistics(df, "dist_diff", "bucket_dist", edges["dist"])
        bkt_ppr_stats = compute_bucket_statistics(df, "ppr_diff", "bucket_ppr", edges["ppr"])
        
        # Save detailed bucket tables
        bkt_pen_stats.to_csv(cancer_dir / "bucket_table_pen.csv", index=False)
        bkt_dist_stats.to_csv(cancer_dir / "bucket_table_dist.csv", index=False)
        bkt_ppr_stats.to_csv(cancer_dir / "bucket_table_ppr.csv", index=False)
        
        # ══════════════════════════════════════════════════════════════
        # Save bucket policy (simple version focused on PEN-diff)
        # ══════════════════════════════════════════════════════════════
        
        bucket_policy_df = pd.DataFrame(bucket_summary)
        bucket_policy_df.to_csv(cancer_dir / "bucket_policy.csv", index=False)
        
        # ══════════════════════════════════════════════════════════════
        # Save metadata
        # ══════════════════════════════════════════════════════════════
        
        meta = {
            "source": "PANACEA published histogram output",
            "method": "Exact boundaries from PANACEA (NOT computed from data)",
            "histogram_files": {
                "pen": f"{cancer.capitalize()}_Cancer_oncogenes_PEN-diff_percentage_plot_des.png",
                "dist": f"{cancer.capitalize()}_Cancer_oncogenes_Distance-diff_percentage_plot_des.png",
                "ppr": f"{cancer.capitalize()}_Cancer_oncogenes_ppr-diff_percentage_plot_des.png"
            },
            "n_buckets": N_BUCKETS,
            "edges": edges,
            "explored_buckets": explored,
            "unexplored_buckets": unexplored,
            "bucket_summary": bucket_summary,
            "note": "ALL bucket boundaries (PEN, Distance, PPR) come from PANACEA's analysis, not computed by us"
        }
        
        with open(cancer_dir / "bucket_policy_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        print(f"\n  ✅ Saved:")
        print(f"     • pairs_with_buckets.csv       ({len(df):,} pairs)")
        print(f"     • bucket_policy.csv")
        print(f"     • bucket_table_pen.csv")
        print(f"     • bucket_table_dist.csv")
        print(f"     • bucket_table_ppr.csv")
        print(f"     • bucket_policy_meta.json")
        print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()