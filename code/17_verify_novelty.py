"""Confirm top unexplored candidates are not already known drug-target pairs."""
from pathlib import Path
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"

def verify_novelty(cancer="prostate"):
    """Check that top candidates aren't already known targets."""
    
    cancer_dir = OUTPUTS / cancer
    
    # Load all pairs with is_known labels
    all_pairs = pd.read_csv(cancer_dir / "pairs_k2_standardized.csv")
    
    # Load your top candidates
    top_candidates = pd.read_csv(cancer_dir / "unexplored_top_candidates.csv")
    
    print(f"\n{'='*70}")
    print(f"  NOVELTY VERIFICATION - {cancer.upper()}")
    print(f"{'='*70}\n")
    
    # Create gene pair identifiers for matching
    all_pairs['pair_id'] = all_pairs.apply(
        lambda r: tuple(sorted([r['gene_u'], r['gene_v']])), axis=1
    )
    
    top_candidates['pair_id'] = top_candidates.apply(
        lambda r: tuple(sorted([r['gene_u'], r['gene_v']])), axis=1
    )
    
    # Merge to check is_known status
    verified = top_candidates.merge(
        all_pairs[['pair_id', 'is_known']], 
        on='pair_id', 
        how='left'
    )
    
    # Count how many are actually known
    n_known = (verified['is_known'] == 1).sum()
    n_unknown = (verified['is_known'] == 0).sum()
    n_missing = verified['is_known'].isna().sum()
    
    print(f"  Total top candidates: {len(top_candidates)}")
    print(f"  Already known (is_known=1): {n_known} {'PROBLEM!' if n_known > 0 else 'GOOD'}")
    print(f"  Truly novel (is_known=0): {n_unknown} GOOD")
    print(f"  Missing in dataset: {n_missing}")
    
    # Show any problematic cases
    if n_known > 0:
        print(f"\n WARNING: Found {n_known} candidates that are already known targets!")
        print(f"\n  Known targets in top candidates:")
        known_cases = verified[verified['is_known'] == 1]
        display_cols = [c for c in ['gene_u', 'gene_v', 'composite_score', 'confidence', 'global_rank'] 
                        if c in known_cases.columns]
        print(known_cases[display_cols].to_string(index=False))
        
        print(f"\nThese should NOT be in your top novel discoveries!")
        print(f"     They were used as TRAINING DATA (is_known=1)")
    else:
        print(f"\n All top candidates are truly novel (is_known=0)")
        print(f"     No training data leakage detected!")
    
    # Additional check: bucket verification
    print(f"\n  Bucket Distribution of Top Candidates:")
    bucket_dist = verified.groupby('bucket_pen').size()
    for bucket_id, count in bucket_dist.items():
        print(f"     Bucket {int(bucket_id)}: {count} candidates")
    
    # ══════════════════════════════════════════════════════════════════
    # FIX: Need to assign buckets to all_pairs to check unexplored status
    # ══════════════════════════════════════════════════════════════════
    
    # Load bucket metadata to get edges
    meta_path = cancer_dir / "bucket_policy_meta.json"
    
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        edges_pen = np.array(meta["edges"]["pen"])
        
        # Assign bucket_pen to all_pairs
        all_pairs['bucket_pen'] = pd.cut(
            all_pairs['pen_diff'],
            bins=edges_pen, # type: ignore
            labels=False,
            include_lowest=True
        ) # type: ignore
        
        # Now check which buckets are unexplored
        known_per_bucket = all_pairs.groupby('bucket_pen')['is_known'].sum()
        unexplored = known_per_bucket[known_per_bucket == 0].index.tolist()
        
        print(f"\n  Unexplored buckets (is_known=0 for entire bucket): {unexplored}")
        print(f"  Are all top candidates from unexplored buckets? ", end="")
        
        all_from_unexplored = verified['bucket_pen'].isin(unexplored).all()
        if all_from_unexplored:
            print("YES")
        else:
            non_unexplored = verified[~verified['bucket_pen'].isin(unexplored)]
            print(f"NO - {len(non_unexplored)} from explored buckets!")
            print(f"\n  Candidates from EXPLORED buckets (should be 0):")
            display_cols = [c for c in ['gene_u', 'gene_v', 'bucket_pen', 'is_known'] 
                            if c in non_unexplored.columns]
            print(non_unexplored[display_cols].head(10).to_string(index=False))
    else:
        print(f"\n Skipping bucket verification - bucket_policy_meta.json not found")
        print(f"     Run scripts/02_bucket_policy.py first to enable bucket checks")
    
    print(f"\n{'='*70}\n")
    
    return verified


if __name__ == "__main__":
    import argparse
    
    # Allow command-line argument for cancer type
    ap = argparse.ArgumentParser(description="Verify novelty of top candidates")
    ap.add_argument("--cancer", default="prostate", choices=["breast", "prostate"],
                    help="Cancer type to verify")
    args = ap.parse_args()
    
    # Run verification
    verified = verify_novelty(args.cancer)
    
    # Save verification report
    output_path = OUTPUTS / args.cancer / "novelty_verification.csv"
    verified.to_csv(output_path, index=False)
    
    print(f"Saved verification report to: {output_path}")