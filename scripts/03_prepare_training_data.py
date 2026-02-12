import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"

# ---------- CONFIG DEFAULTS ----------
DEFAULT_NEG_PER_POS = 8
DEFAULT_TEST_FRAC = 0.2
DEFAULT_SEED = 42

FEATURE_COLS = [
    "pen_diff", "dist_diff", "ppr_diff",
    "bucket_pen", "bucket_dist", "bucket_ppr",
    "pos_in_bucket_pen"
]

def assign_bucket(values: pd.Series, edges: list[float]) -> pd.Series:
    # include_lowest=True and right=False to mimic typical binning (consistent once fixed)
    bins = np.array(edges, dtype=float)
    # pandas cut returns NaN if outside; edges from meta should cover min/max
    return pd.cut(values, bins=bins, labels=False, include_lowest=True)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True, choices=["breast", "prostate"])
    ap.add_argument("--neg_per_pos", type=int, default=DEFAULT_NEG_PER_POS)
    ap.add_argument("--test_frac", type=float, default=DEFAULT_TEST_FRAC)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    cancer_dir = OUTPUTS / args.cancer
    in_csv = cancer_dir / "pairs_k2_standardized.csv"
    meta_path = cancer_dir / "bucket_policy_meta.json"

    if not in_csv.exists():
        raise FileNotFoundError(f"Missing {in_csv}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path} (run Phase 2 first)")

    meta = json.loads(meta_path.read_text())
    edges_pen = meta["edges"]["pen"]
    edges_dist = meta["edges"]["dist"]
    edges_ppr = meta["edges"]["ppr"]

    # Load only what we need (avoid gene strings for memory)
    usecols = ["pen_diff", "dist_diff", "ppr_diff", "is_known"]
    df = pd.read_csv(in_csv, usecols=usecols)

    # Buckets from Phase 2 edges
    df["bucket_pen"] = assign_bucket(df["pen_diff"], edges_pen).astype("Int64")
    df["bucket_dist"] = assign_bucket(df["dist_diff"], edges_dist).astype("Int64")
    df["bucket_ppr"] = assign_bucket(df["ppr_diff"], edges_ppr).astype("Int64")

    # Drop any rows that fail binning (should be rare)
    df = df.dropna(subset=["bucket_pen", "bucket_dist", "bucket_ppr"]).copy()
    df["bucket_pen"] = df["bucket_pen"].astype(int)
    df["bucket_dist"] = df["bucket_dist"].astype(int)
    df["bucket_ppr"] = df["bucket_ppr"].astype(int)

    # Position-in-bucket (fast approx): normalize pen_diff within bucket by min/max
    grp = df.groupby("bucket_pen")["pen_diff"]
    bmin = grp.transform("min")
    bmax = grp.transform("max")
    df["pos_in_bucket_pen"] = (df["pen_diff"] - bmin) / (bmax - bmin + 1e-12)

    # Split positives into train/test sets
    pos_idx = df.index[df["is_known"] == 1].to_numpy()
    if len(pos_idx) == 0:
        raise ValueError("No positives found (is_known==1). Check Phase 1 output.")

    rng.shuffle(pos_idx)
    n_test = int(len(pos_idx) * args.test_frac)
    test_pos_idx = pos_idx[:n_test]
    train_pos_idx = pos_idx[n_test:]

    # Pre-index negatives per bucket for fast sampling
    neg_idx = df.index[df["is_known"] == 0].to_numpy()
    # bucket -> list of indices
    neg_by_bucket = {}
    for b in range(int(df["bucket_pen"].max()) + 1):
        bucket_neg = df.index[(df["is_known"] == 0) & (df["bucket_pen"] == b)].to_numpy()
        neg_by_bucket[b] = bucket_neg

    def build_groups(pos_indices: np.ndarray, out_path: Path):
        rows = []
        group_id = 0

        for pi in pos_indices:
            b = int(df.at[pi, "bucket_pen"])
            neg_pool = neg_by_bucket.get(b, np.array([], dtype=int))
            if len(neg_pool) == 0:
                # If no negatives in that bucket, skip (rare)
                continue

            # sample negatives (with replacement if pool too small)
            k = args.neg_per_pos
            replace = len(neg_pool) < k
            chosen_negs = rng.choice(neg_pool, size=k, replace=replace)

            # add positive then negatives
            group_rows = [pi] + chosen_negs.tolist()
            for idx in group_rows:
                r = df.loc[idx, FEATURE_COLS + ["is_known"]]
                rows.append({
                    "group_id": group_id,
                    "label": int(r["is_known"]),
                    **{c: float(r[c]) for c in FEATURE_COLS}
                })
            group_id += 1

        out_df = pd.DataFrame(rows)
        out_df.to_csv(out_path, index=False)
        print(f"[OK] wrote {out_path} (groups={out_df['group_id'].nunique():,}, rows={len(out_df):,})")

    build_groups(train_pos_idx, cancer_dir / "train_rank.csv")
    build_groups(test_pos_idx,  cancer_dir / "test_rank.csv")

if __name__ == "__main__":
    main()
