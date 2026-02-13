import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
MODELS = PROJECT_ROOT / "models"

FEATURE_COLS = [
    "pen_diff", "dist_diff", "ppr_diff",
    "bucket_pen", "bucket_dist", "bucket_ppr",
    "pos_in_bucket_pen"
]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True, choices=["breast","prostate"])
    ap.add_argument("--neg_per_pos", type=int, default=8)
    #ap.add_argument("--hard_frac", type=float, default=0.5)
    ap.add_argument("--hard_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    cancer_dir = OUTPUTS / args.cancer
    model_path = MODELS / args.cancer / "lgbm_ranker.joblib"

    df = pd.read_csv(cancer_dir / "pairs_k2_standardized.csv")
    meta = json.loads((cancer_dir / "bucket_policy_meta.json").read_text())

    # assign all buckets (must match training features)
    edges_pen  = np.array(meta["edges"]["pen"])
    edges_dist = np.array(meta["edges"]["dist"])
    edges_ppr  = np.array(meta["edges"]["ppr"])

    df["bucket_pen"]  = pd.cut(df["pen_diff"],  bins=edges_pen,  labels=False, include_lowest=True)
    df["bucket_dist"] = pd.cut(df["dist_diff"], bins=edges_dist, labels=False, include_lowest=True)
    df["bucket_ppr"]  = pd.cut(df["ppr_diff"],  bins=edges_ppr,  labels=False, include_lowest=True)

    # position inside PEN bucket (same as Phase 3)
    grp = df.groupby("bucket_pen")["pen_diff"]
    df["pos_in_bucket_pen"] = (
        df["pen_diff"] - grp.transform("min")
    ) / (grp.transform("max") - grp.transform("min") + 1e-12)

    """
    # assign bucket_pen
    edges = np.array(meta["edges"]["pen"])
    df["bucket_pen"] = pd.cut(df["pen_diff"], bins=edges, labels=False, include_lowest=True)

    grp = df.groupby("bucket_pen")["pen_diff"]
    df["pos_in_bucket_pen"] = (df["pen_diff"] - grp.transform("min")) / (grp.transform("max") - grp.transform("min") + 1e-12)

    """

    model = joblib.load(model_path)

    # score all pairs (vectorized, fast)
    df["pred"] = model.predict(df[FEATURE_COLS])

    positives = df[df.is_known == 1].index.to_numpy()
    negatives = df[df.is_known == 0]

    neg_by_bucket = {
        b: g.sort_values("pred", ascending=False)
        for b,g in negatives.groupby("bucket_pen")
    }

    rows = []
    group_id = 0

    for pi in positives:
        b = df.at[pi, "bucket_pen"]
        pool = neg_by_bucket.get(b)
        if pool is None or len(pool) == 0:
            continue

        k = args.neg_per_pos
        k_hard = int(k * args.hard_frac)
        k_rand = k - k_hard

        hard = pool.head(max(k_hard,1)).sample(k_hard, replace=len(pool)<k_hard, random_state=args.seed)
        rand = pool.sample(k_rand, replace=len(pool)<k_rand, random_state=args.seed)

        group = pd.concat([df.loc[[pi]], hard, rand])

        for _,r in group.iterrows():
            rows.append({
                "group_id": group_id,
                "label": int(r.is_known),
                **{c: float(r[c]) for c in FEATURE_COLS}
            })
        group_id += 1

    out = pd.DataFrame(rows)
    out.to_csv(cancer_dir / "train_rank_hn.csv", index=False)
    print(f"[OK] {args.cancer}: hard-neg train groups={out.group_id.nunique():,}")

if __name__ == "__main__":
    main()
