"""Generate K=3 gene triplets from pairwise ranked candidates."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
MODELS = PROJECT_ROOT / "models"

FEATURES = [
    "pen_diff", "dist_diff", "ppr_diff",
    "bucket_pen", "bucket_dist", "bucket_ppr",
    "pos_in_bucket_pen"
]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True, choices=["breast","prostate"])
    ap.add_argument("--top_pairs", type=int, default=5000)
    ap.add_argument("--third_per_pair", type=int, default=15)
    args = ap.parse_args()

    cancer_dir = OUTPUTS / args.cancer
    model = joblib.load(MODELS / args.cancer / "lgbm_ranker.joblib")

    df = pd.read_csv(cancer_dir / "pairs_k2_standardized.csv")
    policy = pd.read_csv(cancer_dir / "bucket_policy.csv")
    meta = json.loads((cancer_dir / "bucket_policy_meta.json").read_text())

    # buckets
    df["bucket_pen"] = pd.cut(df["pen_diff"], meta["edges"]["pen"], labels=False, include_lowest=True)
    df["bucket_dist"] = pd.cut(df["dist_diff"], meta["edges"]["dist"], labels=False, include_lowest=True)
    df["bucket_ppr"] = pd.cut(df["ppr_diff"], meta["edges"]["ppr"], labels=False, include_lowest=True)

    grp = df.groupby("bucket_pen")["pen_diff"]
    df["pos_in_bucket_pen"] = (
        df["pen_diff"] - grp.transform("min")
    ) / (grp.transform("max") - grp.transform("min") + 1e-12)

    df["pair_score"] = model.predict(df[FEATURES])

    triples = []

    for _, row in policy.iterrows():
        b = int(row.bucket)
        w = row.W_norm
        k = max(1, int(w * args.top_pairs))

        top_pairs = df[df.bucket_pen == b].nlargest(k, "pair_score")
        genes = list(set(top_pairs.gene_u) | set(top_pairs.gene_v))

        for _, p in top_pairs.iterrows():
            candidates = np.random.choice(
                genes, size=min(args.third_per_pair, len(genes)), replace=False
            )
            for g in candidates:
                if g in (p.gene_u, p.gene_v):
                    continue
                triples.append({
                    "g1": p.gene_u,
                    "g2": p.gene_v,
                    "g3": g,
                    "score": p.pair_score
                })

    out = pd.DataFrame(triples).sort_values("score", ascending=False)
    out.to_csv(cancer_dir / "k3_candidates.csv", index=False)
    print(f"[OK] {args.cancer}: generated {len(out):,} k=3 candidates")

if __name__ == "__main__":
    main()
