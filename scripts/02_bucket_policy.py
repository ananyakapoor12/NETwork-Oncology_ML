from pathlib import Path
import json
import numpy as np
import pandas as pd
from math import floor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"

CANCERS = ["breast", "prostate"]
BUCKETS = 5
TOP_FRACS = [0.01, 0.10, 0.20, 0.50]

# weights for combining histogram signals (tune later)
ALPHA_PEN = 0.5
BETA_DIST = 0.3
GAMMA_PPR = 0.2

def make_equal_width_bins(x: pd.Series, n_bins: int):
    xmin, xmax = float(x.min()), float(x.max())
    edges = np.linspace(xmin, xmax, n_bins + 1)
    # ensure last edge includes max
    edges[-1] = np.nextafter(edges[-1], edges[-1] + 1)
    return edges

def delta_histogram_like_panacea(df: pd.DataFrame, score_col: str, n_bins: int):
    """
    PANACEA-style coverage:
    - equal-width buckets
    - within each bucket, sort candidates by score desc
    - compute coverage of known targets within top 1/10/20/50% of bucket
    """
    edges = make_equal_width_bins(df[score_col], n_bins)
    bucket_id = pd.cut(df[score_col], bins=edges, labels=False, include_lowest=True)

    df2 = df.copy()
    df2["bucket"] = bucket_id

    bucket_rows = []
    total_known = int(df2["is_known"].sum())

    for b in range(n_bins):
        bucket_df = df2[df2["bucket"] == b]
        candidates = bucket_df[score_col].sort_values(ascending=False).reset_index(drop=True)
        known_scores = bucket_df[bucket_df["is_known"] == 1][score_col].sort_values(ascending=False).reset_index(drop=True)

        cand_n = len(candidates)
        known_n = len(known_scores)

        row = {
            "bucket": b,
            "range": f"{edges[b]:.6g} - {edges[b+1]:.6g}",
            "candidate_count": cand_n,
            "known_count": known_n,
            "percentage_of_all_known_in_bucket": (known_n / total_known) if total_known > 0 else 0.0,
        }

        for m in TOP_FRACS:
            if cand_n == 0 or known_n == 0:
                cov = 0.0
            else:
                idx = floor(cand_n * m)
                idx = min(idx, cand_n - 1)
                thr = candidates.iloc[idx]
                cov = float((known_scores >= thr).sum() / known_n)
            row[f"coverage_{int(m*100)}"] = cov

        bucket_rows.append(row)

    return pd.DataFrame(bucket_rows), edges

def build_bucket_policy(bkt_pen: pd.DataFrame, bkt_dist: pd.DataFrame, bkt_ppr: pd.DataFrame, lam: float):
    """
    W(b) = lam*novelty + (1-lam)*exploitation
    novelty = 1 if known_count==0 else 0
    exploitation = weighted sum of coverage_50 across 3 histograms
    """
    merged = bkt_pen[["bucket","range","candidate_count","known_count","coverage_50"]].merge(
        bkt_dist[["bucket","coverage_50"]], on="bucket", suffixes=("_pen","_dist")
    ).merge(
        bkt_ppr[["bucket","coverage_50"]], on="bucket"
    )
    merged = merged.rename(columns={"coverage_50": "coverage_50_ppr"})

    merged["novelty"] = (merged["known_count"] == 0).astype(int)
    merged["exploitation"] = (
        ALPHA_PEN * merged["coverage_50_pen"] +
        BETA_DIST * merged["coverage_50_dist"] +
        GAMMA_PPR * merged["coverage_50_ppr"]
    )
    merged["W"] = lam * merged["novelty"] + (1 - lam) * merged["exploitation"]

    s = merged["W"].sum()
    merged["W_norm"] = merged["W"] / s if s > 0 else 0.0
    return merged

def main():
    for cancer in CANCERS:
        in_path = OUTPUTS / cancer / "pairs_k2_standardized.csv"
        df = pd.read_csv(in_path)

        # Compute PANACEA-style bucket stats for each score
        bkt_pen, pen_edges = delta_histogram_like_panacea(df, "pen_diff", BUCKETS)
        bkt_dist, dist_edges = delta_histogram_like_panacea(df, "dist_diff", BUCKETS)
        bkt_ppr, ppr_edges = delta_histogram_like_panacea(df, "ppr_diff", BUCKETS)

        out_dir = OUTPUTS / cancer
        bkt_pen.to_csv(out_dir / "bucket_table_pen.csv", index=False)
        bkt_dist.to_csv(out_dir / "bucket_table_dist.csv", index=False)
        bkt_ppr.to_csv(out_dir / "bucket_table_ppr.csv", index=False)

        # Novelty emphasis: prostate typically has more "empty-bar" buckets; breast tends to be low-evidence rather than empty
        lam = 0.7 if cancer == "prostate" else 0.6

        policy = build_bucket_policy(bkt_pen, bkt_dist, bkt_ppr, lam=lam)
        policy.to_csv(out_dir / "bucket_policy.csv", index=False)

        meta = {
            "cancer": cancer,
            "bucket_no": BUCKETS,
            "top_fracs": TOP_FRACS,
            "lambda": lam,
            "weights": {"alpha_pen": ALPHA_PEN, "beta_dist": BETA_DIST, "gamma_ppr": GAMMA_PPR},
            "edges": {"pen": pen_edges.tolist(), "dist": dist_edges.tolist(), "ppr": ppr_edges.tolist()},
        }
        (out_dir / "bucket_policy_meta.json").write_text(json.dumps(meta, indent=2))

        print(f"[OK] {cancer}: wrote bucket tables + policy to {out_dir}")

if __name__ == "__main__":
    main()
