"""Deduplicate K=3 triplets (canonical ordering) and rank by aggregated pairwise score."""
import pandas as pd # type: ignore
from pathlib import Path

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True)
    args = ap.parse_args()

    path = Path(f"outputs/{args.cancer}/k3_candidates.csv")
    df = pd.read_csv(path)

    # canonical ordering to avoid permutations
    df[["g1","g2","g3"]] = df.apply(
        lambda r: sorted([r.g1, r.g2, r.g3]),
        axis=1, result_type="expand"
    )

    df = (
        df.drop_duplicates(subset=["g1","g2","g3"])
          .sort_values("score", ascending=False)
          .reset_index(drop=True)
    )

    out = path.with_name("k3_ranked_final.csv")
    df.to_csv(out, index=False)

    print(f"[OK] Saved {len(df):,} ranked triples → {out}")

if __name__ == "__main__":
    main()
