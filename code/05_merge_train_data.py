"""Concatenate standard and hard-negative training sets into train_rank_final.csv."""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True)
    args = ap.parse_args()

    out_dir = OUTPUTS / args.cancer

    base = pd.read_csv(out_dir / "train_rank.csv")
    hard = pd.read_csv(out_dir / "train_rank_hn.csv")

    # offset group ids to avoid collision
    hard["group_id"] += base["group_id"].max() + 1

    merged = pd.concat([base, hard], ignore_index=True)
    merged.to_csv(out_dir / "train_rank_final.csv", index=False)

    print(f"[OK] {args.cancer}: merged rows={len(merged):,}")

if __name__ == "__main__":
    main()
