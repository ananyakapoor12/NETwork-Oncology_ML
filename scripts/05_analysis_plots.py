#!/usr/bin/env python3
"""
scripts/05_analysis_plots.py

Works with your FYP_PANACEA outputs schema:
Columns typically include:
['group_id','label','pen_diff','dist_diff','ppr_diff',
 'bucket_pen','bucket_dist','bucket_ppr','pos_in_bucket_pen']

Generates:
1) Bucket distribution plots (combined + split train/test if possible)
2) Rank position histogram (pos_in_bucket_pen)
3) Cross-cancer Recall@K using pos_in_bucket_pen (lower is better)

Run:
  python scripts/05_analysis_plots.py
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUTPUTS_DIR = Path("outputs")
FIGURES_DIR = Path("figures")

# If filename contains these, assign cancer type.
CANCER_KEYWORDS = {
    "breast": ["breast", "brca"],
    "prostate": ["prostate", "prad"],
}

# prefer these columns from your schema
PREFERRED_BUCKET_COLS = ["bucket_pen", "bucket_dist", "bucket_ppr", "bucket"]
PREFERRED_RANK_COLS = ["pos_in_bucket_pen", "rank", "rank_position", "position"]
PREFERRED_LABEL_COLS = ["label", "y", "target"]

# "split" detection by filename (optional)
SPLIT_KEYWORDS = {
    "train": ["train"],
    "test": ["test"],
    "k3": ["k3"],
    "other": []
}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def infer_cancer_type(path: Path) -> str:
    fn = str(path).lower()
    for cancer, keys in CANCER_KEYWORDS.items():
        if any(k in fn for k in keys):
            return cancer
    # fallback: if folder name is breast/prostate
    parent = path.parent.name.lower()
    if parent in CANCER_KEYWORDS:
        return parent
    return "unknown"


def infer_split(path: Path) -> str:
    fn = path.name.lower()
    for split, keys in SPLIT_KEYWORDS.items():
        if split == "other":
            continue
        if any(k in fn for k in keys):
            return split
    return "other"


def pick_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def find_ranked_csvs(outputs_dir: Path) -> List[Path]:
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")

    csvs = sorted(outputs_dir.rglob("*.csv"))

    # keep only likely ranking outputs
    ranked_keywords = ("rank", "ranked", "train_rank", "test_rank", "k3_ranked")
    csvs = [p for p in csvs if any(k in p.name.lower() for k in ranked_keywords)]
    return csvs


def load_all_ranked(outputs_dir: Path) -> Dict[str, List[Tuple[str, Path, pd.DataFrame]]]:
    """
    Returns:
      cancer -> list of (split, path, df)
    split is inferred from filename: train/test/k3/other
    """
    csvs = find_ranked_csvs(outputs_dir)
    if not csvs:
        raise FileNotFoundError("No ranked CSVs found under outputs/")

    by_cancer: Dict[str, List[Tuple[str, Path, pd.DataFrame]]] = {}

    for p in csvs:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[SKIP] Could not read {p}: {e}")
            continue

        cancer = infer_cancer_type(p)
        split = infer_split(p)

        # choose columns
        bucket_col = pick_first_existing_col(df, PREFERRED_BUCKET_COLS)
        rank_col = pick_first_existing_col(df, PREFERRED_RANK_COLS)
        label_col = pick_first_existing_col(df, PREFERRED_LABEL_COLS)

        # must have at least rank or bucket to be useful
        if bucket_col is None and rank_col is None:
            print(f"[SKIP] {p.name} missing both bucket and rank columns.")
            continue

        # normalize: create standard columns if possible
        if bucket_col is not None and bucket_col != "bucket":
            df = df.rename(columns={bucket_col: "bucket"})
        if rank_col is not None and rank_col != "rank_pos":
            df = df.rename(columns={rank_col: "rank_pos"})
        if label_col is not None and label_col != "label":
            df = df.rename(columns={label_col: "label"})

        # create a score for histogram plots if we have rank_pos
        if "rank_pos" in df.columns and "score" not in df.columns:
            # lower rank_pos is better -> higher score should be better
            df["score"] = -pd.to_numeric(df["rank_pos"], errors="coerce")

        by_cancer.setdefault(cancer, []).append((split, p, df))

    return by_cancer


# -----------------------
# Plot helpers
# -----------------------

def plot_bucket_distribution_combined(by_cancer, outpath: Path) -> None:
    """
    Combined bucket counts per cancer across ALL files.
    """
    bucket_counts = {}
    all_buckets = set()

    for cancer, items in by_cancer.items():
        series_list = []
        for _, _, df in items:
            if "bucket" not in df.columns:
                continue
            c = df["bucket"].value_counts(dropna=False)
            series_list.append(c)
            all_buckets.update(c.index.tolist())
        if series_list:
            bucket_counts[cancer] = pd.concat(series_list, axis=1).fillna(0).sum(axis=1)

    if not bucket_counts:
        print("[WARN] No 'bucket' column found; skipping bucket distribution.")
        return

    all_buckets_sorted = sorted(list(all_buckets), key=lambda x: str(x))
    plot_df = pd.DataFrame(bucket_counts).reindex(all_buckets_sorted).fillna(0)

    plt.figure()
    x = np.arange(len(plot_df.index))
    cancers = list(plot_df.columns)
    width = 0.8 / max(1, len(cancers))

    for i, cancer in enumerate(cancers):
        plt.bar(x + i * width, plot_df[cancer].values, width=width, label=cancer)

    plt.xlabel("Bucket (bucket_pen)")
    plt.ylabel("Count")
    plt.title("Bucket Distribution of Ranked Candidates (All Outputs)")
    plt.xticks(x + (len(cancers) - 1) * width / 2, [str(b) for b in plot_df.index], rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print("[OK] Saved:", outpath)


def plot_bucket_distribution_by_split(by_cancer, split_name: str, outpath: Path) -> None:
    """
    Bucket counts per cancer for a specific split (train/test/k3).
    """
    bucket_counts = {}
    all_buckets = set()

    for cancer, items in by_cancer.items():
        series_list = []
        for split, _, df in items:
            if split != split_name:
                continue
            if "bucket" not in df.columns:
                continue
            c = df["bucket"].value_counts(dropna=False)
            series_list.append(c)
            all_buckets.update(c.index.tolist())
        if series_list:
            bucket_counts[cancer] = pd.concat(series_list, axis=1).fillna(0).sum(axis=1)

    if not bucket_counts:
        print(f"[WARN] No bucket data for split={split_name}; skipping.")
        return

    all_buckets_sorted = sorted(list(all_buckets), key=lambda x: str(x))
    plot_df = pd.DataFrame(bucket_counts).reindex(all_buckets_sorted).fillna(0)

    plt.figure()
    x = np.arange(len(plot_df.index))
    cancers = list(plot_df.columns)
    width = 0.8 / max(1, len(cancers))

    for i, cancer in enumerate(cancers):
        plt.bar(x + i * width, plot_df[cancer].values, width=width, label=cancer)

    plt.xlabel("Bucket (bucket_pen)")
    plt.ylabel("Count")
    plt.title(f"Bucket Distribution (split={split_name})")
    plt.xticks(x + (len(cancers) - 1) * width / 2, [str(b) for b in plot_df.index], rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print("[OK] Saved:", outpath)


def plot_rank_position_hist(by_cancer, outpath: Path) -> None:
    """
    Histogram of rank positions (pos_in_bucket_pen) across cancers.
    Uses rank_pos if available.
    """
    plt.figure()
    plotted_any = False

    for cancer, items in by_cancer.items():
        vals = []
        for _, _, df in items:
            if "rank_pos" not in df.columns:
                continue
            v = pd.to_numeric(df["rank_pos"], errors="coerce").dropna().values
            if len(v) > 0:
                vals.append(v)
        if vals:
            allv = np.concatenate(vals)
            plt.hist(allv, bins=50, alpha=0.5, label=cancer)
            plotted_any = True

    if not plotted_any:
        print("[WARN] No rank_pos found; skipping rank position histogram.")
        plt.close()
        return

    plt.xlabel("Rank Position within bucket (lower = better)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Intra-bucket Rank Positions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print("[OK] Saved:", outpath)


def plot_recall_at_k(by_cancer, ks: List[int], outpath: Path) -> None:
    """
    Recall@K per cancer using rank_pos (lower is better).
    Recall@K = (# positives in top-K) / (total positives)
    Computed on merged dataframe per cancer (across all files that contain labels).
    """
    recalls = {}

    for cancer, items in by_cancer.items():
        frames = []
        for _, _, df in items:
            if "label" in df.columns and "rank_pos" in df.columns:
                frames.append(df[["label", "rank_pos"]].copy())
        if not frames:
            continue

        merged = pd.concat(frames, ignore_index=True)
        merged["label"] = pd.to_numeric(merged["label"], errors="coerce").fillna(0).astype(int)
        merged["rank_pos"] = pd.to_numeric(merged["rank_pos"], errors="coerce")
        merged = merged.dropna(subset=["rank_pos"])

        total_pos = int(merged["label"].sum())
        if total_pos == 0:
            recalls[cancer] = [np.nan for _ in ks]
            continue

        merged = merged.sort_values("rank_pos", ascending=True)  # best first
        recs = []
        for k in ks:
            topk = merged.head(k)
            found = int(topk["label"].sum())
            recs.append(found / total_pos)
        recalls[cancer] = recs

    if not recalls:
        print("[WARN] No label+rank_pos data; skipping Recall@K.")
        return

    plt.figure()
    x = np.arange(len(ks))
    cancers = list(recalls.keys())
    width = 0.8 / max(1, len(cancers))

    for i, cancer in enumerate(cancers):
        plt.bar(x + i * width, recalls[cancer], width=width, label=cancer)

    plt.xlabel("K")
    plt.ylabel("Recall@K")
    plt.title("Cross-Cancer Recall@K (using intra-bucket rank positions)")
    plt.xticks(x + (len(cancers) - 1) * width / 2, [str(k) for k in ks])
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print("[OK] Saved:", outpath)


def main() -> None:
    ensure_dir(FIGURES_DIR)

    by_cancer = load_all_ranked(OUTPUTS_DIR)

    print("\nDetected ranked CSVs:")
    for cancer, items in by_cancer.items():
        for split, p, _ in items:
            print(f" - {cancer:8s} | {split:5s} | {p}")

    # 1) bucket distribution plots
    plot_bucket_distribution_combined(by_cancer, FIGURES_DIR / "fig_3_2_bucket_distribution.png")
    plot_bucket_distribution_by_split(by_cancer, "train", FIGURES_DIR / "fig_3_2a_bucket_distribution_train.png")
    plot_bucket_distribution_by_split(by_cancer, "test",  FIGURES_DIR / "fig_3_2b_bucket_distribution_test.png")

    # 2) rank position histogram
    plot_rank_position_hist(by_cancer, FIGURES_DIR / "fig_3_3_rank_positions.png")

    # 3) recall@K plot
    plot_recall_at_k(by_cancer, ks=[10, 20, 50, 100], outpath=FIGURES_DIR / "fig_4_1_recall_comparison.png")

    print("\nDone. Figures saved to:", FIGURES_DIR.resolve())


if __name__ == "__main__":
    main()
