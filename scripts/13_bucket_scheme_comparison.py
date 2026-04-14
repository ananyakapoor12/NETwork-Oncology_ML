"""
Three experiments:
  1. Bucket Quality Metrics  — compare PEN / PPR / dist bucket distributions from existing data
  2. Ranker Performance      — retrain LightGBM with PPR-based grouping vs PEN-based grouping
  3. Novel Discovery Impact  — show how the unexplored region changes under each scheme

"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"

FULL_FEATURES_PEN = [
    "pen_diff", "dist_diff", "ppr_diff",
    "bucket_pen", "bucket_dist", "bucket_ppr",
    "pos_in_bucket_pen"
]

FULL_FEATURES_PPR = [
    "pen_diff", "dist_diff", "ppr_diff",
    "bucket_pen", "bucket_dist", "bucket_ppr",
    "pos_in_bucket_ppr"   # swapped
]

N_BUCKETS = 5
NEG_PER_POS = 8
TEST_FRAC = 0.2
SEED = 42


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def recall_at_k(df_scored: pd.DataFrame, k: int) -> float:
    hits = 0
    for _, g in df_scored.groupby("group_id"):
        if (g.nlargest(k, "pred")["label"] == 1).any():
            hits += 1
    return hits / df_scored["group_id"].nunique()


def train_and_eval(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   features: list, label: str) -> dict:
    print(f"  [{label}] training...", end="", flush=True)

    train_df = train_df.sort_values("group_id").reset_index(drop=True)
    test_df  = test_df.sort_values("group_id").reset_index(drop=True)

    groups = train_df.groupby("group_id").size().to_numpy()

    model = lgb.LGBMRanker(
        objective="lambdarank",
        num_leaves=63,
        learning_rate=0.05,
        n_estimators=300,
        random_state=SEED,
        verbosity=-1,
        force_col_wise=True,
    )
    model.fit(train_df[features], train_df["label"].astype(int), group=groups)

    preds = model.predict(test_df[features])
    scored = test_df.copy()
    scored["pred"] = preds

    r1  = recall_at_k(scored, 1)
    r10 = recall_at_k(scored, 10)
    print(f"  Recall@1={r1:.4f}  Recall@10={r10:.4f}")
    return {"label": label, "Recall@1": r1, "Recall@10": r10}


def build_training_data(df: pd.DataFrame, bucket_col: str,
                        pos_in_bucket_col: str, rng: np.random.Generator):
    """
    Build train/test ranking groups using the specified primary bucket column.
    Negatives are sampled from the SAME bucket as each positive (bucket-aware sampling).
    """
    # Compute pos_in_bucket for whichever axis is primary
    grp = df.groupby(bucket_col)[bucket_col.replace("bucket_", "") + "_diff"]
    # handle column name mapping
    score_col = {"bucket_pen": "pen_diff", "bucket_ppr": "ppr_diff",
                 "bucket_dist": "dist_diff"}[bucket_col]
    grp = df.groupby(bucket_col)[score_col]
    bmin = grp.transform("min")
    bmax = grp.transform("max")
    df = df.copy()
    df[pos_in_bucket_col] = (df[score_col] - bmin) / (bmax - bmin + 1e-12)

    # Build neg index per bucket
    neg_by_bucket = {
        b: df.index[(df["is_known"] == 0) & (df[bucket_col] == b)].to_numpy().copy()
        for b in range(N_BUCKETS)
    }

    pos_idx = df.index[df["is_known"] == 1].to_numpy().copy()
    rng.shuffle(pos_idx)
    n_test   = int(len(pos_idx) * TEST_FRAC)
    test_pos = pos_idx[:n_test]
    train_pos = pos_idx[n_test:]

    feature_cols = FULL_FEATURES_PPR if pos_in_bucket_col == "pos_in_bucket_ppr" \
                   else FULL_FEATURES_PEN

    def _build(pos_indices):
        rows = []
        for gid, pi in enumerate(pos_indices):
            b = int(df.at[pi, bucket_col])
            neg_pool = neg_by_bucket.get(b, np.array([], dtype=int))
            if len(neg_pool) == 0:
                continue
            replace = len(neg_pool) < NEG_PER_POS
            negs = rng.choice(neg_pool, size=NEG_PER_POS, replace=replace)
            for idx in [pi] + negs.tolist():
                r = df.loc[idx]
                rows.append({
                    "group_id": gid,
                    "label": int(r["is_known"]),
                    **{c: float(r[c]) for c in feature_cols}
                })
        return pd.DataFrame(rows)

    return _build(train_pos), _build(test_pos)


# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Bucket Quality Metrics
# ════════════════════════════════════════════════════════════════════════════

def exp1_bucket_quality(cancer: str, df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'─'*60}")
    print(f"  EXPERIMENT 1: Bucket Quality Metrics ({cancer.upper()})")
    print(f"{'─'*60}")

    scheme_rows = []
    for scheme, bucket_col in [("PEN-diff", "bucket_pen"),
                                ("PPR-diff", "bucket_ppr"),
                                ("dist-diff", "bucket_dist")]:
        counts = []
        known_counts = []
        for b in range(N_BUCKETS):
            sub = df[df[bucket_col] == b]
            counts.append(len(sub))
            known_counts.append(int(sub["is_known"].sum()))

        total_known = sum(known_counts)
        max_bucket  = max(counts)
        n_unexplored = sum(1 for k in known_counts if k == 0)

        # Known-target spread: higher entropy = better spread across buckets
        probs = np.array(known_counts, dtype=float)
        probs = probs / (probs.sum() + 1e-12)
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        max_entropy = np.log(N_BUCKETS)
        spread_pct = 100 * entropy / max_entropy

        dominant_bucket_pct = 100 * max(known_counts) / (total_known + 1e-12)

        print(f"\n  {scheme}:")
        print(f"    Bucket sizes        : {counts}")
        print(f"    Known per bucket    : {known_counts}")
        print(f"    Worst-case size     : {max_bucket:,}")
        print(f"    Unexplored buckets  : {n_unexplored}")
        print(f"    Known in top bucket : {dominant_bucket_pct:.1f}%")
        print(f"    Known-target spread : {spread_pct:.1f}%  (100%=uniform)")

        scheme_rows.append({
            "Scheme": scheme,
            "Bucket sizes": str(counts),
            "Worst-case bucket": max_bucket,
            "Unexplored buckets": n_unexplored,
            "% known in largest bucket": f"{dominant_bucket_pct:.1f}%",
            "Known-target spread (entropy %)": f"{spread_pct:.1f}%",
        })

    return pd.DataFrame(scheme_rows)


# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Ranker Performance Under Each Bucketing Scheme
# ════════════════════════════════════════════════════════════════════════════

def exp2_ranker_comparison(cancer: str, df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'─'*60}")
    print(f"  EXPERIMENT 2: Ranker Performance ({cancer.upper()})")
    print(f"  (bucket-aware sampling — negatives from same primary bucket)")
    print(f"{'─'*60}\n")

    rng = np.random.default_rng(SEED)
    results = []

    # ── PEN-based bucketing (current system) ──
    train_pen, test_pen = build_training_data(
        df, bucket_col="bucket_pen",
        pos_in_bucket_col="pos_in_bucket_pen", rng=rng
    )
    r = train_and_eval(train_pen, test_pen, FULL_FEATURES_PEN,
                       "PEN-bucketed training (current)")
    results.append(r)

    # ── PPR-based bucketing (professor's question) ──
    rng2 = np.random.default_rng(SEED)
    train_ppr, test_ppr = build_training_data(
        df, bucket_col="bucket_ppr",
        pos_in_bucket_col="pos_in_bucket_ppr", rng=rng2
    )
    r = train_and_eval(train_ppr, test_ppr, FULL_FEATURES_PPR,
                       "PPR-bucketed training (professor's question)")
    results.append(r)

    # ── PPR features for ranking, but still PEN-bucketed groups ──
    # (diagnostic: does swapping just the pos_in_bucket feature matter?)
    rng3 = np.random.default_rng(SEED)
    train_pen2, test_pen2 = build_training_data(
        df, bucket_col="bucket_pen",
        pos_in_bucket_col="pos_in_bucket_pen", rng=rng3
    )
    # Add pos_in_bucket_ppr as extra feature
    train_pen2["pos_in_bucket_ppr"] = _add_pos_in_bucket(df, "bucket_ppr", "ppr_diff",
                                                          train_pen2)
    test_pen2["pos_in_bucket_ppr"]  = _add_pos_in_bucket(df, "bucket_ppr", "ppr_diff",
                                                          test_pen2)
    r = train_and_eval(train_pen2, test_pen2, FULL_FEATURES_PPR,
                       "PEN-bucketed groups + pos_in_bucket_ppr feature")
    results.append(r)

    return pd.DataFrame(results)


def _add_pos_in_bucket(source_df: pd.DataFrame, bucket_col: str,
                       score_col: str, target_df: pd.DataFrame) -> pd.Series:
    """Compute pos_in_bucket for score_col grouped by bucket_col, aligned to target_df index."""
    grp = source_df.groupby(bucket_col)[score_col]
    bmin = grp.transform("min")
    bmax = grp.transform("max")
    pos = (source_df[score_col] - bmin) / (bmax - bmin + 1e-12)
    # target_df may be a subset — fill missing with 0
    return target_df.index.map(lambda i: pos.get(i, 0.0))


# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — Novel Discovery Space Under Each Scheme
# ════════════════════════════════════════════════════════════════════════════

def exp3_discovery_space(cancer: str, df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'─'*60}")
    print(f"  EXPERIMENT 3: Novel Discovery Space ({cancer.upper()})")
    print(f"{'─'*60}")

    rows = []
    for scheme, bucket_col in [("PEN-diff", "bucket_pen"),
                                ("PPR-diff", "bucket_ppr"),
                                ("dist-diff", "bucket_dist")]:
        unexplored_buckets = []
        unexplored_candidates = 0
        for b in range(N_BUCKETS):
            sub = df[df[bucket_col] == b]
            if sub["is_known"].sum() == 0:
                unexplored_buckets.append(b)
                unexplored_candidates += len(sub)

        print(f"\n  {scheme}:")
        print(f"    Unexplored buckets     : {unexplored_buckets}")
        print(f"    Unexplored candidates  : {unexplored_candidates:,}")

        rows.append({
            "Scheme": scheme,
            "Unexplored bucket IDs": str(unexplored_buckets),
            "Unexplored candidates": unexplored_candidates,
        })

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True, choices=["breast", "prostate"])
    args = ap.parse_args()
    cancer = args.cancer

    print(f"\n{'='*60}")
    print(f"  BUCKET SCHEME COMPARISON — {cancer.upper()}")
    print(f"  Research question: Should we bucket by PPR-diff instead of PEN-diff?")
    print(f"{'='*60}")

    cancer_dir = OUTPUTS / cancer
    pairs_path = cancer_dir / "pairs_with_buckets.csv"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Run script 02 first. Missing: {pairs_path}")

    print(f"\n  Loading pairs_with_buckets.csv...")
    df = pd.read_csv(pairs_path)
    print(f"  {len(df):,} gene pairs loaded\n")

    # ── Run experiments ──
    q_df   = exp1_bucket_quality(cancer, df)
    r_df   = exp2_ranker_comparison(cancer, df)
    d_df   = exp3_discovery_space(cancer, df)

    # ── Save results ──
    out_dir = cancer_dir
    q_df.to_csv(out_dir / "bucket_scheme_quality.csv", index=False)
    r_df.to_csv(out_dir / "bucket_scheme_ranker.csv",  index=False)
    d_df.to_csv(out_dir / "bucket_scheme_discovery.csv", index=False)

    print(f"\n{'='*60}")
    print(f"  SUMMARY — {cancer.upper()}")
    print(f"{'='*60}\n")

    print("  Bucket Quality:")
    print(q_df.to_string(index=False))
    print("\n  Ranker Performance:")
    print(r_df.to_string(index=False))
    print("\n  Novel Discovery Space:")
    print(d_df.to_string(index=False))

    print(f"\n  Saved to: {out_dir}/bucket_scheme_*.csv\n")


if __name__ == "__main__":
    main()
