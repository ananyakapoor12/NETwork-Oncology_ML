import re
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data_raw"
OUTPUTS = PROJECT_ROOT / "outputs"

CANCERS = ["breast", "prostate"]

def parse_pair(s: str):
    s = str(s).strip()
    s = s.strip("()[]")
    s = s.replace("'", "").replace('"', "")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None, None

def load_known_targets(path: Path) -> set[tuple[str,str]]:
    known = set()
    for line in path.read_text().splitlines():
        u,v = parse_pair(line)
        if u and v:
            known.add(tuple(sorted((u,v))))
    return known


def load_ranked(path: Path, score_name: str, keep_known: bool = False):
    df = pd.read_csv(path, sep="\t")
    print(f"[DEBUG] {path.name} columns:", df.columns.tolist())

    combo_col = [c for c in df.columns if "comb" in c.lower()][0]

    # prefer *-diff columns
    diff_cols = [c for c in df.columns if "diff" in c.lower()]
    candidates = [c for c in diff_cols if score_name.split("_")[0] in c.lower()]
    if len(candidates) != 1:
        raise ValueError(f"Could not identify {score_name} column in {path.name}. diff_cols={diff_cols}")
    score_col = candidates[0]

    # parse pair
    pairs = df[combo_col].apply(parse_pair)
    df["gene_u"] = pairs.apply(lambda x: x[0])
    df["gene_v"] = pairs.apply(lambda x: x[1])
    df = df.dropna(subset=["gene_u", "gene_v"])

    out = df[["gene_u", "gene_v", score_col]].rename(columns={score_col: score_name})

    if keep_known:
        # detect known column
        known_cols = [c for c in df.columns if "known" in c.lower()]
        if not known_cols:
            raise ValueError(f"No known column found in {path.name}. Columns={df.columns.tolist()}")
        out["is_known"] = pd.to_numeric(df[known_cols[0]], errors="coerce").fillna(0).astype(int)

    return out


def main():
    OUTPUTS.mkdir(exist_ok=True)

    for cancer in CANCERS:
        in_dir = DATA_RAW / cancer
        out_dir = OUTPUTS / cancer
        out_dir.mkdir(parents=True, exist_ok=True)

        df_pen  = load_ranked(in_dir / "ranked_pen.tsv",  "pen_diff", keep_known=True)
        df_dist = load_ranked(in_dir / "ranked_dist.tsv", "dist_diff", keep_known=False)
        df_ppr  = load_ranked(in_dir / "ranked_ppr.tsv",  "ppr_diff", keep_known=False)

        # merge on gene pairs
        df = df_pen.merge(df_dist, on=["gene_u","gene_v"], how="inner")
        df = df.merge(df_ppr, on=["gene_u","gene_v"], how="inner")
        
        # use PANACEA's own known-target flag if present
        known_col = [c for c in df.columns if "known" in c.lower()]
        if known_col:
            df["is_known"] = pd.to_numeric(df[known_col[0]], errors="coerce").fillna(0).astype(int)
        else:
            raise ValueError("No known-target column found; please check rankedresults files.")


        df.to_csv(out_dir / "pairs_k2_standardized.csv", index=False)
        print(f"[OK] {cancer}: {len(df):,} pairs")

if __name__ == "__main__":
    main()
