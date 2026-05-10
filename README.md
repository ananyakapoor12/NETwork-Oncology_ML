
# Network Embedding for Drug Target Discovery in Cancer Signalling Networks
## A Machine Learning Based Approach

**Final Year Project**  
**Author:** Ananya Kapoor  
**Supervisor:** Assoc Prof. Sourav S. Bhowmick  
**Institution:** Nanyang Technological University, Singapore

---

## Table of Contents

- [Relationship to PANACEA](#relationship-to-panacea)
- [Overview](#overview)
- [Results Summary](#results-summary)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Reproducing Results](#reproducing-results)
- [Citation](#citation)

---

## Relationship to PANACEA

This project is a **direct extension layer** built on top of the [PANACEA framework](https://github.com/NerissaX/PANACEA).

```
PANACEA (upstream)                   This project (ML layer)
─────────────────────────────────    ──────────────────────────────────────
panacea.py + modules                 code/01_ingest.py  →  ...  →  22
       │                                         ▲
       │  produces                               │  consumes
       ▼                                         │
outputs/<cancer>/
  ranked_pen.tsv   ──────────────▶  input/<cancer>/ranked_pen.tsv
  ranked_dist.tsv  ──────────────▶  input/<cancer>/ranked_dist.tsv
  ranked_ppr.tsv   ──────────────▶  input/<cancer>/ranked_ppr.tsv
  known_targets.txt ─────────────▶  input/<cancer>/known_targets.txt
```

Copy PANACEA's `outputs/<cancer>/` TSV files into `input/<cancer>/` before running Script 01.

---

## Overview

This project extends PANACEA by training ML ranking models to prioritise therapeutic targets and discover novel candidates in unexplored network regions.

**Contributions:**
- Five-model comparison (LightGBM, XGBoost, CatBoost, Random Forest, Neural MLP) under identical conditions
- Per-bucket model evaluation revealing heterogeneous difficulty across PEN-diff regions (Script 14)
- Composite scoring for unexplored buckets: 70% model + 30% novelty weight (Script 15)
- Confidence tier assignment (High/Medium/Low) for wet-lab prioritisation
- Bucket scheme validation — PEN-diff vs PPR-diff vs dist-diff (Script 20)
- Systematic ablation studies quantifying each feature group's contribution (Script 18)
- Heuristic and single-score baseline comparisons (Script 21)
- Full evaluation against PANACEA's native coverage@m% protocol (Script 22)

---

## Results Summary

### Model Performance (Explored Buckets)

| Cancer | Model | Recall@1 | Recall@10 | NDCG@10 |
|--------|-------|----------|-----------|---------|
| **Breast** | LightGBM | 45.8% | 100% | 70.6% |
| | XGBoost | 41.2% | 100% | 68.5% |
| | CatBoost | 31.3% | 100% | 62.5% |
| | Random Forest | 28.1% | 100% | 60.2% |
| | Neural MLP | 35.6% | 100% | 64.8% |
| **Prostate** | LightGBM | **73.8%** | 100% | **88.2%** |
| | XGBoost | 63.1% | 100% | 82.0% |
| | CatBoost | 60.7% | 100% | 78.9% |
| | Random Forest | 54.2% | 100% | 74.3% |
| | Neural MLP | 58.9% | 100% | 77.1% |

### Novel Candidate Discovery (Unexplored Buckets)

| Cancer | Unexplored Candidates | High Confidence | Top Candidate |
|--------|----------------------|-----------------|---------------|
| **Breast** | 59 | 0 (0%) | SALL4 combinations |
| **Prostate** | 602 | **567 (94.2%)** | RAP1GAP–PHB2 (score=1.0) |

---

## Installation

### Prerequisites
- Python 3.10, pip, 8GB+ RAM

### Setup

```bash
git clone https://github.com/ananyakapoor12/NETwork-Oncology_ML.git
cd NETwork-Oncology_ML
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Copy PANACEA outputs before running the pipeline
mkdir -p input/breast input/prostate
cp <panacea_repo>/outputs/breast/ranked_*.tsv  input/breast/
cp <panacea_repo>/outputs/breast/known_targets.txt input/breast/
cp <panacea_repo>/outputs/prostate/ranked_*.tsv input/prostate/
cp <panacea_repo>/outputs/prostate/known_targets.txt input/prostate/
```

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `lightgbm` | 4.6.0 | LambdaMART ranker |
| `xgboost` | 3.2.0 | Pairwise ranker |
| `catboost` | 1.2.10 | YetiRank ranker |
| `scikit-learn` | 1.7.2 | Random Forest, Neural MLP, metrics |
| `pandas` | 2.3.3 | Data handling |
| `numpy` | 2.2.6 | Numerical operations |
| `matplotlib` | 3.10.8 | Visualisations |
| `seaborn` | 0.13.2 | Statistical plots |

---

## Usage

Replace `breast` with `prostate` to run for the other cancer type.

### Phase 1 — Data Preparation

```bash
python code/01_ingest.py                          # merge PANACEA TSVs
python code/02_assign_buckets.py                  # assign histogram buckets
python code/03_build_train_pairs.py --cancer breast
python code/04_add_hard_negatives.py --cancer breast
python code/05_merge_train_data.py  --cancer breast
```

**Key outputs:** `outputs/breast/pairs_with_buckets.csv`, `train_rank_final.csv`

---

### Phase 2 — Model Training

```bash
python code/06_train_lgbm.py   --cancer breast
python code/07_train_xgboost.py --cancer breast
python code/08_train_catboost.py --cancer breast
python code/09_train_rf.py      --cancer breast
python code/10_train_nn.py      --cancer breast
```

**Key outputs:** `models/breast/` — one saved model per ranker

---

### Phase 3 — K=3 Triplet Generation

```bash
python code/11_generate_triplets.py --cancer breast
python code/12_finalize_triplets.py --cancer breast
```

**Key output:** `outputs/breast/k3_ranked_final.csv`

---

### Phase 4 — Model Evaluation

```bash
python code/13_compare_models.py          --cancer breast
python code/14_compare_models_by_bucket.py --cancer breast
python code/14_compare_models_by_bucket.py --cancer prostate
```

**Key outputs:** `model_comparison.csv`, `explored_model_comparison.csv`, `explored_bucket_summary.csv`

---

### Phase 5 — Novel Candidate Discovery

```bash
python code/15_rank_unexplored.py --cancer breast
python code/15_rank_unexplored.py --cancer prostate
```

**Key outputs:** `unexplored_ranked_all.csv`, `unexplored_top_candidates.csv`

---

### Phase 6 — Analysis

```bash
python code/17_verify_novelty.py --cancer breast
python code/18_ablation.py       --cancer breast
python code/18_ablation.py       --cancer prostate
python code/20_compare_bucket_schemes.py --cancer breast
python code/20_compare_bucket_schemes.py --cancer prostate
python code/21_compare_baselines.py      --cancer breast
python code/21_compare_baselines.py      --cancer prostate
python code/22_compare_panacea_coverage.py --cancer breast
python code/22_compare_panacea_coverage.py --cancer prostate
```

---

### Phase 7 — Visualisation

```bash
python code/16_visualize.py               # figs 1–7
python code/19_visualize_supplementary.py # figs 8–15
```

**All figures saved to `figures/`.**

---

## Project Structure

```
FYP_PANACEA/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── input/                  ← PANACEA framework outputs (not in git)
│   ├── breast/
│   │   ├── ranked_pen.tsv
│   │   ├── ranked_dist.tsv
│   │   ├── ranked_ppr.tsv
│   │   └── known_targets.txt
│   └── prostate/
│       └── (same structure)
│
├── code/
│   │   — Data preparation —
│   ├── 01_ingest.py
│   ├── 02_assign_buckets.py
│   ├── 03_build_train_pairs.py
│   ├── 04_add_hard_negatives.py
│   ├── 05_merge_train_data.py
│   │   — Model training —
│   ├── 06_train_lgbm.py
│   ├── 07_train_xgboost.py
│   ├── 08_train_catboost.py
│   ├── 09_train_rf.py
│   ├── 10_train_nn.py
│   │   — K=3 triplets —
│   ├── 11_generate_triplets.py
│   ├── 12_finalize_triplets.py
│   │   — Evaluation —
│   ├── 13_compare_models.py
│   ├── 14_compare_models_by_bucket.py
│   ├── 15_rank_unexplored.py
│   │   — Analysis & visualisation —
│   ├── 16_visualize.py
│   ├── 17_verify_novelty.py
│   ├── 18_ablation.py
│   ├── 19_visualize_supplementary.py
│   ├── 20_compare_bucket_schemes.py
│   ├── 21_compare_baselines.py
│   └── 22_compare_panacea_coverage.py
│
├── outputs/                         ← Generated CSVs (not in git)
│   ├── breast/
│   └── prostate/
│
├── models/                          ← Trained model files (not in git)
│   ├── breast/
│   └── prostate/
│
└── figures/                         ← All publication figures (in git)
```

---

## Methodology

### Features (7 total)
- `pen_diff`, `dist_diff`, `ppr_diff` — core PANACEA network scores
- `bucket_pen`, `bucket_dist`, `bucket_ppr` — PANACEA bucket assignments
- `pos_in_bucket_pen` — position within PEN-diff bucket

### Unexplored Bucket Scoring
```
composite_score = 0.7 × model_score + 0.3 × novelty_weight
```
- **High confidence**: score ≥ 67th percentile AND pen_diff > 0.15
- **Medium**: score ≥ 33rd percentile
- **Low**: score < 33rd percentile

---

## Reproducing Results

| Phase | Scripts | Runtime (both cancers) |
|-------|---------|------------------------|
| Data prep | 01–05 | ~5 min |
| Training | 06–10 | ~15 min |
| Triplets | 11–12 | ~5 min |
| Evaluation | 13–15 | ~15 min |
| Analysis | 17–22 | ~30 min |
| Visualisation | 16, 19 | ~2 min |
| **Total** | | **~72 min** |

---

## Citation

**PANACEA Framework:** https://github.com/NerissaX/PANACEA

---

MIT License | Last Updated: May 2026
