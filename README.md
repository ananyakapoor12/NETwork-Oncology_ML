
# Network Embedding for Drug Target Discovery in Cancer Signalling Networks
## A Machine Learning Based Approach

**Final Year Project**  
**Author:** Ananya Kapoor  
**Supervisor:** Assoc Prof. Sourav S. Bhowmick  
**Institution:** Nanyang Technological University, Singapore

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Results Summary](#results-summary)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Reproducing Results](#reproducing-results)
- [Citation](#citation)

---

## Overview

This project extends the **PANACEA** framework by implementing machine learning-based ranking models to:

1. **Rank therapeutic targets within explored histogram buckets** using five ML models (LightGBM, XGBoost, CatBoost, Random Forest, Neural MLP)
2. **Discover novel therapeutic targets in unexplored buckets** with zero known drug combinations
3. **Provide bucket-aware model evaluation** showing which models perform best in different PEN-diff regions
4. **Validate the PEN-diff bucketing design** empirically against alternative schemes (PPR-diff, dist-diff)

### Novel Contributions

- **Five-model comparison** — LightGBM, XGBoost, CatBoost, Random Forest, Neural MLP evaluated under identical conditions
- **Per-bucket multi-model comparison** (Script 07b) — first implementation showing bucket-specific model performance
- **Composite scoring for unexplored regions** (Script 08) — 70% model score + 30% novelty weight
- **Confidence tier assignment** (High/Medium/Low) for experimental validation prioritisation
- **Bucket scheme validation** (Script 13) — empirical comparison of PEN-diff vs PPR-diff vs dist-diff bucketing
- **Systematic ablation studies** (Script 11) — feature and group ablation quantifying each component's contribution

---

## Key Features

### Multi-Model Ranking
- **LightGBM LambdaMART** — primary ranking model (best overall performance)
- **XGBoost Pairwise** — difference-vector pairwise ranker
- **CatBoost YetiRank** — gradient boosted ranker
- **Random Forest** — classification-based ranking baseline
- **Neural MLP** — 3-layer deep learning ranker with StandardScaler normalisation

### Bucket-Aware Analysis
- Histogram buckets based on PEN-diff (network penetration difference) using PANACEA's published boundaries
- Per-bucket model evaluation revealing heterogeneous difficulty across the search space
- Bucket features contribute 12–13% to model importance (validated by ablation)

### Unexplored Discovery
- Identifies candidates in zero-coverage PEN-diff buckets
- Composite scoring combines model prediction + structural novelty
- Confidence tiers enable prioritisation for wet-lab validation

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

### Key Findings

1. **LightGBM dominates** across all explored buckets in both cancers
2. **Prostate >> Breast** in ranking performance (73.8% vs 45.8% Recall@1)
3. **Bucket features are used** (12–13% importance, validates bucket-aware approach)
4. **602 high-confidence novel prostate targets** in unexplored bucket (PEN-diff 0.80–1.13)
5. **PEN-diff bucketing validated** — PPR-diff bucketing produces 39.8% larger worst-case buckets and collapses 93.1% of prostate known targets into a single bucket, eliminating the novel discovery space

---

## Installation

### Prerequisites
- Python 3.10
- pip package manager
- 8GB+ RAM (for training)
- ~500MB disk space (excluding raw data)

### Setup

```bash
# Clone repository
git clone https://github.com/ananyakapoor12/NETwork-Oncology_ML.git
cd NETwork-Oncology_ML

# Create virtual environment (use Python 3.10 explicitly)
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
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
| `joblib` | — | Model persistence |

See `requirements.txt` for the complete list.

---

## Usage

### Step-by-Step Execution

Replace `breast` with `prostate` to run for the other cancer type. Run both cancers for all phases.

#### Phase 1: Data Preparation

```bash
# Merge PANACEA outputs and annotate known targets
python scripts/01_prepare_data.py

# Assign PANACEA histogram bucket boundaries
python scripts/02_bucket_policy.py

# Build ranking training groups (1 positive : 8 negatives, bucket-aware sampling)
python scripts/03_prepare_training_data.py --cancer breast

# Generate hard-negative training pairs
python scripts/03b_prepare_training_data_hardneg.py --cancer breast

# Merge standard and hard-negative sets
python scripts/03c_merge_training_data.py --cancer breast
```

**Outputs:**
- `outputs/breast/pairs_k2_standardized.csv` — standardised gene pairs
- `outputs/breast/pairs_with_buckets.csv` — with bucket assignments
- `outputs/breast/bucket_policy.csv` — explored/unexplored classification
- `outputs/breast/train_rank_final.csv` — training data with hard negatives

---

#### Phase 2: Model Training

```bash
python scripts/04_train_ranker.py               --cancer breast   # LightGBM LambdaMART
python scripts/04b_train_xgboost_ranker.py      --cancer breast   # XGBoost pairwise
python scripts/04c_train_catboost_ranker.py     --cancer breast   # CatBoost YetiRank
python scripts/04d_train_RandomForest_ranker.py --cancer breast   # Random Forest
python scripts/04e_train_NeuralNetwork_ranker.py --cancer breast  # Neural MLP
```

**Outputs (per cancer, in `models/breast/`):**
- `lgbm_ranker.joblib` / `lgbm_ranker.txt`
- `xgb_ranker.joblib` / `xgb_ranker.ubj`
- `catboost_ranker.cbm` / `catboost_ranker.joblib`
- `rf_ranker.joblib`
- `nn_ranker.pth` / `nn_scaler.joblib`
- `feature_importance.csv`

---

#### Phase 3: K=3 Triplet Generation

```bash
python scripts/05_generate_k3.py  --cancer breast   # generate triplet combinations
python scripts/06_finalize_k3.py  --cancer breast   # deduplicate and rank
```

**Outputs:**
- `outputs/breast/k3_ranked_final.csv`

---

#### Phase 4: Model Comparison & Per-Bucket Analysis

```bash
python scripts/07a_compare_models.py              --cancer breast   # overall metrics
python scripts/07b_multi_model_bucket_ranker.py   --cancer breast   # per-bucket breakdown
python scripts/07b_multi_model_bucket_ranker.py   --cancer prostate
```

**Outputs:**
- `outputs/breast/model_comparison.csv`
- `outputs/breast/explored_model_comparison.csv`
- `outputs/breast/explored_bucket_summary.csv`

---

#### Phase 5: Unexplored Bucket Discovery

```bash
python scripts/08_rank_unexplored_candidates.py --cancer breast
python scripts/08_rank_unexplored_candidates.py --cancer prostate
```

**Outputs:**
- `outputs/breast/unexplored_ranked_all.csv` — all candidates with composite score
- `outputs/breast/unexplored_top_candidates.csv` — top 100 high-confidence
- `outputs/breast/unexplored_by_bucket.csv` — per-bucket summary

**Key result:** 602 high-confidence prostate candidates (e.g., RAP1GAP–PHB2, score=1.0)

---

#### Phase 6: Novelty Verification & Ablation

```bash
python scripts/10_verify_novelty.py   --cancer breast    # confirm candidates are truly novel
python scripts/11_ablation_studies.py --cancer breast    # feature ablation experiments
python scripts/11_ablation_studies.py --cancer prostate
```

**Outputs:**
- `outputs/breast/novelty_verification.csv`
- `outputs/breast/ablation_study.csv`

---

#### Phase 7: Bucket Scheme Validation (addresses PEN vs PPR bucketing question)

```bash
python scripts/13_bucket_scheme_comparison.py --cancer breast
python scripts/13_bucket_scheme_comparison.py --cancer prostate
```

**Outputs:**
- `outputs/breast/bucket_scheme_quality.csv` — worst-case size, known-target spread
- `outputs/breast/bucket_scheme_ranker.csv` — Recall@1 under each scheme
- `outputs/breast/bucket_scheme_discovery.csv` — novel candidate space per scheme

---

#### Phase 8: Visualisation

```bash
# Primary figures (figs 1–7): publication-quality, colour-corrected
python scripts/09b_visualize_results_enhanced.py

# Supplementary figures (figs 8–15): performance matrices, curves, radar charts
python scripts/12_generate_final_visualizations.py
```

**Outputs (`figures/`):**

| Figure | Description |
|--------|-------------|
| `fig1_panacea_bucket_distribution.png` | Explored vs unexplored bucket regions |
| `fig2_model_comparison_across_cancers.png` | Cross-cancer model performance |
| `fig3_bucket_specific_performance.png` | Per-bucket heatmap |
| `fig4_feature_importance_comparison.png` | Feature importance by cancer |
| `fig5_novel_candidate_scores.png` | Novel candidate score distributions |
| `fig6_top_discoveries_heatmap.png` | Top 20 high-confidence candidates |
| `fig7_ablation_study.png` | Feature ablation results |
| `fig8–fig15` | Supplementary performance figures |

---

## Project Structure

```
NETwork-Oncology_ML/
│
├── README.md                                    ← This file
├── RESULTS_SUMMARY.md                           ← Quick results reference
├── EXECUTIVE_SUMMARY.md                         ← High-level summary
├── requirements.txt                             ← Python dependencies
├── .gitignore
│
├── scripts/
│   ├── 01_prepare_data.py                       ← Merge PANACEA TSVs, annotate known targets
│   ├── 02_bucket_policy.py                      ← Assign PANACEA histogram buckets
│   ├── 03_prepare_training_data.py              ← Build ranking groups (bucket-aware sampling)
│   ├── 03b_prepare_training_data_hardneg.py     ← Hard negative mining
│   ├── 03c_merge_training_data.py               ← Merge standard + hard negative sets
│   ├── 04_train_ranker.py                       ← LightGBM LambdaMART
│   ├── 04b_train_xgboost_ranker.py              ← XGBoost pairwise ranker
│   ├── 04c_train_catboost_ranker.py             ← CatBoost YetiRank
│   ├── 04d_train_RandomForest_ranker.py         ← Random Forest baseline
│   ├── 04e_train_NeuralNetwork_ranker.py        ← Neural MLP ranker
│   ├── 05_generate_k3.py                        ← K=3 triplet generation
│   ├── 05_analysis_plots.py                     ← Training diagnostics
│   ├── 06_finalize_k3.py                        ← Deduplicate and rank K=3 triplets
│   ├── 07a_compare_models.py                    ← Overall model comparison
│   ├── 07b_multi_model_bucket_ranker.py         ← Per-bucket model comparison
│   ├── 08_rank_unexplored_candidates.py         ← Novel candidate discovery
│   ├── 09_visualize_results.py                  ← Legacy visualisation script
│   ├── 09b_visualize_results_enhanced.py        ← Primary visualisation (figs 1–7)
│   ├── 10_verify_novelty.py                     ← Novelty verification
│   ├── 11_ablation_studies.py                   ← Feature and group ablation
│   ├── 12_generate_final_visualizations.py      ← Supplementary figures (figs 8–15)
│   └── 13_bucket_scheme_comparison.py           ← PEN vs PPR vs dist bucketing validation
│
├── data_raw/                                    ← Raw PANACEA outputs (not in git)
│   ├── breast/
│   │   ├── ranked_pen.tsv
│   │   ├── ranked_dist.tsv
│   │   └── ranked_ppr.tsv
│   └── prostate/
│       └── (same structure)
│
├── outputs/                                     ← Processed data & results (not in git)
│   ├── breast/
│   │   ├── pairs_k2_standardized.csv
│   │   ├── pairs_with_buckets.csv
│   │   ├── bucket_policy.csv
│   │   ├── train_rank_final.csv
│   │   ├── model_comparison.csv
│   │   ├── explored_model_comparison.csv
│   │   ├── ablation_study.csv
│   │   ├── bucket_scheme_*.csv
│   │   ├── unexplored_ranked_all.csv
│   │   ├── unexplored_top_candidates.csv
│   │   └── novelty_verification.csv
│   └── prostate/
│       └── (same structure)
│
├── models/                                      ← Trained models (not in git)
│   ├── breast/
│   │   ├── lgbm_ranker.joblib / lgbm_ranker.txt
│   │   ├── xgb_ranker.joblib / xgb_ranker.ubj
│   │   ├── catboost_ranker.cbm
│   │   ├── rf_ranker.joblib
│   │   ├── nn_ranker.pth / nn_scaler.joblib
│   │   └── feature_importance.csv
│   └── prostate/
│       └── (same structure)
│
└── figures/                                     ← Publication-quality plots (in git)
    ├── fig1_panacea_bucket_distribution.png
    ├── fig2_model_comparison_across_cancers.png
    ├── fig3_bucket_specific_performance.png
    ├── fig4_feature_importance_comparison.png
    ├── fig5_novel_candidate_scores.png
    ├── fig6_top_discoveries_heatmap.png
    ├── fig7_ablation_study.png
    └── fig8–fig15 (supplementary figures)
```

---

## Methodology

### 1. Bucket-Based Ranking Framework

**Histogram Buckets:**
- Gene pairs binned using PANACEA's published PEN-diff boundaries (5 buckets per cancer)
- Explored buckets: contain ≥1 known drug combination
- Unexplored buckets: zero known combinations (novel search space)
- PEN-diff chosen over PPR-diff/dist-diff — empirically validated in Script 13

**Features (7 total):**
- `pen_diff`, `dist_diff`, `ppr_diff` — core PANACEA network scores
- `bucket_pen`, `bucket_dist`, `bucket_ppr` — bucket assignments (categorical)
- `pos_in_bucket_pen` — position within PEN bucket (continuous)

### 2. Multi-Model Ranking

**LightGBM LambdaMART:**
```python
objective="lambdarank", num_leaves=63, learning_rate=0.05, n_estimators=500
```

**XGBoost Pairwise:**
```python
X_train = X_pos - X_neg   # difference vectors
y_train = [1] * n_pairs   # positive always ranked higher
```

**Neural MLP:**
```python
hidden_layers=(128, 64, 32), activation="relu", StandardScaler normalisation
```

### 3. Unexplored Bucket Discovery

**Composite Scoring:**
```
composite_score = 0.7 × model_score + 0.3 × novelty_weight
```

**Confidence Tiers:**
- **High**: composite_score ≥ 67th percentile AND pen_diff > 0.15
- **Medium**: composite_score ≥ 33rd percentile
- **Low**: composite_score < 33rd percentile

---

## Reproducing Results

### Expected Runtime

| Phase | Scripts | Cancer | Approx. Time |
|-------|---------|--------|-------------|
| Data prep | 01–03c | Both | ~5 min |
| Model training | 04–04e | Both | ~15 min |
| K=3 generation | 05–06 | Both | ~5 min |
| Comparison | 07a–07b | Both | ~15 min |
| Discovery | 08 | Both | ~2 min |
| Ablation | 11 | Both | ~10 min |
| Bucket validation | 13 | Both | ~10 min |
| Visualisation | 09b, 12 | — | ~2 min |

**Total: ~65 minutes for complete pipeline (both cancers)**

### Hardware Requirements

- **Minimum**: 8GB RAM, 4 cores
- **Recommended**: 16GB RAM, 8 cores
- **Disk**: ~500MB (excluding raw data)

---

## Citation

```bibtex
@mastersthesis{kapoor2026network,
  title={Network Embedding for Drug Target Discovery in Cancer Signalling Networks:
         A Machine Learning Based Approach},
  author={Kapoor, Ananya},
  year={2026},
  school={Nanyang Technological University},
  type={Final Year Project}
}
```

**PANACEA Framework:**
```bibtex
@article{panacea2024,
  title={PANACEA: Pan-cancer Analysis of Cancer Drug Target Combinations},
  author={[Bhowmick et al.]},
  journal={[Journal]},
  year={2024}
}
```

---

## Contact

**Author:** Ananya Kapoor  
**GitHub:** https://github.com/ananyakapoor12  
**Email:** ananya.kapoor.1103@gmail.com

**Supervisor:** Assoc Prof. Sourav S. Bhowmick  
**Institution:** Nanyang Technological University, Singapore

---

## License

MIT License

---

## Acknowledgments

- Assoc Prof. Sourav S. Bhowmick for supervision and guidance
- PANACEA framework developers
- College of Computing and Data Science, Nanyang Technological University, Singapore

---

**Last Updated:** April 2026
