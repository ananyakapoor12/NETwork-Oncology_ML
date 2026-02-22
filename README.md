
# Network Embedding for Drug Target Discovery in Cancer Signalling Networks
## A Machine Learning Based Approach

**Final Year Project**  
**Author:** Ananya Kapoor  
**GitHub:** https://github.com/ananyakapoor12/NETwork-Oncology_ML

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Results Summary](#results-summary)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Citation](#citation)

---

## Overview

This project extends the **PANACEA** framework by implementing machine learning-based ranking models to:

1. **Rank therapeutic targets within explored histogram buckets** using multiple ML models (LightGBM, XGBoost, Neural MLP)
2. **Discover novel therapeutic targets in unexplored buckets** with zero known drug combinations
3. **Provide bucket-aware model evaluation** showing which models perform best in different PEN-diff regions

### Novel Contributions

-  **Per-bucket multi-model comparison** (Script 07b) - First implementation showing bucket-specific model performance
-  **Composite scoring for unexplored regions** (Script 08) - 70% model score + 30% novelty weight
-  **Confidence tier assignment** (High/Medium/Low) for experimental validation prioritization
-  **Cross-cancer comparative analysis** revealing prostate cancer has clearer ranking signals than breast cancer

---

##  Key Features

### Multi-Model Ranking
- **LightGBM LambdaMART** - Primary ranking model (best overall performance)
- **XGBoost Pairwise** - Comparison model with strong bucket feature usage
- **Neural MLP Ranker** - Novel deep learning approach for ranking

### Bucket-Aware Analysis
- Histogram buckets based on PEN-diff (network penetration difference)
- Per-bucket model evaluation showing heterogeneous difficulty
- Bucket features contribute 12-13% to model importance

### Unexplored Discovery
- Identifies candidates in zero-coverage buckets
- Composite scoring combines model prediction + structural novelty
- Confidence tiers enable prioritization for wet-lab validation

---

## Results Summary

### Model Performance (Explored Buckets)

| Cancer | Model | Recall@1 | Recall@10 | NDCG@10 |
|--------|-------|----------|-----------|---------|
| **Breast** | LightGBM | 45.8% | 100% | 70.6% |
| | XGBoost | 41.2% | 100% | 68.5% |
| | CatBoost | 31.3% | 100% | 62.5% |
| **Prostate** | LightGBM | **73.8%** | 100% | **88.2%** |
| | XGBoost | 63.1% | 100% | 82.0% |
| | CatBoost | 60.7% | 100% | 78.9% |

### Novel Candidate Discovery (Unexplored Buckets)

| Cancer | Unexplored Candidates | High Confidence | Top Candidate |
|--------|----------------------|-----------------|---------------|
| **Breast** | 59 | 0 (0%) | SALL4 combinations |
| **Prostate** | 602 | **567 (94.2%)** | RAP1GAP–PHB2 (score=1.0) |

### Key Findings

1. **LightGBM dominates** across all explored buckets in both cancers
2. **Prostate >> Breast** in ranking performance (73.8% vs 45.8% Recall@1)
3. **Bucket features are used** (12-13% importance in XGBoost, validates bucket-aware approach)
4. **602 high-confidence novel prostate targets** in unexplored bucket (PEN-diff 0.80-1.13)
5. **Perfect NDCG@10 (100%)** in per-bucket evaluation shows correct ranking within groups

---

## Installation

### Prerequisites
- Python 3.10+
- pip package manager
- 8GB+ RAM (for training)
- ~500MB disk space (excluding raw data)

### Setup

```bash
# Clone repository
git clone https://github.com/ananyakapoor12/NETwork-Oncology_ML.git
cd NETwork-Oncology_ML

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Core packages:
- `lightgbm` - LambdaRank model
- `xgboost` - Pairwise ranking
- `catboost` - YetiRank model
- `scikit-learn` - Neural MLP, metrics
- `pandas`, `numpy` - Data handling
- `matplotlib` - Visualizations
- `joblib` - Model persistence

See `requirements.txt` for exact versions.

---

## Usage

### Quick Start

Run the complete pipeline for one cancer type:

```bash
# Full pipeline for breast cancer
bash scripts/run_pipeline.sh breast

# Or run manually step-by-step (see below)
```

### Step-by-Step Execution

#### Phase 1: Data Preparation (Scripts 01-03)

```bash
# Step 1: Merge PANACEA outputs
python scripts/01_prepare_data.py

# Step 2: Create histogram buckets
python scripts/02_bucket_policy.py

# Step 3: Prepare training data
python scripts/03_prepare_training_data.py --cancer breast
python scripts/03b_prepare_training_data_hardneg.py --cancer breast
python scripts/03c_merge_training_data.py --cancer breast
```

**Outputs:** 
- `outputs/breast/pairs_k2_standardized.csv` - Standardized gene pairs
- `outputs/breast/bucket_table_pen.csv` - Bucket assignments
- `outputs/breast/train_rank_final.csv` - Training data with hard negatives

---

#### Phase 2: Model Training (Scripts 04)

```bash
# Train LightGBM (primary model)
python scripts/04_train_ranker.py --cancer breast

# Train XGBoost (comparison)
python scripts/04b_train_xgboost_ranker.py --cancer breast

# Train CatBoost (comparison)
python scripts/04c_train_catboost_ranker.py --cancer breast
```

**Outputs:**
- `models/breast/lgbm_ranker.joblib` - Trained LightGBM model
- `models/breast/xgb_ranker.ubj` - Trained XGBoost model
- `models/breast/catboost_ranker.cbm` - Trained CatBoost model
- `models/breast/feature_importance.csv` - Feature importance rankings

---

#### Phase 3: K=3 Generation (Scripts 05-06)

```bash
# Generate K=3 triplet combinations
python scripts/05_generate_k3.py --cancer breast

# Deduplicate and finalize
python scripts/06_finalize_k3.py --cancer breast
```

**Outputs:**
- `outputs/breast/k3_ranked_final.csv` - Top K=3 therapeutic combinations

---

#### Phase 4: Model Comparison & Analysis (Scripts 07)

```bash
# Overall model comparison
python scripts/07a_compare_models.py --cancer both

# Per-bucket model comparison (NEW!)
python scripts/07b_multi_model_bucket_ranker.py --cancer breast
python scripts/07b_multi_model_bucket_ranker.py --cancer prostate
```

**Outputs:**
- `models/breast/model_comparison.csv` - Overall metrics
- `outputs/breast/explored_model_comparison.csv` - Per-bucket metrics
- `outputs/breast/explored_bucket_summary.csv` - Best model per bucket
- Figures showing model performance

---

#### Phase 5: Unexplored Bucket Discovery (Script 08)

```bash
# Discover novel candidates in unexplored buckets
python scripts/08_rank_unexplored_candidates.py --cancer breast
python scripts/08_rank_unexplored_candidates.py --cancer prostate
```

**Outputs:**
- `outputs/breast/unexplored_ranked_all.csv` - All unexplored candidates ranked
- `outputs/breast/unexplored_top_candidates.csv` - Top 100 high-confidence
- `outputs/breast/unexplored_by_bucket.csv` - Per-bucket summary

**Key Result:** 602 high-confidence prostate candidates (e.g., RAP1GAP–PHB2)

---

#### Phase 6: Visualization (Script 09)

```bash
# Generate all publication-quality figures
python scripts/09_visualize_results.py
```

**Outputs (figures/):**
- `fig4_unexplored_score_distribution.png` - Novel candidate distribution
- `fig5_feature_importance.png` - Bucket feature usage proof
- `fig6_top_novel_candidates_heatmap.png` - Top targets visualization
- `fig7_model_comparison_summary.png` - Cross-cancer comparison

---

## Project Structure

```
NETwork-Oncology_ML/
│
├── README.md                          ← This file
├── requirements.txt                   ← Python dependencies
├── .gitignore                         ← Git exclusions
├── SCRIPT_07_GUIDE.md                 ← 07a vs 07b explanation
├── RESULTS_SUMMARY.md                 ← Quick results reference
│
├── scripts/                           ← All Python scripts
│   ├── 01_prepare_data.py             ← Data merging
│   ├── 02_bucket_policy.py            ← Histogram buckets
│   ├── 03*.py                         ← Training data prep
│   ├── 04*.py                         ← Model training
│   ├── 05*.py                         ← K=3 generation
│   ├── 06*.py                         ← K=3 finalization
│   ├── 07a_compare_models.py          ← Overall comparison
│   ├── 07b_multi_model_bucket_ranker.py ← Per-bucket comparison 
│   ├── 08_rank_unexplored_candidates.py ← Novel discovery 
│   └── 09_visualize_results.py        ← Visualization suite
│
├── data_raw/                          ← Raw PANACEA outputs (not in git)
│   ├── breast/
│   │   ├── ranked_pen.tsv
│   │   ├── ranked_dist.tsv
│   │   └── ranked_ppr.tsv
│   └── prostate/
│       └── ... (same structure)
│
├── outputs/                           ← Processed data & results
│   ├── breast/
│   │   ├── pairs_k2_standardized.csv
│   │   ├── bucket_table_pen.csv
│   │   ├── explored_model_comparison.csv  
│   │   ├── unexplored_ranked_all.csv      
│   │   └── unexplored_top_candidates.csv  
│   └── prostate/
│       └── ... (same structure)
│
├── models/                            ← Trained models (large files not in git)
│   ├── breast/
│   │   ├── lgbm_ranker.joblib
│   │   ├── xgb_ranker.ubj
│   │   ├── model_comparison.csv
│   │   └── feature_importance.csv
│   └── prostate/
│       └── ... (same structure)
│
└── figures/                           ← Publication-quality plots
    ├── fig4_unexplored_score_distribution.png  
    ├── fig5_feature_importance.png             
    ├── fig6_top_novel_candidates_heatmap.png   
    ├── fig7_model_comparison_summary.png       
    └── ... (11 more plots)
```

---

## Methodology

### 1. Bucket-Based Ranking Framework

**Histogram Buckets:**
- Gene pairs binned by PEN-diff (network penetration difference)
- 5 equi-width buckets per cancer type
- Explored buckets: contain ≥1 known drug combination
- Unexplored buckets: zero known combinations (novel space)

**Features (7 total):**
- `pen_diff`, `dist_diff`, `ppr_diff` - Core PANACEA scores
- `bucket_pen`, `bucket_dist`, `bucket_ppr` - Bucket assignments (categorical)
- `pos_in_bucket_pen` - Position within PEN bucket (continuous)

### 2. Multi-Model Ranking (Explored Buckets)

**LightGBM LambdaMART:**
```python
objective="lambdarank"
num_leaves=63
learning_rate=0.05
n_estimators=500
```

**XGBoost Pairwise:**
```python
# Pairwise learning: pos vs neg difference vectors
X_train = X_pos - X_neg
y_train = [1] * n_pairs  # pos is better
```

**Neural MLP:**
```python
# Regression on +1 (pos) vs -1 (neg)
hidden_layers=(128, 64)
activation="relu"
StandardScaler normalization
```

### 3. Unexplored Bucket Discovery

**Composite Scoring:**
```
composite_score = 0.7 * model_score + 0.3 * novelty_weight
```

**Novelty Weight:**
```python
weight(bucket) = mean_pen_diff * pen_bonus * (1 + log(n_candidates)/10)
# Normalized to sum=1
```

**Confidence Tiers:**
- **High**: composite_score ≥ 67th percentile AND pen_diff > 0.15
- **Medium**: composite_score ≥ 33rd percentile
- **Low**: composite_score < 33rd percentile

---

## Reproducing Results

### Expected Runtime

| Script | Cancer | Time | Output |
|--------|--------|------|--------|
| 01-03 | Both | ~5 min | Training data |
| 04 (all) | Both | ~10 min | Trained models |
| 07b | Breast | ~8 min | Per-bucket metrics |
| 07b | Prostate | ~5 min | Per-bucket metrics |
| 08 | Both | ~2 min | Novel candidates |
| 09 | Both | ~1 min | All figures |

**Total: ~30 minutes for complete pipeline**

### Hardware Requirements

- **Minimum**: 8GB RAM, 4 cores
- **Recommended**: 16GB RAM, 8 cores
- **Disk**: 500MB (excluding raw data)

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{kapoor2026network,
  title={Network Embedding for Drug Target Discovery in Cancer Signalling Networks: A Machine Learning Based Approach},
  author={Kapoor, Ananya},
  year={2026},
  school={[Your University]},
  type={Final Year Project}
}
```

**PANACEA Framework:**
```bibtex
@article{panacea2024,
  title={PANACEA: Pan-cancer Analysis of Cancer Drug Target Combinations},
  author={[Professor's Name] et al.},
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

- Professor Sourav S. Bhwomick for supervision and guidance
- PANACEA framework developers
- College of Computing and Data Science, Nanyang Technological University, Singapore

---

**Last Updated:** February 2026