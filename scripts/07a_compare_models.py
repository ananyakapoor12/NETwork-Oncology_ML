"""
Script 07a: Compare Models

Compares different ML model architectures (LightGBM, XGBoost, CatBoost, RF, NN)
on the same test set and generates a comparison report.
"""
from pathlib import Path
import pandas as pd
import numpy as np 
import json
import joblib 
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
MODELS = PROJECT_ROOT / "models"

FEATURES = [
    "pen_diff", "dist_diff", "ppr_diff",
    "bucket_pen", "bucket_dist", "bucket_ppr",
    "pos_in_bucket_pen"
]


def recall_at_k(df_scored, k):
    """Compute Recall@K for ranking groups."""
    hits = 0
    for gid, g in df_scored.groupby("group_id"):
        g_sorted = g.sort_values("pred", ascending=False)
        if (g_sorted.head(k)["label"] == 1).any():
            hits += 1
    return hits / df_scored["group_id"].nunique()


def ndcg_at_k(df_scored, k):
    """Compute NDCG@K for ranking groups."""
    ndcgs = []
    for gid, g in df_scored.groupby("group_id"):
        g_sorted = g.sort_values("pred", ascending=False).head(k)
        dcg = sum((2 ** lab - 1) / np.log2(idx + 2)
                  for idx, lab in enumerate(g_sorted["label"]))
        ideal = sum((2 ** lab - 1) / np.log2(idx + 2)
                    for idx, lab in enumerate(sorted(g["label"], reverse=True)[:k]))
        ndcgs.append(dcg / ideal if ideal > 0 else 0)
    return np.mean(ndcgs)


def load_and_predict_lgbm(model_dir, X_test):
    """Load and predict with LightGBM."""
    try:
        import lightgbm as lgb
        model_path = model_dir / "lgbm_ranker.txt"
        if not model_path.exists():
            return None
        model = lgb.Booster(model_file=str(model_path))
        return model.predict(X_test)
    except Exception as e:
        print(f"    ✗ LightGBM error: {str(e)[:60]}")
        return None


def load_and_predict_xgboost(model_dir, X_test):
    """Load and predict with XGBoost."""
    try:
        import xgboost as xgb 
        model_path = model_dir / "xgboost_ranker.json"
        if not model_path.exists():
            return None
        model = xgb.Booster()
        model.load_model(str(model_path))
        dtest = xgb.DMatrix(X_test)
        return model.predict(dtest)
    except Exception as e:
        print(f"    ✗ XGBoost error: {str(e)[:60]}")
        return None


def load_and_predict_catboost(model_dir, X_test):
    """Load and predict with CatBoost."""
    try:
        from catboost import CatBoost 
        model_path = model_dir / "catboost_ranker.cbm"
        if not model_path.exists():
            return None
        model = CatBoost()
        model.load_model(str(model_path))
        return model.predict(X_test)
    except Exception as e:
        print(f"    ✗ CatBoost error: {str(e)[:60]}")
        return None


def load_and_predict_rf(model_dir, X_test):
    """Load and predict with Random Forest."""
    try:
        model_path = model_dir / "rf_ranker.joblib"
        if not model_path.exists():
            return None
        model = joblib.load(model_path)
        return model.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"    ✗ Random Forest error: {str(e)[:60]}")
        return None


def load_and_predict_nn(model_dir, X_test):
    """Load and predict with Neural Network."""
    try:
        import torch 
        import torch.nn as nn 
        from sklearn.preprocessing import StandardScaler
        
        model_path = model_dir / "nn_ranker.pth"
        scaler_path = model_dir / "nn_scaler.joblib"
        
        if not model_path.exists() or not scaler_path.exists():
            return None
        
        # Load scaler and scale features
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X_test)
        
        # Define model architecture
        class NeuralRanker(nn.Module):
            def __init__(self, input_dim=7):
                super(NeuralRanker, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.network(x).squeeze()
        
        # Load model
        device = torch.device('cpu')
        model = NeuralRanker(input_dim=len(FEATURES)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Predict
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)
            preds = model(X_tensor).numpy()
        
        return preds
    except Exception as e:
        print(f"    ✗ Neural Network error: {str(e)[:60]}")
        return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True, choices=["breast", "prostate"])
    args = ap.parse_args()
    
    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON - {args.cancer.upper()}")
    print(f"{'='*70}\n")
    
    # Load test data
    cancer_dir = OUTPUTS / args.cancer
    test_path = cancer_dir / "test_rank.csv"
    
    if not test_path.exists():
        print(f"  ✗ Test data not found: {test_path}")
        return
    
    test_df = pd.read_csv(test_path).sort_values("group_id").reset_index(drop=True)
    X_test = test_df[FEATURES]
    
    print(f"  Test samples: {len(test_df):,}")
    print(f"  Test groups: {test_df['group_id'].nunique():,}\n")
    
    model_dir = MODELS / args.cancer
    
    # Define models to test
    models_config = [
        ("LightGBM LambdaMART", load_and_predict_lgbm),
        ("XGBoost Ranker", load_and_predict_xgboost),
        ("CatBoost YetiRank", load_and_predict_catboost),
        ("Random Forest", load_and_predict_rf),
        ("Neural Network", load_and_predict_nn)
    ]
    
    results = []
    
    # Test each model
    for model_name, predict_func in models_config:
        print(f"  Testing {model_name}...")
        
        preds = predict_func(model_dir, X_test)
        
        if preds is None:
            print(f"    ⚠ Model not found, skipping\n")
            continue
        
        # Evaluate
        test_copy = test_df.copy()
        test_copy["pred"] = preds
        
        r1 = recall_at_k(test_copy, 1)
        r3 = recall_at_k(test_copy, 3)
        r5 = recall_at_k(test_copy, 5)
        r10 = recall_at_k(test_copy, 10)
        ndcg5 = ndcg_at_k(test_copy, 5)
        ndcg10 = ndcg_at_k(test_copy, 10)
        
        results.append({
            "Model": model_name,
            "Recall@1": r1,
            "Recall@3": r3,
            "Recall@5": r5,
            "Recall@10": r10,
            "NDCG@5": ndcg5,
            "NDCG@10": ndcg10
        })
        
        print(f"    ✓ Recall@1: {r1:.4f}, Recall@10: {r10:.4f}, NDCG@10: {ndcg10:.4f}\n")
    
    if not results:
        print(f"  ✗ No models found to compare")
        return
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Recall@1", ascending=False)
    
    # Save results
    output_path = cancer_dir / "model_comparison.csv"
    results_df.to_csv(output_path, index=False)
    
    # Also save to models directory for consistency
    (model_dir / "model_comparison.csv").parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(model_dir / "model_comparison.csv", index=False)
    
    print(f"{'='*70}")
    print(f"  RESULTS (sorted by Recall@1)")
    print(f"{'='*70}\n")
    
    # Pretty print
    print(f"  {'Model':<25} {'Recall@1':>10} {'Recall@10':>11} {'NDCG@10':>10}")
    print(f"  {'-'*65}")
    for _, row in results_df.iterrows():
        print(f"  {row['Model']:<25} {row['Recall@1']:>10.4f} {row['Recall@10']:>11.4f} {row['NDCG@10']:>10.4f}")
    
    print(f"\n{'='*70}")
    print(f"\n Saved comparison to:")
    print(f"   • {output_path}")
    print(f"   • {model_dir / 'model_comparison.csv'}\n")


if __name__ == "__main__":
    main()