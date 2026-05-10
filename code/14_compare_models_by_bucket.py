"""Evaluate per-bucket model performance across all PANACEA PEN-diff regions."""
from pathlib import Path
import pandas as pd 
import numpy as np 
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


def recall_at_k_by_bucket(df_scored, bucket_col="bucket_pen", k=1):
    """Compute Recall@K for each bucket."""
    bucket_recalls = {}
    
    for bucket_id in sorted(df_scored[bucket_col].unique()):
        bucket_df = df_scored[df_scored[bucket_col] == bucket_id]
        
        if len(bucket_df) == 0:
            continue
        
        hits = 0
        for gid, g in bucket_df.groupby("group_id"):
            g_sorted = g.sort_values("pred", ascending=False)
            if (g_sorted.head(k)["label"] == 1).any():
                hits += 1
        
        total_groups = bucket_df["group_id"].nunique()
        bucket_recalls[bucket_id] = hits / total_groups if total_groups > 0 else 0
    
    return bucket_recalls


def load_model_predictions(model_name, model_dir, X_test):
    """Load a specific model and generate predictions."""
    try:
        if model_name == "LightGBM":
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(model_dir / "lgbm_ranker.txt"))
            return model.predict(X_test)
        
        elif model_name == "XGBoost":
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(str(model_dir / "xgboost_ranker.json"))
            dtest = xgb.DMatrix(X_test)
            return model.predict(dtest)
        
        elif model_name == "CatBoost":
            from catboost import CatBoost
            model = CatBoost()
            model.load_model(str(model_dir / "catboost_ranker.cbm"))
            return model.predict(X_test)
        
        elif model_name == "RandomForest":
            model = joblib.load(model_dir / "rf_ranker.joblib")
            return model.predict_proba(X_test)[:, 1]
        
        elif model_name == "NeuralNet":
            import torch  
            import torch.nn as nn
            
            scaler = joblib.load(model_dir / "nn_scaler.joblib")
            X_scaled = scaler.transform(X_test)
            
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
            
            device = torch.device('cpu')
            model = NeuralRanker(input_dim=len(FEATURES)).to(device)
            model.load_state_dict(torch.load(model_dir / "nn_ranker.pth", map_location=device))
            model.eval()
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled)
                preds = model(X_tensor).numpy()
            
            return preds
        
    except Exception as e:
        print(f"    ⚠ {model_name} not available: {str(e)[:50]}")
        return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True, choices=["breast", "prostate"])
    args = ap.parse_args()
    
    print(f"\n{'='*70}")
    print(f"  BUCKET-SPECIFIC MODEL ANALYSIS - {args.cancer.upper()}")
    print(f"{'='*70}\n")
    
    # Load test data
    cancer_dir = OUTPUTS / args.cancer
    test_path = cancer_dir / "test_rank.csv"
    
    if not test_path.exists():
        print(f"  ✗ Test data not found: {test_path}")
        return
    
    test_df = pd.read_csv(test_path).sort_values("group_id").reset_index(drop=True)
    X_test = test_df[FEATURES]
    
    model_dir = MODELS / args.cancer
    
    # Models to test
    model_names = ["LightGBM", "XGBoost", "CatBoost", "RandomForest", "NeuralNet"]
    
    # Collect per-bucket results
    bucket_results = []
    
    for model_name in model_names:
        print(f"  Analyzing {model_name}...")
        
        preds = load_model_predictions(model_name, model_dir, X_test)
        
        if preds is None:
            continue
        
        test_copy = test_df.copy()
        test_copy["pred"] = preds
        
        # Get per-bucket Recall@1
        bucket_recalls = recall_at_k_by_bucket(test_copy, bucket_col="bucket_pen", k=1)
        
        for bucket_id, recall in bucket_recalls.items():
            # Get bucket metadata
            bucket_df = test_df[test_df["bucket_pen"] == bucket_id]
            n_pairs = len(bucket_df)
            n_known = int(bucket_df["label"].sum())
            
            bucket_results.append({
                "Model": model_name,
                "Bucket": int(bucket_id),
                "Recall@1": recall,
                "n_test_pairs": n_pairs,
                "n_known_targets": n_known
            })
    
    if not bucket_results:
        print(f"  ✗ No models available for bucket analysis")
        return
    
    # Create results DataFrame
    results_df = pd.DataFrame(bucket_results)
    
    # Pivot for better visualization
    pivot_df = results_df.pivot(index="Bucket", columns="Model", values="Recall@1")
    
    # Save results
    output_path = cancer_dir / "bucket_model_comparison.csv"
    results_df.to_csv(output_path, index=False)
    
    pivot_path = cancer_dir / "bucket_model_comparison_pivot.csv"
    pivot_df.to_csv(pivot_path)
    
    print(f"\n{'='*70}")
    print(f"  PER-BUCKET MODEL PERFORMANCE (Recall@1)")
    print(f"{'='*70}\n")
    
    # Pretty print pivot table
    print(pivot_df.to_string())
    
    # Find best model per bucket
    print(f"\n{'='*70}")
    print(f"  BEST MODEL PER BUCKET")
    print(f"{'='*70}\n")
    
    for bucket_id in sorted(pivot_df.index):
        bucket_scores = pivot_df.loc[bucket_id].dropna()
        if len(bucket_scores) > 0:
            best_model = bucket_scores.idxmax()
            best_score = bucket_scores.max()
            print(f"  Bucket {bucket_id}: {best_model:<15} (Recall@1: {best_score:.4f})")
    
    print(f"\n{'='*70}")
    print(f"\nSaved bucket analysis to:")
    print(f"   • {output_path}")
    print(f"   • {pivot_path}\n")


if __name__ == "__main__":
    main()