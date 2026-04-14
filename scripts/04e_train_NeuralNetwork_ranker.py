"""
Script 04e: Train Neural Network Ranker
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
MODELS = PROJECT_ROOT / "models"

FEATURES = [
    "pen_diff", "dist_diff", "ppr_diff",
    "bucket_pen", "bucket_dist", "bucket_ppr",
    "pos_in_bucket_pen"
]


class RankingDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.FloatTensor(np.array(features.values, dtype=np.float32))
        self.y = torch.FloatTensor(np.array(labels.values, dtype=np.float32))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NeuralRanker(nn.Module):
    def __init__(self, input_dim=7):
        super(NeuralRanker, self).__init__()
        # Simpler architecture to prevent instability
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1)
        )
        
        # Initialize weights properly
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x).squeeze()


def recall_at_k(df_scored, k):
    hits = 0
    for gid, g in df_scored.groupby("group_id"):
        g_sorted = g.sort_values("pred", ascending=False)
        if (g_sorted.head(k)["label"] == 1).any():
            hits += 1
    return hits / df_scored["group_id"].nunique()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True, choices=["breast", "prostate"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.0001)  # Lower learning rate
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()
    
    cancer_dir = OUTPUTS / args.cancer
    train_path = cancer_dir / "train_rank_final.csv"
    if not train_path.exists():
        train_path = cancer_dir / "train_rank.csv"
    test_path = cancer_dir / "test_rank.csv"
    
    print(f"\n{'='*70}")
    print(f"  NEURAL NETWORK RANKER - {args.cancer.upper()}")
    print(f"{'='*70}\n")
    
    # Load data
    train_df = pd.read_csv(train_path).sort_values("group_id")
    test_df = pd.read_csv(test_path).sort_values("group_id")
    
    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    X_train = pd.DataFrame(
        scaler.fit_transform(train_df[FEATURES]),
        columns=FEATURES
    )
    y_train = train_df["label"].astype(int)
    
    X_test = pd.DataFrame(
        scaler.transform(test_df[FEATURES]),
        columns=FEATURES
    )
    
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}\n")
    print(f"  Training Neural Network...")
    print(f"    Epochs: {args.epochs}")
    print(f"    Learning rate: {args.lr}")
    print(f"    Batch size: {args.batch_size}\n")
    
    # Create datasets
    train_dataset = RankingDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralRanker(input_dim=len(FEATURES)).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        batch_count = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            loss = criterion(outputs, batch_y)
            
            # Check for nan
            if torch.isnan(loss):
                print(f"  ⚠ NaN detected at epoch {epoch+1}, skipping batch")
                continue
            
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else float('nan')
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")
    
    # Predict
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(np.array(X_test.values, dtype=np.float32)).to(device)
        logits = model(X_test_tensor)
        preds = torch.sigmoid(logits).cpu().numpy()
    
    # Evaluate
    test_df_copy = test_df.copy()
    test_df_copy["pred"] = preds
    
    print(f"\n=== Evaluation (test mini-queries) ===")
    print(f"Recall@1  : {recall_at_k(test_df_copy, 1):.4f}")
    print(f"Recall@3  : {recall_at_k(test_df_copy, 3):.4f}")
    print(f"Recall@5  : {recall_at_k(test_df_copy, 5):.4f}")
    print(f"Recall@10 : {recall_at_k(test_df_copy, 10):.4f}\n")
    
    # Save model and scaler
    out_dir = MODELS / args.cancer
    out_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), out_dir / "nn_ranker.pth")
    joblib.dump(scaler, out_dir / "nn_scaler.joblib")
    
    print(f"[OK] Saved model to: {out_dir / 'nn_ranker.pth'}")
    print(f"[OK] Saved scaler to: {out_dir / 'nn_scaler.joblib'}\n")


if __name__ == "__main__":
    main()