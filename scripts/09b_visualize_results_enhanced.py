"""
Script 09: Enhanced Visualizations

Comprehensive visualization suite for FYP presentation.
Generates 10+ publication-quality figures.
"""
from pathlib import Path
import warnings
import numpy as np 
import pandas as pd 
import matplotlib 
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import seaborn as sns   

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
MODELS = PROJECT_ROOT / "models"
FIGURES = PROJECT_ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

CANCERS = ["breast", "prostate"]
# Nature/ColorBrewer-style palette — colorblind-safe, publication-ready
PALETTE = {"breast": "#C0392B", "prostate": "#2471A3"}  # deep crimson / steel blue
UNEXPLORED_COLOR = "#27AE60"   # forest green — contrasts with both crimson and steel blue
# Per-cancer bucket feature accent colours (classic publication pairs)
BUCKET_FEAT_COLOR = {"breast": "#148F77",   # deep teal  — crimson + teal
                     "prostate": "#E67E22"}  # burnt amber — steel blue + amber

DPI = 300
FONT_TITLE = 14
FONT_LABEL = 11
FONT_TICK = 9

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']


def save_fig(fig, name: str):
    """Save figure with high DPI."""
    out = FIGURES / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [✓] {out.name}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: PANACEA Bucket Distribution
# ═══════════════════════════════════════════════════════════════════════════

def fig1_panacea_bucket_distribution():
    """Show PANACEA bucket structure and explored/unexplored regions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PANACEA Bucket Policy — Explored vs Unexplored Regions",
                 fontsize=FONT_TITLE + 2, fontweight="bold")
    
    for idx, cancer in enumerate(CANCERS):
        bucket_policy_path = OUTPUTS / cancer / "bucket_policy.csv"
        
        if not bucket_policy_path.exists():
            continue
        
        df = pd.read_csv(bucket_policy_path)
        
        # Plot 1: Bucket sizes
        ax = axes[idx, 0]
        colors = [UNEXPLORED_COLOR if row['status'] == 'UNEXPLORED' else PALETTE[cancer]
                  for _, row in df.iterrows()]

        bars = ax.bar(df['bucket'], df['n_pairs'], color=colors, alpha=0.8, edgecolor='white')
        ax.set_xlabel("Bucket ID", fontsize=FONT_LABEL)
        ax.set_ylabel("Number of Gene Pairs", fontsize=FONT_LABEL)
        ax.set_title(f"{cancer.title()} — Bucket Sizes", fontsize=FONT_LABEL, fontweight="bold")
        ax.set_yscale('log')

        # Add legend
        from matplotlib.patches import Patch # type: ignore
        legend_elements = [
            Patch(facecolor=PALETTE[cancer], label='Explored'),
            Patch(facecolor=UNEXPLORED_COLOR, label='Unexplored')
        ]
        ax.legend(handles=legend_elements, fontsize=FONT_TICK)
        
        # Plot 2: Known targets per bucket
        ax = axes[idx, 1]
        bars = ax.bar(df['bucket'], df['n_known'], color=colors, alpha=0.8, edgecolor='white')
        ax.set_xlabel("Bucket ID", fontsize=FONT_LABEL)
        ax.set_ylabel("Known Target Count", fontsize=FONT_LABEL)
        ax.set_title(f"{cancer.title()} — Known Targets per Bucket",
                     fontsize=FONT_LABEL, fontweight="bold")
        
        # Annotate unexplored buckets
        for _, row in df.iterrows():
            if row['status'] == 'UNEXPLORED':
                ax.text(row['bucket'], row['n_known'] + 5, 'Unexplored',
                        ha='center', fontsize=FONT_TICK + 2, color=UNEXPLORED_COLOR,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor=UNEXPLORED_COLOR, linewidth=1.2))
    
    plt.tight_layout()
    save_fig(fig, "fig1_panacea_bucket_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Model Comparison Across Cancers
# ═══════════════════════════════════════════════════════════════════════════

def fig2_model_comparison_across_cancers():
    """Compare all models across both cancers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Model Performance Comparison — Breast vs Prostate",
                 fontsize=FONT_TITLE + 2, fontweight="bold")
    
    all_data = []
    for cancer in CANCERS:
        comp_path = OUTPUTS / cancer / "model_comparison.csv"
        if comp_path.exists():
            df = pd.read_csv(comp_path)
            df['Cancer'] = cancer.title()
            all_data.append(df)
    
    if not all_data:
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Plot 1: Recall@1
    ax = axes[0]
    if 'Recall@1' in combined.columns:
        x = np.arange(len(CANCERS))
        width = 0.15
        models = combined['Model'].unique()
        
        for i, model in enumerate(models):
            vals = []
            for cancer in CANCERS:
                cancer_data = combined[combined['Cancer'] == cancer.title()]
                model_data = cancer_data[cancer_data['Model'] == model]
                val = model_data['Recall@1'].values[0] if len(model_data) > 0 else 0
                vals.append(val)
            
            offset = (i - len(models)/2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=model, alpha=0.85)
        
        ax.set_xticks(x)
        ax.set_xticklabels([c.title() for c in CANCERS])
        ax.set_ylabel("Recall@1", fontsize=FONT_LABEL)
        ax.set_title("Recall@1 Comparison", fontsize=FONT_LABEL, fontweight="bold")
        ax.legend(fontsize=FONT_TICK - 1, loc='lower left')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: NDCG@10
    ax = axes[1]
    if 'NDCG@10' in combined.columns:
        x = np.arange(len(CANCERS))
        width = 0.15
        models = combined['Model'].unique()
        
        for i, model in enumerate(models):
            vals = []
            for cancer in CANCERS:
                cancer_data = combined[combined['Cancer'] == cancer.title()]
                model_data = cancer_data[cancer_data['Model'] == model]
                val = model_data['NDCG@10'].values[0] if len(model_data) > 0 else 0
                vals.append(val)
            
            offset = (i - len(models)/2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=model, alpha=0.85)
        
        ax.set_xticks(x)
        ax.set_xticklabels([c.title() for c in CANCERS])
        ax.set_ylabel("NDCG@10", fontsize=FONT_LABEL)
        ax.set_title("NDCG@10 Comparison", fontsize=FONT_LABEL, fontweight="bold")
        ax.legend(fontsize=FONT_TICK - 1, loc='lower left')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_fig(fig, "fig2_model_comparison_across_cancers.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Bucket-Specific Model Performance
# ═══════════════════════════════════════════════════════════════════════════

def fig3_bucket_specific_performance():
    """Heatmap showing which models perform best in each bucket."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Model Performance by Bucket — Which Model Excels Where?",
                 fontsize=FONT_TITLE + 2, fontweight="bold")
    
    for ax, cancer in zip(axes, CANCERS):
        pivot_path = OUTPUTS / cancer / "bucket_model_comparison_pivot.csv"
        
        if not pivot_path.exists():
            ax.text(0.5, 0.5, f"{cancer.title()}\n(no bucket comparison)",
                    ha='center', va='center')
            ax.axis('off')
            continue
        
        df = pd.read_csv(pivot_path, index_col=0)
        
        # Create heatmap
        im = ax.imshow(df.T.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(range(len(df.index)))
        ax.set_xticklabels([f"Bucket {i}" for i in df.index], fontsize=FONT_TICK)
        ax.set_yticks(range(len(df.columns)))
        ax.set_yticklabels(df.columns, fontsize=FONT_TICK)
        
        # Add values
        for i in range(len(df.columns)):
            for j in range(len(df.index)):
                val = df.T.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                            color='black' if val > 0.5 else 'white',
                            fontsize=FONT_TICK - 1)
        
        ax.set_title(f"{cancer.title()} — Recall@1 by Bucket",
                     fontsize=FONT_LABEL, fontweight="bold")
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Recall@1", fontsize=FONT_TICK)
    
    plt.tight_layout()
    save_fig(fig, "fig3_bucket_specific_performance.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Feature Importance Comparison
# ═══════════════════════════════════════════════════════════════════════════

def fig4_feature_importance_comparison():
    """Compare feature importance across cancers."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Feature Importance — What Drives Drug Target Discovery?",
                 fontsize=FONT_TITLE + 2, fontweight="bold")
    
    for ax, cancer in zip(axes, CANCERS):
        path = MODELS / cancer / "feature_importance.csv"
        if not path.exists():
            ax.text(0.5, 0.5, f"{cancer.title()}\n(no data)", ha="center", va="center")
            ax.axis("off")
            continue
        
        df = pd.read_csv(path).sort_values("gain_importance", ascending=True)
        
        # Color bucket features differently
        bucket_color = BUCKET_FEAT_COLOR[cancer]
        colors = [bucket_color if 'bucket' in f.lower() else PALETTE[cancer]
                  for f in df["feature"]]

        bars = ax.barh(df["feature"], df["gain_importance"], color=colors, alpha=0.85)

        ax.set_xlabel("Gain Importance", fontsize=FONT_LABEL)
        ax.set_title(f"{cancer.title()} Cancer", fontsize=FONT_TITLE, fontweight="bold")
        ax.tick_params(labelsize=FONT_TICK)

        # Add legend
        from matplotlib.patches import Patch # type: ignore
        legend_elements = [
            Patch(facecolor=PALETTE[cancer], label="Core PANACEA features"),
            Patch(facecolor=bucket_color, label="Bucket features")
        ]
        ax.legend(handles=legend_elements, fontsize=FONT_TICK - 1, loc="lower right")
    
    plt.tight_layout()
    save_fig(fig, "fig4_feature_importance_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Novel Candidate Score Distribution
# ═══════════════════════════════════════════════════════════════════════════

def fig5_novel_candidate_scores():
    """Distribution of scores for novel candidates."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Novel Candidate Score Distribution — High-Confidence Discoveries",
                 fontsize=FONT_TITLE + 2, fontweight="bold")
    
    for ax, cancer in zip(axes, CANCERS):
        path = OUTPUTS / cancer / "unexplored_ranked_all.csv"
        if not path.exists():
            ax.text(0.5, 0.5, f"{cancer.title()}\n(no data)", ha="center", va="center")
            ax.axis("off")
            continue
        
        df = pd.read_csv(path)
        
        if "composite_score" not in df.columns or len(df) == 0:
            ax.text(0.5, 0.5, f"{cancer.title()}\n(no scores)", ha="center", va="center")
            ax.axis("off")
            continue
        
        # Plot histogram
        ax.hist(df["composite_score"], bins=40, alpha=0.7, color=PALETTE[cancer],
                edgecolor="white", linewidth=0.5)
        
        # Add confidence thresholds
        if "confidence" in df.columns:
            tier_counts = df["confidence"].value_counts()
            
            # Calculate thresholds (assuming top 67% is High, next 25% is Medium, rest is Low)
            sorted_scores = df["composite_score"].sort_values(ascending=False)
            high_cutoff = sorted_scores.iloc[int(len(sorted_scores) * 0.67)] if len(sorted_scores) > 1 else 0
            
            ax.axvline(high_cutoff, color="red", linestyle="--", linewidth=2,
                       label=f"High threshold ({high_cutoff:.2f})")
            
            # Annotate counts
            text = f"High: {tier_counts.get('High', 0):,}\n"
            text += f"Medium: {tier_counts.get('Medium', 0):,}\n"
            text += f"Low: {tier_counts.get('Low', 0):,}"
            ax.text(0.98, 0.98, text, transform=ax.transAxes,
                    fontsize=FONT_TICK, ha="right", va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        
        ax.set_xlabel("Composite Score (70% ML + 30% Novelty)", fontsize=FONT_LABEL)
        ax.set_ylabel("Number of Candidates", fontsize=FONT_LABEL)
        ax.set_title(f"{cancer.title()} — {len(df):,} Novel Targets",
                     fontsize=FONT_LABEL, fontweight="bold")
        ax.legend(fontsize=FONT_TICK - 1)
    
    plt.tight_layout()
    save_fig(fig, "fig5_novel_candidate_scores.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Top Discoveries Heatmap
# ═══════════════════════════════════════════════════════════════════════════

def fig6_top_discoveries_heatmap():
    """Heatmap of top novel candidates."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Top 20 High-Confidence Novel Drug Target Pairs",
                 fontsize=FONT_TITLE + 2, fontweight="bold")
    
    for ax, cancer in zip(axes, CANCERS):
        path = OUTPUTS / cancer / "unexplored_top_candidates.csv"
        if not path.exists():
            ax.text(0.5, 0.5, f"{cancer.title()}\n(no data)", ha="center", va="center")
            ax.axis("off")
            continue
        
        df = pd.read_csv(path).head(20)
        
        if len(df) == 0:
            ax.text(0.5, 0.5, f"{cancer.title()}\n(empty)", ha="center", va="center")
            ax.axis("off")
            continue
        
        # Select score columns
        score_cols = [c for c in ["pen_diff", "dist_diff", "ppr_diff",
                                    "model_score", "composite_score"]
                      if c in df.columns]
        
        if len(score_cols) == 0:
            ax.text(0.5, 0.5, f"{cancer.title()}\n(no scores)", ha="center", va="center")
            ax.axis("off")
            continue
        
        matrix = df[score_cols].values.astype(float)
        
        # Normalize columns
        col_min = matrix.min(axis=0, keepdims=True)
        col_max = matrix.max(axis=0, keepdims=True)
        matrix_norm = (matrix - col_min) / (col_max - col_min + 1e-12)
        
        # Plot heatmap
        im = ax.imshow(matrix_norm, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        
        # Labels
        ax.set_xticks(range(len(score_cols)))
        ax.set_xticklabels(score_cols, rotation=35, ha="right", fontsize=FONT_TICK)
        
        if "gene_u" in df.columns and "gene_v" in df.columns:
            ylabels = [f"{u}–{v}" for u, v in zip(df["gene_u"], df["gene_v"])]
        else:
            ylabels = [f"Rank {i+1}" for i in range(len(df))]
        
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(ylabels, fontsize=max(6, FONT_TICK - 1))
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Normalized Score", fontsize=FONT_TICK)
        
        ax.set_title(f"{cancer.title()} — Top {len(df)}",
                     fontsize=FONT_LABEL, fontweight="bold")
    
    plt.tight_layout()
    save_fig(fig, "fig6_top_discoveries_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7: Ablation Study Results
# ═══════════════════════════════════════════════════════════════════════════

def fig7_ablation_study():
    """Visualize ablation study results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Ablation Study — Feature Contribution Analysis",
                 fontsize=FONT_TITLE + 2, fontweight="bold")
    
    for ax, cancer in zip(axes, CANCERS):
        path = OUTPUTS / cancer / "ablation_study.csv"
        
        if not path.exists():
            ax.text(0.5, 0.5, f"{cancer.title()}\n(no ablation data)",
                    ha='center', va='center')
            ax.axis('off')
            continue
        
        df = pd.read_csv(path)
        
        # Remove baseline
        df = df[df['Ablation'] != 'Baseline (all features)']
        
        # Parse Delta
        df['Delta_float'] = df['Delta'].replace('—', '0').astype(float)
        df = df.sort_values('Delta_float')  # type: ignore
        
        # Plot
        colors = ['#FF6B6B' if x < -0.01 else '#90EE90' for x in df['Delta_float']]
        bars = ax.barh(range(len(df)), df['Delta_float'], color=colors, alpha=0.8)
        
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['Ablation'], fontsize=FONT_TICK - 1)
        ax.set_xlabel("Change in Recall@1", fontsize=FONT_LABEL)
        ax.set_title(f"{cancer.title()} Cancer", fontsize=FONT_LABEL, fontweight="bold")
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    save_fig(fig, "fig7_ablation_study.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*70}")
    print(f"  GENERATING COMPREHENSIVE VISUALIZATIONS")
    print(f"{'='*70}\n")
    
    print("  Creating figures...")
    
    fig1_panacea_bucket_distribution()
    fig2_model_comparison_across_cancers()
    fig3_bucket_specific_performance()
    fig4_feature_importance_comparison()
    fig5_novel_candidate_scores()
    fig6_top_discoveries_heatmap()
    fig7_ablation_study()
    
    print(f"\n{'='*70}")
    print(f"  ✅ All figures saved to: {FIGURES.resolve()}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()