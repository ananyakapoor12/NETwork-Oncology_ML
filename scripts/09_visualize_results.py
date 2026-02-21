from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
MODELS  = PROJECT_ROOT / "models"
FIGURES = PROJECT_ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

CANCERS = ["breast", "prostate"]
PALETTE = {"breast": "#E85D75", "prostate": "#4A90D9"}

DPI = 150
FONT_TITLE = 14
FONT_LABEL = 11
FONT_TICK  = 9


def save_fig(fig, name: str):
    out = FIGURES / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [✓] Saved: {out.name}")


# ─────────────────────────────────────────────
# Fig 4: Unexplored Score Distribution
# ─────────────────────────────────────────────

def fig4_unexplored_score_distribution():
    """Plot composite score distribution from unexplored_ranked_all.csv"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Unexplored Bucket Candidate Score Distribution\n(Novel Therapeutic Targets)",
                 fontsize=FONT_TITLE, fontweight="bold")
    
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
        
        # Plot histogram with confidence tier overlays
        ax.hist(df["composite_score"], bins=40, alpha=0.7, color=PALETTE[cancer],
                edgecolor="white", linewidth=0.5)
        
        # Add vertical lines for confidence thresholds
        if "confidence" in df.columns:
            q33 = df["composite_score"].quantile(0.33)
            q67 = df["composite_score"].quantile(0.67)
            ax.axvline(q33, color="orange", linestyle="--", linewidth=1.5,
                       label=f"Medium threshold ({q33:.2f})")
            ax.axvline(q67, color="red", linestyle="--", linewidth=1.5,
                       label=f"High threshold ({q67:.2f})")
        
        # Annotate counts
        if "confidence" in df.columns:
            tier_counts = df["confidence"].value_counts()
            y_max = ax.get_ylim()[1]
            text = f"High: {tier_counts.get('High', 0)}\n"
            text += f"Medium: {tier_counts.get('Medium', 0)}\n"
            text += f"Low: {tier_counts.get('Low', 0)}"
            ax.text(0.98, 0.98, text, transform=ax.transAxes,
                    fontsize=FONT_TICK, ha="right", va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        
        ax.set_xlabel("Composite Score (70% model + 30% novelty)", fontsize=FONT_LABEL)
        ax.set_ylabel("Number of Candidate Pairs", fontsize=FONT_LABEL)
        ax.set_title(f"{cancer.title()} Cancer ({len(df):,} novel candidates)",
                     fontsize=FONT_LABEL, fontweight="bold")
        ax.legend(fontsize=FONT_TICK - 1, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    save_fig(fig, "fig4_unexplored_score_distribution.png")


# ─────────────────────────────────────────────
# Fig 5: Feature Importance (from existing CSVs)
# ─────────────────────────────────────────────

def fig5_feature_importance():
    """Plot feature importance from models/*/feature_importance.csv"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Feature Importance — LightGBM LambdaRank\n(Gain Importance)",
                 fontsize=FONT_TITLE, fontweight="bold")
    
    for ax, cancer in zip(axes, CANCERS):
        path = MODELS / cancer / "feature_importance.csv"
        if not path.exists():
            ax.text(0.5, 0.5, f"{cancer.title()}\n(no data)", ha="center", va="center")
            ax.axis("off")
            continue
        
        df = pd.read_csv(path)
        
        # Use gain_importance if available, else first numeric column
        if "gain_importance" in df.columns:
            df = df.sort_values("gain_importance", ascending=True)
            vals = df["gain_importance"]
            xlabel = "Gain Importance"
        elif df.shape[1] > 1:
            numeric_col = df.select_dtypes(include=[np.number]).columns[0]
            df = df.sort_values(numeric_col, ascending=True)
            vals = df[numeric_col]
            xlabel = numeric_col
        else:
            ax.text(0.5, 0.5, f"{cancer.title()}\n(invalid format)", ha="center", va="center")
            ax.axis("off")
            continue
        
        bars = ax.barh(df["feature"], vals, color=PALETTE[cancer], alpha=0.85, edgecolor="white")
        
        # Highlight bucket features
        for i, feat in enumerate(df["feature"]):
            if "bucket" in feat.lower():
                bars[i].set_color("#FF6B6B")
                bars[i].set_alpha(1.0)
        
        ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
        ax.set_title(f"{cancer.title()} Cancer", fontsize=FONT_TITLE, fontweight="bold")
        ax.tick_params(labelsize=FONT_TICK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        # Add legend for bucket features
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=PALETTE[cancer], label="Core features"),
            Patch(facecolor="#FF6B6B", label="Bucket features (prof wanted!)")
        ]
        ax.legend(handles=legend_elements, fontsize=FONT_TICK - 1, loc="lower right")
    
    plt.tight_layout()
    save_fig(fig, "fig5_feature_importance.png")


# ─────────────────────────────────────────────
# Fig 6: Top Novel Candidates Heatmap
# ─────────────────────────────────────────────

def fig6_top_novel_candidates_heatmap():
    """Heatmap of top 20 high-confidence unexplored candidates"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Top High-Confidence Novel Therapeutic Target Combinations\n(Unexplored Buckets)",
                 fontsize=FONT_TITLE, fontweight="bold")
    
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
        
        # Select numeric columns for heatmap
        score_cols = [c for c in ["pen_diff", "dist_diff", "ppr_diff",
                                    "model_score", "composite_score", "novelty_weight"]
                      if c in df.columns]
        
        if len(score_cols) == 0:
            ax.text(0.5, 0.5, f"{cancer.title()}\n(no score columns)", ha="center", va="center")
            ax.axis("off")
            continue
        
        matrix = df[score_cols].values.astype(float)
        
        # Normalize each column to [0,1] for visualization
        col_min = matrix.min(axis=0, keepdims=True)
        col_max = matrix.max(axis=0, keepdims=True)
        matrix_norm = (matrix - col_min) / (col_max - col_min + 1e-12)
        
        # Plot heatmap
        im = ax.imshow(matrix_norm, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        
        # X-axis: feature names
        ax.set_xticks(range(len(score_cols)))
        ax.set_xticklabels(score_cols, rotation=35, ha="right", fontsize=FONT_TICK)
        
        # Y-axis: gene pair labels
        if "gene_u" in df.columns and "gene_v" in df.columns:
            ylabels = [f"{u}–{v}" for u, v in zip(df["gene_u"], df["gene_v"])]
        elif "global_rank" in df.columns:
            ylabels = [f"Rank {r}" for r in df["global_rank"]]
        else:
            ylabels = [str(i + 1) for i in range(len(df))]
        
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(ylabels, fontsize=max(6, FONT_TICK - int(len(df) / 5)))
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Normalized Score", fontsize=FONT_TICK)
        
        ax.set_title(f"{cancer.title()} — Top {len(df)} Novel Candidates",
                     fontsize=FONT_LABEL, fontweight="bold")
    
    plt.tight_layout()
    save_fig(fig, "fig6_top_novel_candidates_heatmap.png")


# ─────────────────────────────────────────────
# Fig 7: Model Comparison Summary (if available)
# ─────────────────────────────────────────────

def fig7_model_comparison_summary():
    """Cross-cancer model comparison from model_comparison.csv"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Model Performance Comparison (Explored Buckets)",
                 fontsize=FONT_TITLE, fontweight="bold")
    
    data = []
    for cancer in CANCERS:
        path = MODELS / cancer / "model_comparison.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["Cancer"] = cancer
            data.append(df)
    
    if not data:
        for ax in axes:
            ax.text(0.5, 0.5, "(no model_comparison.csv found)", ha="center", va="center")
            ax.axis("off")
        plt.tight_layout()
        save_fig(fig, "fig7_model_comparison_summary.png")
        return
    
    combined = pd.concat(data, ignore_index=True)
    
    metrics = ["Recall@1", "Recall@10", "NDCG@5", "NDCG@10"]
    metric_cols = [c for c in metrics if c in combined.columns]
    
    if len(metric_cols) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "(no metrics found)", ha="center", va="center")
            ax.axis("off")
        plt.tight_layout()
        save_fig(fig, "fig7_model_comparison_summary.png")
        return
    
    # Plot 1: Recall@1 and Recall@10
    ax = axes[0]
    recall_cols = [c for c in ["Recall@1", "Recall@10"] if c in combined.columns]
    if recall_cols:
        x = np.arange(len(CANCERS))
        width = 0.2
        models = combined["Model"].unique()
        
        for i, model in enumerate(models):
            for j, metric in enumerate(recall_cols):
                vals = [combined[(combined["Cancer"] == c) & (combined["Model"] == model)][metric].mean()
                        if len(combined[(combined["Cancer"] == c) & (combined["Model"] == model)]) > 0 else 0
                        for c in CANCERS]
                offset = (i - len(models) / 2 + 0.5) * width + j * 0.05
                ax.bar(x + offset, vals, width=width, label=f"{model} {metric}" if j == 0 else "")
        
        ax.set_xticks(x)
        ax.set_xticklabels([c.title() for c in CANCERS])
        ax.set_ylabel("Recall Score")
        ax.set_title("Recall@1 and Recall@10 Comparison", fontsize=FONT_LABEL, fontweight="bold")
        ax.legend(fontsize=FONT_TICK - 1)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    # Plot 2: NDCG
    ax = axes[1]
    ndcg_cols = [c for c in ["NDCG@5", "NDCG@10"] if c in combined.columns]
    if ndcg_cols:
        x = np.arange(len(CANCERS))
        width = 0.2
        models = combined["Model"].unique()
        
        for i, model in enumerate(models):
            for j, metric in enumerate(ndcg_cols):
                vals = [combined[(combined["Cancer"] == c) & (combined["Model"] == model)][metric].mean()
                        if len(combined[(combined["Cancer"] == c) & (combined["Model"] == model)]) > 0 else 0
                        for c in CANCERS]
                offset = (i - len(models) / 2 + 0.5) * width + j * 0.05
                ax.bar(x + offset, vals, width=width, label=f"{model} {metric}" if j == 0 else "")
        
        ax.set_xticks(x)
        ax.set_xticklabels([c.title() for c in CANCERS])
        ax.set_ylabel("NDCG Score")
        ax.set_title("NDCG@5 and NDCG@10 Comparison", fontsize=FONT_LABEL, fontweight="bold")
        ax.legend(fontsize=FONT_TICK - 1)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    save_fig(fig, "fig7_model_comparison_summary.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print(f"\n{'='*70}")
    print(f" Generating Additional Figures for Friday Demo")
    print(f"{'='*70}\n")
    
    fig4_unexplored_score_distribution()
    fig5_feature_importance()
    fig6_top_novel_candidates_heatmap()
    fig7_model_comparison_summary()
    
    print(f"\n All figures saved to: {FIGURES.resolve()}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()