"""
external_validation.py
======================
Runs external validation of trained affinity-prediction models on the
Moulana et al. dataset and evaluates performance stratified by mutation count.

Usage
-----
    python external_validation.py \
        --model_dir  /path/to/trained_models \
        --embedding  onehot \
        --bam_csv    /path/to/cleaned_Kds_RBD_CoV555_proper.csv \
        --bam_tensor /path/to/moulana_onehot_layer15_meanpool0_...CoV555_tensor.pt \
        --imd_csv    /path/to/cleaned_Kds_RBD_REGN10987_proper.csv \
        --imd_tensor /path/to/moulana_onehot_layer15_meanpool0_...REGN10987_tensor.pt \
        --output_dir ./results

All paths that depend on your local environment are supplied as arguments —
no hard-coded paths appear in this file.
"""

import argparse
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_MODELS = [
    "Linear Regression",
    "GP",
    "MLP",
    "Ridge",
    "ElasticNet",
    "Random Forest",
    "XGBoost",
    "SVM_rbf",
    "SVM_linear",
]

# Ground-truth column name in the Moulana CSV
Y_COL = "relative_affinity"


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics for a single group of predictions."""
    if len(y_true) < 2:
        return {"MSE": np.nan, "Pearson": np.nan, "Spearman": np.nan,
                "R2": np.nan, "Count": len(y_true)}
    return {
        "MSE":      mean_squared_error(y_true, y_pred),
        "Pearson":  pearsonr(y_true, y_pred)[0],
        "Spearman": spearmanr(y_true, y_pred).correlation,
        "R2":       r2_score(y_true, y_pred),
        "Count":    len(y_true),
    }


# ---------------------------------------------------------------------------
# Prediction pipeline
# ---------------------------------------------------------------------------

def load_embedding(tensor_path: str) -> np.ndarray:
    """Load a saved embedding tensor and return a NumPy array."""
    data = torch.load(tensor_path, map_location="cpu")
    return data.numpy() if isinstance(data, torch.Tensor) else data


def run_predictions(
    df: pd.DataFrame,
    X: np.ndarray,
    model_dir: str,
    embedding_type: str,
    antibody: str,
    model_names: list,
) -> pd.DataFrame:
    """
    Load each trained model and append predictions as new columns.

    Model filenames are expected to follow the convention:
        {model_name}_{antibody}_{embedding_type}_best_model.pkl
    """
    if len(df) != X.shape[0]:
        print(
            f"  [Warning] CSV rows ({len(df)}) ≠ tensor rows ({X.shape[0]}) "
            f"for {antibody}. Results may be misaligned."
        )

    for model_name in model_names:
        pkl_name = f"{model_name}_{antibody}_{embedding_type}_best_model.pkl"
        pkl_path = os.path.join(model_dir, pkl_name)

        if not os.path.exists(pkl_path):
            print(f"  [Skip] Model file not found: {pkl_name}")
            continue

        try:
            model = joblib.load(pkl_path)
            df[f"pred_{model_name}"] = model.predict(X)
            print(f"  [OK]   {model_name}")
        except Exception as exc:
            print(f"  [Error] {model_name}: {exc}")

    return df


# ---------------------------------------------------------------------------
# Stratified evaluation
# ---------------------------------------------------------------------------

def evaluate_by_n_mut(results_dfs: dict) -> pd.DataFrame:
    """
    Compute per-antibody, per-model, per-mutation-count-group metrics.

    Sequences with more than 10 mutations are grouped as '>10'.
    """
    records = []

    for antibody, df in results_dfs.items():
        print(f"Evaluating {antibody} stratified by n_mut …")

        nan_count = df[Y_COL].isna().sum()
        if nan_count:
            print(f"  [Warning] {nan_count} NaN(s) in '{Y_COL}' — will be dropped.")

        # Create grouped mutation-count column
        df = df.copy()
        df["n_mut_group"] = df["n_mut"].apply(lambda x: x if x <= 10 else 11)
        groups = sorted(df["n_mut_group"].unique())

        pred_cols = [c for c in df.columns if c.startswith("pred_")]

        for pred_col in pred_cols:
            model_name = pred_col.replace("pred_", "")

            for group in groups:
                subset = df[df["n_mut_group"] == group].dropna(
                    subset=[Y_COL, pred_col]
                )
                if subset.empty:
                    continue

                metrics = calculate_metrics(
                    subset[Y_COL].values, subset[pred_col].values
                )
                records.append({
                    "Antibody":    antibody,
                    "Model":       model_name,
                    "n_mut_group": str(group) if group <= 10 else ">10",
                    **metrics,
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_spearman_by_n_mut(
    metrics_df: pd.DataFrame,
    antibody: str,
    embedding_type: str,
    output_dir: str,
) -> None:
    """Bar chart of Spearman ρ by mutation-count group for one antibody."""
    subset = metrics_df[metrics_df["Antibody"] == antibody]
    if subset.empty:
        print(f"No data for {antibody}. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))

    sns.barplot(
        data=subset,
        x="n_mut_group",
        y="Spearman",
        hue="Model",
        palette="tab10",
        edgecolor="white",
        linewidth=0.5,
        capsize=0,
        errwidth=0.4,
        ax=ax,
    )

    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_facecolor("white")
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.4)
        ax.spines[spine].set_color("black")
    ax.tick_params(axis="both", which="major", labelsize=12, width=0.4, color="black")

    ax.set_title(antibody, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Mutations ($n_{mut}$)", fontsize=14, labelpad=10)
    ax.set_ylabel("Spearman's $\\rho$", fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=11)

    plt.tight_layout()
    out_file = os.path.join(
        output_dir, f"moulana_Spearman_nmut_{antibody}_{embedding_type}.pdf"
    )
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


# ---------------------------------------------------------------------------
# Scatter plot: predicted vs measured, coloured by n_mut
# ---------------------------------------------------------------------------

def plot_scatter_by_n_mut(
    results_dfs: dict,
    model_name: str,
    embedding_type: str,
    output_dir: str,
    true_cutoff: float = 0.85,
    pred_cutoff: float = 0.75,
) -> None:
    """
    For each antibody, plot predicted vs measured relative affinity coloured by
    mutation count (n_mut), with an identity line and optional threshold lines.

    Each point is one sequence. Colour gradient runs from deep navy (low n_mut)
    through pale silver to blood red (high n_mut, capped at 11 → '>10').

    Parameters
    ----------
    results_dfs : dict
        {antibody: DataFrame} with columns 'relative_affinity',
        'pred_{model_name}', and 'n_mut'.
    model_name : str
        Name of the model whose predictions to plot.
    embedding_type : str
        Embedding label used in output filenames.
    output_dir : str
        Directory where PDF figures are saved.
    true_cutoff : float
        Horizontal dashed line marking the binder/non-binder threshold in the
        ground truth (default 0.85).
    pred_cutoff : float
        Vertical dashed line marking the binder/non-binder threshold in
        predictions (default 0.75).
    """
    from matplotlib.colors import LinearSegmentedColormap

    pred_col   = f"pred_{model_name}"
    target_col = "relative_affinity"

    # Navy → silver → blood-red gradient mapped to n_mut
    cmap = LinearSegmentedColormap.from_list(
        "n_mut_gradient", ["#001f3f", "#e0e0e0", "#8b0000"]
    )

    for antibody, df in results_dfs.items():
        work_df = df.copy()

        # Keep only clean rows if the flag column exists
        if "is_clean" in work_df.columns:
            work_df = work_df[work_df["is_clean"] == True]

        work_df = work_df.dropna(subset=[target_col, pred_col])

        if work_df.empty:
            print(f"  [Skip scatter] No valid data for {antibody} / {model_name}")
            continue

        # Cap n_mut at 11 so '>10' sequences still get the deepest red
        work_df["color_group"] = work_df["n_mut"].clip(upper=11)

        # ── Axis limits: independent per axis, 5 % padding ──────────────────
        def padded_lims(series: pd.Series):
            lo, hi = series.min(), series.max()
            pad = (hi - lo) * 0.05
            return lo - pad, hi + pad

        x_lims = padded_lims(work_df[target_col])
        y_lims = padded_lims(work_df[pred_col])

        # Identity line spans the full union of both axes
        diag_min = min(x_lims[0], y_lims[0])
        diag_max = max(x_lims[1], y_lims[1])

        # ── Figure ───────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(6, 6))

        sc = ax.scatter(
            work_df[target_col],
            work_df[pred_col],
            c=work_df["color_group"],
            cmap=cmap,
            s=12,
            alpha=0.55,
            edgecolors="none",
            zorder=2,
        )

        # Identity line (x = y)
        ax.plot(
            [diag_min, diag_max],
            [diag_min, diag_max],
            color="black",
            linewidth=0.8,
            linestyle="--",
            alpha=0.6,
            zorder=1,
            label="$y = x$",
        )

        # Optional threshold lines
        ax.axhline(pred_cutoff,  color="gray", linestyle=":", linewidth=0.6, alpha=0.7)
        ax.axvline(true_cutoff,  color="gray", linestyle=":", linewidth=0.6, alpha=0.7)

        # ── Colourbar ────────────────────────────────────────────────────────
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.outline.set_linewidth(0.4)
        cbar.ax.tick_params(width=0.4, labelsize=9)

        # Replace the '11' tick label with '>10'
        cbar_ticks = cbar.get_ticks()
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([
            ">10" if abs(t - 11) < 0.5 else str(int(t))
            for t in cbar_ticks
        ])
        cbar.set_label("$n_{mut}$", rotation=270, labelpad=15, fontsize=11)

        # ── Aesthetics ───────────────────────────────────────────────────────
        ax.set_box_aspect(1)
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        ax.set_facecolor("white")
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_linewidth(0.4)
            ax.spines[spine].set_color("black")
        ax.tick_params(axis="both", which="major", labelsize=11, width=0.4, color="black")

        ax.set_title(
            f"{antibody} — {model_name} ({embedding_type})",
            fontsize=13, fontweight="bold", pad=12,
        )
        ax.set_xlabel("Measured relative affinity ($y_{true}$)", fontsize=11, labelpad=8)
        ax.set_ylabel("Predicted relative affinity ($y_{pred}$)", fontsize=11, labelpad=8)

        plt.tight_layout()
        out_file = os.path.join(
            output_dir,
            f"moulana_scatter_{antibody}_{model_name}_{embedding_type}.pdf",
        )
        fig.savefig(out_file, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="External validation of affinity models on the Moulana dataset."
    )
    parser.add_argument(
        "--model_dir", required=True,
        help="Directory containing trained model .pkl files.",
    )
    parser.add_argument(
        "--embedding", required=True,
        choices=["onehot", "esm2", "16encode"],
        help="Embedding type used for the saved models and tensors.",
    )
    parser.add_argument(
        "--bam_csv", required=True,
        help="Path to Bamlanivimab (CoV555) CSV file.",
    )
    parser.add_argument(
        "--bam_tensor", required=True,
        help="Path to Bamlanivimab embedding tensor (.pt) file.",
    )
    parser.add_argument(
        "--imd_csv", required=True,
        help="Path to Imdevimab (REGN10987) CSV file.",
    )
    parser.add_argument(
        "--imd_tensor", required=True,
        help="Path to Imdevimab embedding tensor (.pt) file.",
    )
    parser.add_argument(
        "--output_dir", default="./results",
        help="Directory where output CSVs and plots are saved (default: ./results).",
    )
    parser.add_argument(
        "--scatter_model", default="SVM_linear",
        help=(
            "Model to use for the predicted-vs-measured scatter plot "
            "(default: SVM_linear). Must match a name in TARGET_MODELS."
        ),
    )
    parser.add_argument(
        "--skip_plots", action="store_true",
        help="Skip generating figures (useful for headless environments).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data_config = {
        "Bamlanivimab": {"csv": args.bam_csv, "tensor": args.bam_tensor},
        "Imdevimab":    {"csv": args.imd_csv, "tensor": args.imd_tensor},
    }

    # ── Step 1: Load data and run predictions ──────────────────────────────
    results_dfs = {}
    for antibody, paths in data_config.items():
        print(f"\n=== Processing {antibody} ===")
        df = pd.read_csv(paths["csv"])
        print(f"  Loaded CSV: {len(df)} rows")

        print(f"  Loading tensor …")
        X = load_embedding(paths["tensor"])
        print(f"  Tensor shape: {X.shape}")

        df = run_predictions(
            df, X, args.model_dir, args.embedding, antibody, TARGET_MODELS
        )
        results_dfs[antibody] = df

        # Save predictions
        pred_csv = os.path.join(
            args.output_dir,
            f"moulana_{args.embedding}_predicted_{antibody}.csv",
        )
        df["Antibody"] = antibody
        df.to_csv(pred_csv, index=False)
        print(f"  Predictions saved: {pred_csv}")

    # ── Step 2: Stratified evaluation ──────────────────────────────────────
    print("\n=== Computing stratified metrics ===")
    metrics_df = evaluate_by_n_mut(results_dfs)
    print(f"metrics_df shape: {metrics_df.shape}")

    metrics_csv = os.path.join(
        args.output_dir,
        f"moulana_{args.embedding}_metrics_by_n_mut.csv",
    )
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Metrics saved: {metrics_csv}")

    # ── Step 3: Plots ──────────────────────────────────────────────────────
    if not args.skip_plots:
        print("\n=== Generating plots ===")
        sns.set(style="whitegrid")

        # Bar charts: Spearman by mutation count group
        for antibody in metrics_df["Antibody"].unique():
            plot_spearman_by_n_mut(
                metrics_df, antibody, args.embedding, args.output_dir
            )

        # Scatter plots: predicted vs measured, coloured by n_mut
        print(f"\n=== Scatter plots (model: {args.scatter_model}) ===")
        plot_scatter_by_n_mut(
            results_dfs,
            model_name=args.scatter_model,
            embedding_type=args.embedding,
            output_dir=args.output_dir,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
