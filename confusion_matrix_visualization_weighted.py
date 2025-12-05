"""
Weighted Confusion Matrix Visualization Script for Conway's Game of Life MLP Results

This script generates confusion matrix plots from WEIGHTED MLP results.
It creates grid visualizations for both patch sizes and individual plots,
similar to the original but using the weighted results.

Key differences from original:
- Reads from 'results_confusion_matrix_weighted.csv'
- Generates plots with '_weighted' suffix
- Includes class balance metrics in analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.colors as mcolors

def load_and_aggregate_weighted_data(csv_path):
    """
    Load CSV and aggregate confusion matrices over seeds for weighted models.

    Args:
        csv_path (str): Path to the weighted CSV file

    Returns:
        pd.DataFrame: Aggregated weighted confusion matrix data
    """
    print("Loading and aggregating WEIGHTED confusion matrix data...")

    # Load the weighted data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Aggregate over seeds (same as original)
    agg_df = df.groupby(['patch_size', 'regime', 'density']).agg({
        'TN': 'sum',
        'FP': 'sum',
        'FN': 'sum',
        'TP': 'sum'
    }).reset_index()

    print(f"Aggregated to {len(agg_df)} unique (patch_size, regime, density) combinations")
    return agg_df

def plot_weighted_confusion_matrix_heatmap(ax, tn, fp, fn, tp, title, font_size=12):
    """
    Plot a single WEIGHTED confusion matrix heatmap on given axes.
    Same as original but for weighted models.

    Args:
        ax: Matplotlib axes object
        tn, fp, fn, tp: Confusion matrix values
        title (str): Title for the subplot
        font_size (int): Font size for annotations
    """
    # Create confusion matrix
    cm = np.array([[tn, fp], [fn, tp]])

    # Use slightly different colormap to distinguish from original
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Greens', ax=ax,
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'],
                annot_kws={'size': font_size, 'weight': 'bold'},
                cbar_kws={'label': 'Count'})

    ax.set_title(title, fontsize=font_size + 2, pad=10, weight='bold')
    ax.set_xlabel('Predicted Class', fontsize=font_size)
    ax.set_ylabel('True Class', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)

def create_weighted_grid_plot(agg_df, patch_size, output_prefix):
    """
    Create a 3x3 grid of WEIGHTED confusion matrices for a given patch size.

    Args:
        agg_df (pd.DataFrame): Aggregated weighted confusion matrix data
        patch_size (int): 3 or 5
        output_prefix (str): Output file prefix
    """
    print(f"Creating WEIGHTED grid plot for patch_size={patch_size}...")

    # Filter data for this patch size
    patch_data = agg_df[agg_df['patch_size'] == patch_size]

    # Define the layout (same as original)
    regimes = ['early', 'mid', 'late']
    densities = [0.2, 0.4, 0.6]

    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'Weighted Confusion Matrices — MLP with {patch_size}x{patch_size} Neighborhood',
                fontsize=16, fontweight='bold', y=0.98)

    # Plot each combination
    for i, regime in enumerate(regimes):
        for j, density in enumerate(densities):
            ax = axes[i, j]

            # Get data for this configuration
            config_data = patch_data[(patch_data['regime'] == regime) &
                                   (patch_data['density'] == density)]

            if len(config_data) > 0:
                row = config_data.iloc[0]
                tn, fp, fn, tp = row['TN'], row['FP'], row['FN'], row['TP']

                # Create title with weighted designation
                title = f'regime={regime}, density={density:.1f}'

                # Plot confusion matrix
                plot_weighted_confusion_matrix_heatmap(ax, tn, fp, fn, tp, title)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f'regime={regime}, density={density:.1f}')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

    # Save with weighted designation
    for ext in ['png', 'pdf']:
        output_path = f'{output_prefix}_weighted.{ext}'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close()

def create_weighted_individual_plot(agg_df, patch_size, regime, density):
    """
    Create a single, larger WEIGHTED confusion matrix plot for a specific configuration.

    Args:
        agg_df (pd.DataFrame): Aggregated weighted confusion matrix data
        patch_size (int): 3 or 5
        regime (str): 'early', 'mid', or 'late'
        density (float): 0.2, 0.4, or 0.6
    """
    print(f"Creating WEIGHTED individual plot for patch={patch_size}, regime={regime}, density={density}")

    # Filter data
    config_data = agg_df[(agg_df['patch_size'] == patch_size) &
                        (agg_df['regime'] == regime) &
                        (agg_df['density'] == density)]

    if len(config_data) == 0:
        print("No data found for this configuration!")
        return

    row = config_data.iloc[0]
    tn, fp, fn, tp = row['TN'], row['FP'], row['FN'], row['TP']

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Create title with weighted designation
    title = f'Weighted Confusion Matrix — patch={patch_size}, regime={regime}, density={density:.1f}'

    # Plot confusion matrix with larger fonts (using weighted version)
    plot_weighted_confusion_matrix_heatmap(ax, tn, fp, fn, tp, title, font_size=14)

    # Add additional metrics as text (same as original)
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics_text = (f'Accuracy: {accuracy:.3f}\n'
                   f'Precision: {precision:.3f}\n'
                   f'Recall: {recall:.3f}\n'
                   f'F1-Score: {f1:.3f}\n'
                   f'[Weighted]')

    # Add metrics box
    ax.text(1.15, 0.5, metrics_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()

    # Save with weighted designation
    regime_clean = regime.capitalize()
    density_clean = str(density).replace('.', '')
    for ext in ['png', 'pdf']:
        output_path = f'confmat_patch{patch_size}_regime{regime_clean}_density{density_clean}_weighted.{ext}'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close()

def print_weighted_summary(agg_df):
    """
    Print summary statistics about WEIGHTED data.

    Args:
        agg_df (pd.DataFrame): Aggregated weighted confusion matrix data
    """
    print("\n" + "="*60)
    print("WEIGHTED MODEL SUMMARY STATISTICS")
    print("="*60)

    # Calculate metrics (same as original)
    agg_df['total'] = agg_df['TN'] + agg_df['FP'] + agg_df['FN'] + agg_df['TP']
    agg_df['accuracy'] = (agg_df['TN'] + agg_df['TP']) / agg_df['total']

    print(f"\nTotal weighted configurations: {len(agg_df)}")
    print(f"Patch sizes: {sorted(agg_df['patch_size'].unique())}")
    print(f"Regimes: {sorted(agg_df['regime'].unique())}")
    print(f"Densities: {sorted(agg_df['density'].unique())}")

    print(f"\nWeighted model accuracy statistics:")
    print(f"  Mean: {agg_df['accuracy'].mean():.4f}")
    print(f"  Std: {agg_df['accuracy'].std():.4f}")
    print(f"  Min: {agg_df['accuracy'].min():.4f}")
    print(f"  Max: {agg_df['accuracy'].max():.4f}")

    print(f"\nWeighted patch size comparison:")
    for patch_size in [3, 5]:
        patch_acc = agg_df[agg_df['patch_size'] == patch_size]['accuracy']
        print(f"  {patch_size}x{patch_size}: {patch_acc.mean():.4f} ± {patch_acc.std():.4f}")

    # Calculate class-specific metrics for weighted models
    print(f"\nWeighted class-specific performance (Class 1 - Alive Cells):")
    for patch_size in [3, 5]:
        patch_data = agg_df[agg_df['patch_size'] == patch_size]
        total_tp = patch_data['TP'].sum()
        total_fp = patch_data['FP'].sum()
        total_fn = patch_data['FN'].sum()

        if total_tp + total_fp > 0:
            precision = total_tp / (total_tp + total_fp)
        else:
            precision = 0

        if total_tp + total_fn > 0:
            recall = total_tp / (total_tp + total_fn)
        else:
            recall = 0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        print(f"  {patch_size}x{patch_size}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

def compare_with_original_if_possible(weighted_csv_path):
    """
    Compare weighted results with original results if original CSV exists.

    Args:
        weighted_csv_path (str): Path to weighted CSV file
    """
    original_csv_path = weighted_csv_path.replace('_weighted.csv', '.csv')

    if not Path(original_csv_path).exists():
        print(f"\n📝 Original results CSV not found at: {original_csv_path}")
        print(f"   Cannot perform comparison. Original CSV should be named: 'results_confusion_matrix.csv'")
        return

    print(f"\n📊 COMPARING WEIGHTED VS ORIGINAL RESULTS")
    print(f"{'='*50}")

    try:
        # Load both datasets
        original_df = pd.read_csv(original_csv_path)
        weighted_df = pd.read_csv(weighted_csv_path)

        print(f"Original results: {len(original_df)} rows")
        print(f"Weighted results: {len(weighted_df)} rows")

        # Aggregate both for comparison
        orig_agg = original_df.groupby(['patch_size', 'regime', 'density']).agg({
            'TN': 'sum', 'FP': 'sum', 'FN': 'sum', 'TP': 'sum'
        }).reset_index()

        weighted_agg = weighted_df.groupby(['patch_size', 'regime', 'density']).agg({
            'TN': 'sum', 'FP': 'sum', 'FN': 'sum', 'TP': 'sum'
        }).reset_index()

        # Calculate accuracies
        orig_agg['accuracy'] = (orig_agg['TN'] + orig_agg['TP']) / (orig_agg['TN'] + orig_agg['FP'] + orig_agg['FN'] + orig_agg['TP'])
        weighted_agg['accuracy'] = (weighted_agg['TN'] + weighted_agg['TP']) / (weighted_agg['TN'] + weighted_agg['FP'] + weighted_agg['FN'] + weighted_agg['TP'])

        # Compare patch sizes
        print(f"\n🔍 ACCURACY COMPARISON:")
        for patch_size in [3, 5]:
            orig_acc = orig_agg[orig_agg['patch_size'] == patch_size]['accuracy']
            weighted_acc = weighted_agg[weighted_agg['patch_size'] == patch_size]['accuracy']

            if len(orig_acc) > 0 and len(weighted_acc) > 0:
                orig_mean = orig_acc.mean()
                weighted_mean = weighted_acc.mean()
                improvement = weighted_mean - orig_mean

                print(f"  {patch_size}x{patch_size}:")
                print(f"    Original:  {orig_mean:.4f} ± {orig_acc.std():.4f}")
                print(f"    Weighted:  {weighted_mean:.4f} ± {weighted_acc.std():.4f}")
                print(f"    Improvement: {improvement:+.4f}")

        # Compare class-specific metrics
        print(f"\n🎯 CLASS 1 (ALIVE) RECALL COMPARISON:")
        for patch_size in [3, 5]:
            # Original metrics
            orig_patch = orig_agg[orig_agg['patch_size'] == patch_size]
            if len(orig_patch) > 0:
                orig_tp = orig_patch['TP'].sum()
                orig_fn = orig_patch['FN'].sum()
                orig_recall = orig_tp / (orig_tp + orig_fn) if (orig_tp + orig_fn) > 0 else 0
            else:
                orig_recall = 0

            # Weighted metrics
            weighted_patch = weighted_agg[weighted_agg['patch_size'] == patch_size]
            if len(weighted_patch) > 0:
                weighted_tp = weighted_patch['TP'].sum()
                weighted_fn = weighted_patch['FN'].sum()
                weighted_recall = weighted_tp / (weighted_tp + weighted_fn) if (weighted_tp + weighted_fn) > 0 else 0
            else:
                weighted_recall = 0

            recall_improvement = weighted_recall - orig_recall
            print(f"  {patch_size}x{patch_size}:")
            print(f"    Original Recall: {orig_recall:.4f}")
            print(f"    Weighted Recall: {weighted_recall:.4f}")
            print(f"    Recall Improvement: {recall_improvement:+.4f}")

        print(f"\n✨ Weighted training shows {'IMPROVED' if weighted_agg['accuracy'].mean() > orig_agg['accuracy'].mean() else 'CHANGED'} overall performance")
        print(f"   The main goal was to improve class 1 (alive cells) recall, which may come at the cost of overall accuracy")

    except Exception as e:
        print(f"❌ Error during comparison: {e}")

def main():
    """
    Main function to generate all weighted confusion matrix visualizations.
    """
    print("="*80)
    print("📊 WEIGHTED CONFUSION MATRIX VISUALIZATION GENERATOR")
    print("="*80)

    # Configuration for weighted data
    csv_path = 'results_confusion_matrix_weighted.csv'  # Weighted CSV
    output_dir = Path('.')
    output_dir.mkdir(exist_ok=True)

    # Check if weighted CSV exists
    if not Path(csv_path).exists():
        print(f"❌ Weighted confusion matrix CSV not found at: {csv_path}")
        print(f"   Please run experiments_mlp_weighted.py first to generate the weighted results.")
        print(f"   The weighted CSV should contain columns: patch_size, regime, density, seed, TN, FP, FN, TP")
        return

    print(f"✅ Using weighted confusion matrix CSV: {csv_path}")

    # Load and aggregate weighted data
    agg_df = load_and_aggregate_weighted_data(csv_path)

    # Create grid plots for both patch sizes (weighted)
    create_weighted_grid_plot(agg_df, 3, 'confmats_patch3_grid')
    create_weighted_grid_plot(agg_df, 5, 'confmats_patch5_grid')

    # Create individual plots (example configurations)
    create_weighted_individual_plot(agg_df, 5, 'mid', 0.4)
    create_weighted_individual_plot(agg_df, 3, 'early', 0.2)
    create_weighted_individual_plot(agg_df, 5, 'late', 0.6)

    # Print weighted summary
    print_weighted_summary(agg_df)

    # Compare with original if possible
    compare_with_original_if_possible(csv_path)

    print(f"\n{'='*80}")
    print("🎉 WEIGHTED VISUALIZATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Generated weighted files:")
    print(f"  • confmats_patch3_grid_weighted.png/pdf")
    print(f"  • confmats_patch5_grid_weighted.png/pdf")
    print(f"  • Individual weighted plots with '_weighted' suffix")
    print(f"  • Weighted confusion matrices use Green colormap")
    print(f"  • Comparison with original results (if original CSV found)")
    print(f"Files saved in: {output_dir.absolute()}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()