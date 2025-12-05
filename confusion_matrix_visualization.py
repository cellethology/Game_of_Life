"""
Confusion Matrix Visualization Script for Conway's Game of Life MLP Results

This script generates confusion matrix plots from aggregated MLP results.
It creates grid visualizations for both patch sizes and individual plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.colors as mcolors

def load_and_aggregate_data(csv_path):
    """
    Load CSV and aggregate confusion matrices over seeds.

    Args:
        csv_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Aggregated confusion matrix data
    """
    print("Loading and aggregating confusion matrix data...")

    # Load the data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Aggregate over seeds
    agg_df = df.groupby(['patch_size', 'regime', 'density']).agg({
        'TN': 'sum',
        'FP': 'sum',
        'FN': 'sum',
        'TP': 'sum'
    }).reset_index()

    print(f"Aggregated to {len(agg_df)} unique (patch_size, regime, density) combinations")
    return agg_df

def plot_confusion_matrix_heatmap(ax, tn, fp, fn, tp, title, font_size=12):
    """
    Plot a single confusion matrix heatmap on given axes.

    Args:
        ax: Matplotlib axes object
        tn, fp, fn, tp: Confusion matrix values
        title (str): Title for the subplot
        font_size (int): Font size for annotations
    """
    # Create confusion matrix
    cm = np.array([[tn, fp], [fn, tp]])

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=ax,
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'],
                annot_kws={'size': font_size, 'weight': 'bold'},
                cbar_kws={'label': 'Count'})

    ax.set_title(title, fontsize=font_size + 2, pad=10, weight='bold')
    ax.set_xlabel('Predicted Class', fontsize=font_size)
    ax.set_ylabel('True Class', fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)

def create_grid_plot(agg_df, patch_size, output_prefix):
    """
    Create a 3x3 grid of confusion matrices for a given patch size.

    Args:
        agg_df (pd.DataFrame): Aggregated confusion matrix data
        patch_size (int): 3 or 5
        output_prefix (str): Output file prefix
    """
    print(f"Creating grid plot for patch_size={patch_size}...")

    # Filter data for this patch size
    patch_data = agg_df[agg_df['patch_size'] == patch_size]

    # Define the layout
    regimes = ['early', 'mid', 'late']
    densities = [0.2, 0.4, 0.6]

    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'Confusion Matrices — MLP with {patch_size}x{patch_size} Neighborhood',
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

                # Create title
                title = f'regime={regime}, density={density:.1f}'

                # Plot confusion matrix
                plot_confusion_matrix_heatmap(ax, tn, fp, fn, tp, title)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f'regime={regime}, density={density:.1f}')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

    # Save in multiple formats
    for ext in ['png', 'pdf']:
        output_path = f'confmats_patch{patch_size}_grid.{ext}'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close()

def create_individual_plot(agg_df, patch_size, regime, density):
    """
    Create a single, larger confusion matrix plot for a specific configuration.

    Args:
        agg_df (pd.DataFrame): Aggregated confusion matrix data
        patch_size (int): 3 or 5
        regime (str): 'early', 'mid', or 'late'
        density (float): 0.2, 0.4, or 0.6
    """
    print(f"Creating individual plot for patch={patch_size}, regime={regime}, density={density}")

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

    # Create title
    title = f'Confusion Matrix — patch={patch_size}, regime={regime}, density={density:.1f}'

    # Plot confusion matrix with larger fonts
    plot_confusion_matrix_heatmap(ax, tn, fp, fn, tp, title, font_size=14)

    # Add additional metrics as text
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics_text = (f'Accuracy: {accuracy:.3f}\n'
                   f'Precision: {precision:.3f}\n'
                   f'Recall: {recall:.3f}\n'
                   f'F1-Score: {f1:.3f}')

    # Add metrics box
    ax.text(1.15, 0.5, metrics_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()

    # Save in multiple formats
    regime_clean = regime.capitalize()
    density_clean = str(density).replace('.', '')
    for ext in ['png', 'pdf']:
        output_path = f'confmat_patch{patch_size}_regime{regime_clean}_density{density_clean}.{ext}'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close()

def print_summary(agg_df):
    """
    Print summary statistics about the data.

    Args:
        agg_df (pd.DataFrame): Aggregated confusion matrix data
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    # Calculate metrics
    agg_df['total'] = agg_df['TN'] + agg_df['FP'] + agg_df['FN'] + agg_df['TP']
    agg_df['accuracy'] = (agg_df['TN'] + agg_df['TP']) / agg_df['total']

    print(f"\nTotal configurations: {len(agg_df)}")
    print(f"Patch sizes: {sorted(agg_df['patch_size'].unique())}")
    print(f"Regimes: {sorted(agg_df['regime'].unique())}")
    print(f"Densities: {sorted(agg_df['density'].unique())}")

    print(f"\nOverall accuracy statistics:")
    print(f"  Mean: {agg_df['accuracy'].mean():.4f}")
    print(f"  Std: {agg_df['accuracy'].std():.4f}")
    print(f"  Min: {agg_df['accuracy'].min():.4f}")
    print(f"  Max: {agg_df['accuracy'].max():.4f}")

    print(f"\nPatch size comparison:")
    for patch_size in [3, 5]:
        patch_acc = agg_df[agg_df['patch_size'] == patch_size]['accuracy']
        print(f"  {patch_size}x{patch_size}: {patch_acc.mean():.4f} ± {patch_acc.std():.4f}")

def main():
    """
    Main function to generate all confusion matrix visualizations.
    """
    print("="*80)
    print("CONFUSION MATRIX VISUALIZATION GENERATOR")
    print("="*80)

    # Configuration
    csv_path = 'results_confusion_matrix.csv'  # Current directory
    output_dir = Path('.')
    output_dir.mkdir(exist_ok=True)

    # Load and aggregate data
    agg_df = load_and_aggregate_data(csv_path)

    # Create grid plots for both patch sizes
    create_grid_plot(agg_df, 3, 'confmats_patch3')
    create_grid_plot(agg_df, 5, 'confmats_patch5')

    # Create individual plots (example configurations)
    create_individual_plot(agg_df, 5, 'mid', 0.4)
    create_individual_plot(agg_df, 3, 'early', 0.2)
    create_individual_plot(agg_df, 5, 'late', 0.6)

    # Print summary
    print_summary(agg_df)

    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE!")
    print(f"Files saved in: {output_dir.absolute()}")
    print("="*80)

if __name__ == "__main__":
    main()