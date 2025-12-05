import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# Define colors
colors = {
    '3x3': '#2E86AB',  # Blue
    '5x5': '#A23B72',  # Red/Maroon
    'improvement': '#F18F01'  # Orange
}

def load_data():
    """Load and prepare data"""
    df = pd.read_csv('results_mlp_grid.csv')
    return df

def plot_overall_performance(df):
    """Figure 1: Overall Performance Comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate statistics
    stats = df.groupby('patch_size')['test_accuracy'].agg(['mean', 'std']).reset_index()

    # Create bar plot
    bars = ax.bar(['3×3', '5×5'], stats['mean'],
                  yerr=stats['std'], capsize=8,
                  color=[colors['3x3'], colors['5x5']], alpha=0.8, width=0.6)

    # Add value labels on bars
    for bar, mean_val in zip(bars, stats['mean']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + stats['std'].max() + 0.003,
                f'{mean_val:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold')

    # Formatting
    ax.set_ylabel('Test Accuracy', fontsize=14)
    ax.set_title('Overall Performance: 3×3 vs 5×5 Neighborhoods', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0.85, 0.96)
    ax.grid(True, alpha=0.3, axis='y')

    # Add improvement annotation
    improvement = stats.loc[1, 'mean'] - stats.loc[0, 'mean']
    ax.annotate(f'Improvement: +{improvement:.4f}',
                xy=(0.5, 0.88), xycoords='data',
                fontsize=13, color=colors['improvement'], fontweight='bold',
                ha='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('figure1_overall_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_regime_comparison(df):
    """Figure 2: Performance by Time Regime"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data
    regime_performance = df.groupby(['regime', 'patch_size'])['test_accuracy'].mean().unstack()
    regimes = ['early', 'mid', 'late']
    regime_labels = ['Early (10 steps)', 'Mid (60 steps)', 'Late (160 steps)']

    x = np.arange(len(regimes))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, regime_performance.loc[regimes, 3], width,
                   label='3×3', color=colors['3x3'], alpha=0.8)
    bars2 = ax.bar(x + width/2, regime_performance.loc[regimes, 5], width,
                   label='5×5', color=colors['5x5'], alpha=0.8)

    # Add improvement annotations
    for i, regime in enumerate(regimes):
        improvement = regime_performance.loc[regime, 5] - regime_performance.loc[regime, 3]
        ax.annotate(f'+{improvement:.4f}',
                    xy=(i + width/2, regime_performance.loc[regime, 5] + 0.004),
                    ha='center', fontsize=11, color=colors['improvement'], fontweight='bold')

    # Formatting
    ax.set_xlabel('Time Regime', fontsize=14)
    ax.set_ylabel('Test Accuracy', fontsize=14)
    ax.set_title('Performance Across Different Time Regimes', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(regime_labels)
    ax.legend(loc='upper left')
    ax.set_ylim(0.85, 0.96)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('figure2_regime_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_density_comparison(df):
    """Figure 3: Performance by Initial Density"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data
    density_performance = df.groupby(['p_alive', 'patch_size'])['test_accuracy'].mean().unstack()
    densities = sorted(df['p_alive'].unique())

    x = np.arange(len(densities))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, density_performance.loc[densities, 3], width,
                   label='3×3', color=colors['3x3'], alpha=0.8)
    bars2 = ax.bar(x + width/2, density_performance.loc[densities, 5], width,
                   label='5×5', color=colors['5x5'], alpha=0.8)

    # Add improvement annotations
    for i, density in enumerate(densities):
        improvement = density_performance.loc[density, 5] - density_performance.loc[density, 3]
        ax.annotate(f'+{improvement:.4f}',
                    xy=(i + width/2, density_performance.loc[density, 5] + 0.004),
                    ha='center', fontsize=11, color=colors['improvement'], fontweight='bold')

    # Formatting
    ax.set_xlabel('Initial Alive Density', fontsize=14)
    ax.set_ylabel('Test Accuracy', fontsize=14)
    ax.set_title('Performance Across Different Initial Densities', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d:.1f}' for d in densities])
    ax.legend(loc='upper left')
    ax.set_ylim(0.85, 0.96)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('figure3_density_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_improvement_heatmap(df):
    """Figure 4: Performance Improvement Heatmap"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate improvements
    regimes = ['early', 'mid', 'late']
    densities = sorted(df['p_alive'].unique())
    regime_labels = ['Early\n(10 steps)', 'Mid\n(60 steps)', 'Late\n(160 steps)']

    improvement_matrix = []
    for regime in regimes:
        row = []
        for density in densities:
            acc_3x3 = df[(df['regime'] == regime) & (df['patch_size'] == 3) &
                        (df['p_alive'] == density)]['test_accuracy'].mean()
            acc_5x5 = df[(df['regime'] == regime) & (df['patch_size'] == 5) &
                        (df['p_alive'] == density)]['test_accuracy'].mean()
            improvement = acc_5x5 - acc_3x3
            row.append(improvement)
        improvement_matrix.append(row)

    improvement_matrix = np.array(improvement_matrix)

    # Create heatmap
    im = ax.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto',
                   vmin=0.005, vmax=0.012)

    # Add text annotations
    for i in range(len(regimes)):
        for j in range(len(densities)):
            text = ax.text(j, i, f'+{improvement_matrix[i, j]:.4f}',
                           ha="center", va="center", color="black",
                           fontweight='bold', fontsize=11)

    # Formatting
    ax.set_xticks(range(len(densities)))
    ax.set_xticklabels([f'{d:.1f}' for d in densities])
    ax.set_yticks(range(len(regimes)))
    ax.set_yticklabels(regime_labels)
    ax.set_xlabel('Initial Alive Density', fontsize=14)
    ax.set_ylabel('Time Regime', fontsize=14)
    ax.set_title('Performance Improvement: 5×5 over 3×3', fontsize=16, fontweight='bold', pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Accuracy Improvement')
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Accuracy Improvement', fontsize=14)

    plt.tight_layout()
    plt.savefig('figure4_improvement_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_loss_accuracy_scatter(df):
    """Figure 5: Loss vs Accuracy Scatter Plot"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create scatter plot
    for patch_size in [3, 5]:
        subset = df[df['patch_size'] == patch_size]
        label = '3×3' if patch_size == 3 else '5×5'
        color = colors['3x3'] if patch_size == 3 else colors['5x5']

        ax.scatter(subset['test_loss'], subset['test_accuracy'],
                   c=color, label=label, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)

    # Formatting
    ax.set_xlabel('Test Loss', fontsize=14)
    ax.set_ylabel('Test Accuracy', fontsize=14)
    ax.set_title('Loss vs Accuracy Relationship', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Set reasonable limits
    ax.set_xlim(0.09, 0.32)
    ax.set_ylim(0.84, 0.96)

    plt.tight_layout()
    plt.savefig('figure5_loss_accuracy_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_stability_analysis(df):
    """Figure 6: Performance Stability (Box Plot)"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for box plot
    data_list = []
    labels = []

    for regime in ['early', 'mid', 'late']:
        for patch_size in [3, 5]:
            subset = df[(df['regime'] == regime) & (df['patch_size'] == patch_size)]['test_accuracy']
            data_list.append(subset.values)
            regime_label = regime.capitalize()
            patch_label = f'{patch_size}×{patch_size}'
            labels.append(f'{regime_label}\n{patch_label}')

    # Create box plot
    box_plot = ax.boxplot(data_list, patch_artist=True, widths=0.6)

    # Color coding
    for i, (regime, patch_size) in enumerate([(r, p) for r in ['early', 'mid', 'late'] for p in [3, 5]]):
        color = colors['3x3'] if patch_size == 3 else colors['5x5']
        box_plot['boxes'][i].set_facecolor(color)
        box_plot['boxes'][i].set_alpha(0.7)

    # Formatting
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Test Accuracy', fontsize=14)
    ax.set_title('Performance Stability Across Configurations', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.84, 0.96)

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['3x3'], alpha=0.7, label='3×3'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['5x5'], alpha=0.7, label='5×5')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.savefig('figure6_stability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Generate all figures"""
    print("Generating clean, publication-quality figures...")

    # Load data
    df = load_data()

    print("\nData Overview:")
    print(f"Total experiments: {len(df)}")
    print(f"Configurations: {df['regime'].unique()}")
    print(f"Densities: {sorted(df['p_alive'].unique())}")
    print(f"Patch sizes: {sorted(df['patch_size'].unique())}")

    # Generate figures
    print("\nGenerating Figure 1: Overall Performance Comparison...")
    plot_overall_performance(df)

    print("Generating Figure 2: Performance by Time Regime...")
    plot_regime_comparison(df)

    print("Generating Figure 3: Performance by Initial Density...")
    plot_density_comparison(df)

    print("Generating Figure 4: Performance Improvement Heatmap...")
    plot_improvement_heatmap(df)

    print("Generating Figure 5: Loss vs Accuracy Scatter Plot...")
    plot_loss_accuracy_scatter(df)

    print("Generating Figure 6: Performance Stability Analysis...")
    plot_stability_analysis(df)

    print("\nAll figures generated successfully!")
    print("Files created:")
    print("- figure1_overall_performance.png")
    print("- figure2_regime_comparison.png")
    print("- figure3_density_comparison.png")
    print("- figure4_improvement_heatmap.png")
    print("- figure5_loss_accuracy_scatter.png")
    print("- figure6_stability_analysis.png")

if __name__ == "__main__":
    main()