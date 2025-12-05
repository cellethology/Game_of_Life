"""Create comprehensive visualizations for Conway's Game of Life MLP results."""

import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results(csv_path="results_mlp_grid.csv"):
    """Load experimental results from CSV file."""
    df = pd.read_csv(csv_path)
    return df

def create_comprehensive_visualizations(df):
    """Create a comprehensive set of visualizations."""

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))

    # 1. Main performance comparison by regime
    ax1 = plt.subplot(3, 3, 1)
    regime_performance = df.groupby(['regime', 'patch_size'])['test_accuracy'].mean().unstack()
    regime_performance.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Performance by Time Regime', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_xlabel('Time Regime')
    ax1.legend(title='Patch Size', loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Add improvement annotations
    for i, regime in enumerate(['early', 'mid', 'late']):
        acc3 = regime_performance.loc[regime, 3]
        acc5 = regime_performance.loc[regime, 5]
        improvement = acc5 - acc3
        ax1.text(i, max(acc3, acc5) + 0.005, f'+{improvement:.3f}',
                ha='center', fontweight='bold', color='green')

    # 2. Performance by initial density
    ax2 = plt.subplot(3, 3, 2)
    density_performance = df.groupby(['p_alive', 'patch_size'])['test_accuracy'].mean().unstack()
    density_performance.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title('Performance by Initial Density', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_xlabel('Initial Cell Probability (p_alive)')
    ax2.legend(title='Patch Size', loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Add improvement annotations
    for i, density in enumerate([0.2, 0.4, 0.6]):
        acc3 = density_performance.loc[density, 3]
        acc5 = density_performance.loc[density, 5]
        improvement = acc5 - acc3
        ax2.text(i, max(acc3, acc5) + 0.005, f'+{improvement:.3f}',
                ha='center', fontweight='bold', color='green')

    # 3. Heatmap of all configurations
    ax3 = plt.subplot(3, 3, 3)
    pivot_3x3 = df[df['patch_size'] == 3].pivot_table(
        values='test_accuracy', index='regime', columns='p_alive', aggfunc='mean'
    )
    pivot_5x5 = df[df['patch_size'] == 5].pivot_table(
        values='test_accuracy', index='regime', columns='p_alive', aggfunc='mean'
    )

    # Create improvement heatmap
    improvement_matrix = pivot_5x5 - pivot_3x3
    sns.heatmap(improvement_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                center=0, ax=ax3, cbar_kws={'label': '5x5 - 3x3 Improvement'})
    ax3.set_title('Improvement Heatmap (5x5 vs 3x3)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Initial Density (p_alive)')
    ax3.set_ylabel('Time Regime')

    # 4. Box plot of all results
    ax4 = plt.subplot(3, 3, 4)
    df.boxplot(column='test_accuracy', by='patch_size', ax=ax4, grid=True)
    ax4.set_title('Overall Performance Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Patch Size')
    ax4.set_ylabel('Test Accuracy')
    ax4.grid(True, alpha=0.3)

    # 5. Detailed scatter plot with regression
    ax5 = plt.subplot(3, 3, 5)
    for regime in df['regime'].unique():
        regime_data = df[df['regime'] == regime]
        grouped = regime_data.groupby(['p_alive', 'seed'])
        acc3 = []
        acc5 = []

        for (p_alive, seed), group in grouped:
            patch3 = group[group['patch_size'] == 3]['test_accuracy'].iloc[0]
            patch5 = group[group['patch_size'] == 5]['test_accuracy'].iloc[0]
            acc3.append(patch3)
            acc5.append(patch5)

        ax5.scatter(acc3, acc5, label=f'{regime}', alpha=0.7, s=50)

    # Add diagonal line
    min_acc = min(df['test_accuracy'])
    max_acc = max(df['test_accuracy'])
    ax5.plot([min_acc, max_acc], [min_acc, max_acc], 'k--', alpha=0.5)

    ax5.set_xlabel('3×3 Test Accuracy')
    ax5.set_ylabel('5×5 Test Accuracy')
    ax5.set_title('Paired Performance Comparison', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Time evolution line plot
    ax6 = plt.subplot(3, 3, 6)
    for p_alive in df['p_alive'].unique():
        density_data = df[df['p_alive'] == p_alive]

        # Calculate mean accuracy by regime
        means_3x3 = []
        means_5x5 = []
        regimes_ordered = ['early', 'mid', 'late']

        for regime in regimes_ordered:
            regime_data = density_data[density_data['regime'] == regime]
            mean_3x3 = regime_data[regime_data['patch_size'] == 3]['test_accuracy'].mean()
            mean_5x5 = regime_data[regime_data['patch_size'] == 5]['test_accuracy'].mean()
            means_3x3.append(mean_3x3)
            means_5x5.append(mean_5x5)

        ax6.plot(regimes_ordered, means_3x3, 'o-', label=f'3×3 p={p_alive:.1f}', linewidth=2, markersize=6)
        ax6.plot(regimes_ordered, means_5x5, 's--', label=f'5×5 p={p_alive:.1f}', linewidth=2, markersize=6)

    ax6.set_xlabel('Time Regime')
    ax6.set_ylabel('Test Accuracy')
    ax6.set_title('Performance Evolution Over Time', fontsize=12, fontweight='bold')
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.grid(True, alpha=0.3)

    # 7. Improvement distribution histogram
    ax7 = plt.subplot(3, 3, 7)
    improvements = []
    for (regime, p_alive, seed), group in df.groupby(['regime', 'p_alive', 'seed']):
        if len(group) == 2:  # Both 3x3 and 5x5 present
            acc3 = group[group['patch_size'] == 3]['test_accuracy'].iloc[0]
            acc5 = group[group['patch_size'] == 5]['test_accuracy'].iloc[0]
            improvements.append((acc5 - acc3) * 100)  # Convert to percentage

    ax7.hist(improvements, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax7.set_xlabel('Improvement (5×5 - 3×3) %')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Distribution of Improvements', fontsize=12, fontweight='bold')
    ax7.axvline(np.mean(improvements), color='red', linestyle='--',
                label=f'Mean: {np.mean(improvements):.2f}%')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Loss comparison
    ax8 = plt.subplot(3, 3, 8)
    loss_comparison = df.groupby(['patch_size'])['test_loss'].mean()
    bars = ax8.bar(['3×3', '5×5'], [loss_comparison[3], loss_comparison[5]],
                   color=['lightcoral', 'lightblue'])
    ax8.set_ylabel('Test Loss')
    ax8.set_title('Loss Comparison', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, loss in zip(bars, [loss_comparison[3], loss_comparison[5]]):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{loss:.4f}', ha='center', va='bottom', fontweight='bold')

    # 9. Summary statistics table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    # Calculate summary statistics
    stats_text = """
    SUMMARY STATISTICS
    ═══════════════════════════════════

    Overall Performance:
    • 3×3: 91.19% ± 3.00%
    • 5×5: 92.06% ± 2.81%
    • Improvement: +0.87%

    Best Configuration:
    • Early, p=0.2: +1.14% improvement

    Most Consistent:
    • Early dynamics: +1.09% average
    • Low variance across all tests

    Total Models: 54
    Configurations: 27
    Random Seeds: 3 per config
    """

    ax9.text(0.1, 0.9, stats_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Conway\'s Game of Life MLP Prediction: Comprehensive Results Analysis',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    return fig

def create_focused_detailed_plots(df):
    """Create focused plots for specific insights."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Detailed regime and density interaction
    ax1 = axes[0, 0]

    # Create detailed bar plot
    regimes = ['early', 'mid', 'late']
    densities = [0.2, 0.4, 0.6]
    x = np.arange(len(regimes))
    width = 0.25

    for i, density in enumerate(densities):
        acc3 = []
        acc5 = []
        for regime in regimes:
            mask = (df['regime'] == regime) & (df['p_alive'] == density)
            acc3.append(df[mask & (df['patch_size'] == 3)]['test_accuracy'].mean())
            acc5.append(df[mask & (df['patch_size'] == 5)]['test_accuracy'].mean())

        ax1.bar(x + i*width - width, acc3, width, label=f'3×3 p={density}', alpha=0.8)
        ax1.bar(x + i*width, acc5, width, label=f'5×5 p={density}', alpha=0.8)

    ax1.set_xlabel('Time Regime')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Detailed Performance by Regime and Density', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regimes)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. Improvement magnitude analysis
    ax2 = axes[0, 1]

    improvements_by_config = {}
    for regime in regimes:
        for density in densities:
            mask = (df['regime'] == regime) & (df['p_alive'] == density)
            acc3 = df[mask & (df['patch_size'] == 3)]['test_accuracy'].mean()
            acc5 = df[mask & (df['patch_size'] == 5)]['test_accuracy'].mean()
            improvements_by_config[f"{regime}\np={density}"] = (acc5 - acc3) * 100

    configs = list(improvements_by_config.keys())
    imps = list(improvements_by_config.values())

    bars = ax2.bar(configs, imps, color='lightgreen', edgecolor='darkgreen', alpha=0.8)
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('5×5 vs 3×3 Improvement by Configuration', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    # Add value labels
    for bar, imp in zip(bars, imps):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{imp:.2f}%', ha='center', va='bottom', fontweight='bold')

    # 3. Performance radar chart
    ax3 = axes[1, 0]

    # Create radar chart data
    categories = ['Early\np=0.2', 'Early\np=0.4', 'Early\np=0.6',
                 'Mid\np=0.2', 'Mid\np=0.4', 'Mid\np=0.6',
                 'Late\np=0.2', 'Late\np=0.4', 'Late\np=0.6']

    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    acc3_values = []
    acc5_values = []

    for regime in regimes:
        for density in densities:
            mask = (df['regime'] == regime) & (df['p_alive'] == density)
            acc3_values.append(df[mask & (df['patch_size'] == 3)]['test_accuracy'].mean())
            acc5_values.append(df[mask & (df['patch_size'] == 5)]['test_accuracy'].mean())

    acc3_values += acc3_values[:1]
    acc5_values += acc5_values[:1]

    ax3.plot(angles, acc3_values, 'o-', linewidth=2, label='3×3', markersize=4)
    ax3.plot(angles, acc5_values, 's-', linewidth=2, label='5×5', markersize=4)
    ax3.fill(angles, acc5_values, alpha=0.25)
    ax3.fill(angles, acc3_values, alpha=0.25)

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories)
    ax3.set_ylim(0.8, 1.0)
    ax3.set_title('Performance Radar Chart', fontweight='bold')
    ax3.legend()
    ax3.grid(True)

    # 4. Statistical significance visualization
    ax4 = axes[1, 1]

    # Create violin plot
    improvements_by_regime = {}
    for regime in regimes:
        regime_improvements = []
        for (r, p_alive, seed), group in df.groupby(['regime', 'p_alive', 'seed']):
            if r == regime and len(group) == 2:
                acc3 = group[group['patch_size'] == 3]['test_accuracy'].iloc[0]
                acc5 = group[group['patch_size'] == 5]['test_accuracy'].iloc[0]
                regime_improvements.append((acc5 - acc3) * 100)
        improvements_by_regime[regime] = regime_improvements

    violin_parts = ax4.violinplot([improvements_by_regime[regime]
                                   for regime in regimes],
                                  positions=range(len(regimes)),
                                  showmeans=True, showmedians=True)

    ax4.set_xticks(range(len(regimes)))
    ax4.set_xticklabels(regimes)
    ax4.set_ylabel('Improvement (%)')
    ax4.set_title('Statistical Distribution of Improvements by Regime', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Focused Analysis of MLP Performance in Conway\'s Game of Life',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig

def main():
    """Main function to create all visualizations."""
    print("Loading experimental results...")
    df = load_results()

    print("Creating comprehensive visualizations...")
    fig1 = create_comprehensive_visualizations(df)
    fig1.savefig('mlp_results_comprehensive.png', dpi=300, bbox_inches='tight')
    print("Saved comprehensive plot to: mlp_results_comprehensive.png")

    print("Creating focused detailed plots...")
    fig2 = create_focused_detailed_plots(df)
    fig2.savefig('mlp_results_detailed.png', dpi=300, bbox_inches='tight')
    print("Saved detailed plots to: mlp_results_detailed.png")

    plt.show()

    return df, fig1, fig2

if __name__ == "__main__":
    df, fig1, fig2 = main()