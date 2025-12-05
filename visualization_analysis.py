import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better presentation
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
fig_size = (12, 8)
dpi = 300

def load_and_analyze_data():
    """Load the results data and perform basic analysis"""
    df = pd.read_csv('results_mlp_grid.csv')

    print("=== 数据概览 ===")
    print(f"总实验数量: {len(df)}")
    print(f"实验配置: {df['regime'].unique()}")
    print(f"存活率: {sorted(df['p_alive'].unique())}")
    print(f"邻域大小: {sorted(df['patch_size'].unique())}")
    print(f"随机种子: {sorted(df['seed'].unique())}")

    return df

def create_performance_comparison(df):
    """Create performance comparison between 3x3 and 5x5 neighborhoods"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Overall accuracy comparison
    ax1 = axes[0, 0]
    accuracy_by_patch = df.groupby('patch_size')['test_accuracy'].agg(['mean', 'std']).reset_index()
    bars = ax1.bar(['3×3', '5×5'], accuracy_by_patch['mean'],
                   yerr=accuracy_by_patch['std'], capsize=5, alpha=0.7)
    ax1.set_ylabel('测试准确率', fontsize=12)
    ax1.set_title('整体性能对比: 3×3 vs 5×5 邻域', fontsize=14, fontweight='bold')
    ax1.set_ylim(0.85, 0.96)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracy_by_patch['mean']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 2. Performance by regime
    ax2 = axes[0, 1]
    regime_performance = df.groupby(['regime', 'patch_size'])['test_accuracy'].mean().unstack()
    regimes = ['early', 'mid', 'late']
    regime_labels = ['早期 (10步)', '中期 (60步)', '晚期 (160步)']

    x = np.arange(len(regimes))
    width = 0.35

    bars1 = ax2.bar(x - width/2, regime_performance.loc[regimes, 3], width,
                    label='3×3', alpha=0.7)
    bars2 = ax2.bar(x + width/2, regime_performance.loc[regimes, 5], width,
                    label='5×5', alpha=0.7)

    ax2.set_xlabel('时间阶段', fontsize=12)
    ax2.set_ylabel('测试准确率', fontsize=12)
    ax2.set_title('不同时间阶段的性能对比', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(regime_labels)
    ax2.legend()
    ax2.set_ylim(0.85, 0.96)

    # Add improvement annotations
    for i, regime in enumerate(regimes):
        improvement = regime_performance.loc[regime, 5] - regime_performance.loc[regime, 3]
        ax2.annotate(f'+{improvement:.4f}',
                    xy=(i + width/2, regime_performance.loc[regime, 5] + 0.003),
                    ha='center', fontsize=10, color='green', fontweight='bold')

    # 3. Performance by initial density
    ax3 = axes[1, 0]
    density_performance = df.groupby(['p_alive', 'patch_size'])['test_accuracy'].mean().unstack()
    densities = sorted(df['p_alive'].unique())

    x = np.arange(len(densities))
    bars1 = ax3.bar(x - width/2, density_performance.loc[densities, 3], width,
                    label='3×3', alpha=0.7)
    bars2 = ax3.bar(x + width/2, density_performance.loc[densities, 5], width,
                    label='5×5', alpha=0.7)

    ax3.set_xlabel('初始存活率', fontsize=12)
    ax3.set_ylabel('测试准确率', fontsize=12)
    ax3.set_title('不同初始密度的性能对比', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{d:.1f}' for d in densities])
    ax3.legend()
    ax3.set_ylim(0.85, 0.96)

    # Add improvement annotations
    for i, density in enumerate(densities):
        improvement = density_performance.loc[density, 5] - density_performance.loc[density, 3]
        ax3.annotate(f'+{improvement:.4f}',
                    xy=(i + width/2, density_performance.loc[density, 5] + 0.003),
                    ha='center', fontsize=10, color='green', fontweight='bold')

    # 4. Performance improvement heatmap
    ax4 = axes[1, 1]
    improvement_data = []
    for regime in regimes:
        for density in densities:
            acc_3x3 = df[(df['regime'] == regime) & (df['patch_size'] == 3) & (df['p_alive'] == density)]['test_accuracy'].mean()
            acc_5x5 = df[(df['regime'] == regime) & (df['patch_size'] == 5) & (df['p_alive'] == density)]['test_accuracy'].mean()
            improvement = acc_5x5 - acc_3x3
            improvement_data.append(improvement)

    improvement_matrix = np.array(improvement_data).reshape(3, 3)

    im = ax4.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=0.02)
    ax4.set_xticks(range(len(densities)))
    ax4.set_xticklabels([f'{d:.1f}' for d in densities])
    ax4.set_yticks(range(len(regimes)))
    ax4.set_yticklabels(regime_labels)
    ax4.set_xlabel('初始存活率', fontsize=12)
    ax4.set_ylabel('时间阶段', fontsize=12)
    ax4.set_title('5×5 相对 3×3 的性能提升', fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(len(regimes)):
        for j in range(len(densities)):
            text = ax4.text(j, i, f'+{improvement_matrix[i, j]:.4f}',
                           ha="center", va="center", color="black", fontweight='bold')

    plt.colorbar(im, ax=ax4, label='准确率提升')

    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=dpi, bbox_inches='tight')
    plt.show()

    return regime_performance, density_performance

def create_detailed_analysis(df):
    """Create detailed analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Loss vs Accuracy scatter plot
    ax1 = axes[0, 0]
    colors = {3: 'blue', 5: 'red'}
    labels = {3: '3×3', 5: '5×5'}

    for patch_size in [3, 5]:
        subset = df[df['patch_size'] == patch_size]
        ax1.scatter(subset['test_loss'], subset['test_accuracy'],
                   c=colors[patch_size], label=labels[patch_size], alpha=0.6, s=50)

    ax1.set_xlabel('测试损失', fontsize=12)
    ax1.set_ylabel('测试准确率', fontsize=12)
    ax1.set_title('损失 vs 准确率关系', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Box plot of accuracy by regime and patch size
    ax2 = axes[0, 1]
    df_melted = df.melt(id_vars=['regime', 'patch_size'],
                       value_vars=['test_accuracy'],
                       var_name='metric', value_name='value')

    sns.boxplot(data=df_melted, x='regime', y='value', hue='patch_size', ax=ax2)
    ax2.set_xlabel('时间阶段', fontsize=12)
    ax2.set_ylabel('测试准确率', fontsize=12)
    ax2.set_title('准确率分布箱线图', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(['早期 (10步)', '中期 (60步)', '晚期 (160步)'])
    ax2.legend(title='邻域大小')

    # 3. Convergence analysis (train vs test accuracy)
    ax3 = axes[1, 0]
    for patch_size in [3, 5]:
        subset = df[df['patch_size'] == patch_size]
        ax3.scatter(subset['train_accuracy'], subset['test_accuracy'],
                   c=colors[patch_size], label=labels[patch_size], alpha=0.6, s=50)

    # Add diagonal line
    min_acc = min(df['train_accuracy'].min(), df['test_accuracy'].min())
    max_acc = max(df['train_accuracy'].max(), df['test_accuracy'].max())
    ax3.plot([min_acc, max_acc], [min_acc, max_acc], 'k--', alpha=0.5, label='完美泛化线')

    ax3.set_xlabel('训练准确率', fontsize=12)
    ax3.set_ylabel('测试准确率', fontsize=12)
    ax3.set_title('训练 vs 测试准确率 (泛化能力)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Performance variance analysis
    ax4 = axes[1, 1]
    variance_data = []
    labels_list = []

    for regime in ['early', 'mid', 'late']:
        for density in [0.2, 0.4, 0.6]:
            for patch_size in [3, 5]:
                subset = df[(df['regime'] == regime) &
                          (df['p_alive'] == density) &
                          (df['patch_size'] == patch_size)]
                variance = subset['test_accuracy'].var()
                variance_data.append(variance)
                labels_list.append(f'{regime[:1].upper()}{density:.1f}_{patch_size}×{patch_size}')

    bars = ax4.bar(range(len(variance_data)), variance_data, alpha=0.7)
    ax4.set_xlabel('实验配置', fontsize=12)
    ax4.set_ylabel('准确率方差', fontsize=12)
    ax4.set_title('不同配置下的性能稳定性', fontsize=14, fontweight='bold')
    ax4.set_xticks(range(0, len(variance_data), 2))
    ax4.set_xticklabels([labels_list[i] for i in range(0, len(labels_list), 2)],
                       rotation=45, ha='right')

    # Color code by regime
    colors = ['red', 'green', 'blue']
    for i, bar in enumerate(bars):
        regime_idx = i // 6  # 6 configurations per regime
        bar.set_color(colors[regime_idx])
        bar.set_alpha(0.7)

    plt.tight_layout()
    plt.savefig('detailed_analysis.png', dpi=dpi, bbox_inches='tight')
    plt.show()

def generate_summary_statistics(df):
    """Generate summary statistics and insights"""
    print("\n" + "="*60)
    print("详细统计分析")
    print("="*60)

    # Overall statistics
    print("\n1. 整体性能统计:")
    overall_stats = df.groupby('patch_size')['test_accuracy'].describe()
    print(overall_stats)

    # Performance improvement
    print("\n2. 性能提升分析:")
    improvement_stats = df.groupby(['regime', 'p_alive']).apply(
        lambda x: x[x['patch_size'] == 5]['test_accuracy'].mean() -
                 x[x['patch_size'] == 3]['test_accuracy'].mean()
    ).round(4)
    print("5×5 相对 3×3 的准确率提升:")
    for (regime, density), improvement in improvement_stats.items():
        print(f"  {regime} (密度 {density}): +{improvement:.4f}")

    print(f"\n平均提升: {improvement_stats.mean():.4f} ± {improvement_stats.std():.4f}")

    # Best and worst configurations
    print("\n3. 最佳和最差配置:")
    best_3x3 = df[df['patch_size'] == 3].loc[df[df['patch_size'] == 3]['test_accuracy'].idxmax()]
    best_5x5 = df[df['patch_size'] == 5].loc[df[df['patch_size'] == 5]['test_accuracy'].idxmax()]

    print("最佳 3×3 配置:")
    print(f"  时间阶段: {best_3x3['regime']}, 密度: {best_3x3['p_alive']}, 准确率: {best_3x3['test_accuracy']:.4f}")
    print("最佳 5×5 配置:")
    print(f"  时间阶段: {best_5x5['regime']}, 密度: {best_5x5['p_alive']}, 准率: {best_5x5['test_accuracy']:.4f}")

    # Statistical significance
    print("\n4. 统计显著性分析:")
    from scipy import stats

    improvements = []
    for regime in ['early', 'mid', 'late']:
        for density in [0.2, 0.4, 0.6]:
            acc_3x3 = df[(df['regime'] == regime) & (df['patch_size'] == 3) &
                        (df['p_alive'] == density)]['test_accuracy']
            acc_5x5 = df[(df['regime'] == regime) & (df['patch_size'] == 5) &
                        (df['p_alive'] == density)]['test_accuracy']

            t_stat, p_value = stats.ttest_ind(acc_5x5, acc_3x3)
            improvements.append((regime, density, acc_5x5.mean() - acc_3x3.mean(), p_value))

    significant_improvements = [imp for imp in improvements if imp[3] < 0.05]
    print(f"显著改进的配置数量 (p < 0.05): {len(significant_improvements)}/{len(improvements)}")

    return improvement_stats

def main():
    """Main analysis function"""
    print("Conway's Game of Life MLP 预测结果分析")
    print("="*60)

    # Load data
    df = load_and_analyze_data()

    # Create visualizations
    print("\n生成性能对比图表...")
    regime_performance, density_performance = create_performance_comparison(df)

    print("\n生成详细分析图表...")
    create_detailed_analysis(df)

    # Generate summary statistics
    improvement_stats = generate_summary_statistics(df)

    print("\n" + "="*60)
    print("图表已生成并保存:")
    print("- performance_comparison.png: 性能对比图表")
    print("- detailed_analysis.png: 详细分析图表")
    print("="*60)

    return df, improvement_stats

if __name__ == "__main__":
    df, improvement_stats = main()