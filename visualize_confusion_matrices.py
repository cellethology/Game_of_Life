"""
📊 混淆矩阵可视化
4张图: 3×3/5×5 patch × 无权重/有权重
每张图显示3×3网格: 3密度 × 3时间点的混淆矩阵
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

def load_results():
    """加载实验结果"""
    # 查找最新的结果文件
    result_files = list(Path('.').glob('results_18_settings_*.json'))
    if not result_files:
        print("❌ 没有找到结果文件!")
        return None

    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    print(f"📂 加载结果文件: {latest_file}")

    with open(latest_file, 'r') as f:
        results = json.load(f)

    return results

def aggregate_results_by_setting(results):
    """按设置聚合结果"""
    # 设置结构: {density}_{time_point}_{patch_size}_{seed}_{weight_type}
    aggregated = {}

    for setting_name, result_list in results.items():
        # 解析设置名称
        parts = setting_name.split('_')
        density = parts[0]  # p0.2, p0.4, p0.6
        time_point = parts[1]  # early, mid, late
        patch_size = parts[2]  # patch3, patch5

        # 按设置分组
        setting_key = f"{density}_{time_point}"
        if patch_size not in aggregated:
            aggregated[patch_size] = {}

        if setting_key not in aggregated[patch_size]:
            aggregated[patch_size][setting_key] = {'unweighted': [], 'weighted': []}

        # 按权重类型分组
        for result in result_list:
            if result['use_weight']:
                aggregated[patch_size][setting_key]['weighted'].append(result)
            else:
                aggregated[patch_size][setting_key]['unweighted'].append(result)

    return aggregated

def compute_confusion_matrix_matrix(results_list):
    """计算聚合的混淆矩阵"""
    if not results_list:
        return np.array([[0, 0], [0, 0]])

    total_tn = sum(r['tn'] for r in results_list)
    total_fp = sum(r['fp'] for r in results_list)
    total_fn = sum(r['fn'] for r in results_list)
    total_tp = sum(r['tp'] for r in results_list)

    return np.array([[total_tn, total_fp], [total_fn, total_tp]])

def compute_normalized_confusion_matrix(cm):
    """计算归一化的混淆矩阵"""
    return cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

def create_confusion_matrix_plot(aggregated_results, patch_size, use_weight):
    """创建单个混淆矩阵图 (3×3网格)"""

    # 密度和时间点顺序
    densities = ['p0.2', 'p0.4', 'p0.6']
    time_points = ['early', 'mid', 'late']

    # 创建3×3子图
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'{patch_size}×{patch_size} Patch - {"Weighted" if use_weight else "Unweighted"}',
                 fontsize=16, fontweight='bold')

    patch_key = f"patch{patch_size}"
    weight_type = "weighted" if use_weight else "unweighted"

    # 为每个子图创建混淆矩阵热力图
    for i, density in enumerate(densities):
        for j, time_point in enumerate(time_points):
            ax = axes[i, j]

            setting_key = f"{density}_{time_point}"

            if (patch_key in aggregated_results and
                setting_key in aggregated_results[patch_key] and
                aggregated_results[patch_key][setting_key][weight_type]):

                results_list = aggregated_results[patch_key][setting_key][weight_type]

                # 计算聚合混淆矩阵
                cm = compute_confusion_matrix_matrix(results_list)

                # 计算指标
                total = cm.sum()
                accuracy = (cm[0,0] + cm[1,1]) / total
                precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
                recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                # 获取权重信息
                avg_weight = np.mean([r['pos_weight'] for r in results_list]) if use_weight else 1.0

                # 创建热力图 (使用归一化值)
                cm_normalized = compute_normalized_confusion_matrix(cm)

                sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                           ax=ax, cbar=False, vmin=0, vmax=1,
                           annot_kws={'size': 10, 'weight': 'bold'})

                # 设置标题和标签
                title = f"{density.upper()} {time_point.upper()}\n"
                title += f"Acc: {accuracy:.3f}, Prec: {precision:.3f}, Rec: {recall:.3f}\n"
                if use_weight:
                    title += f"Weight: {avg_weight:.3f}"
                else:
                    title += f"Weight: 1.000"

                ax.set_title(title, fontsize=10, fontweight='bold')
                ax.set_xlabel('Predicted', fontsize=9)
                ax.set_ylabel('Actual', fontsize=9)

                # 设置刻度标签
                ax.set_xticklabels(['0 (Dead)', '1 (Alive)'], fontsize=8)
                ax.set_yticklabels(['0 (Dead)', '1 (Alive)'], fontsize=8)

            else:
                # 如果没有数据，显示空白
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)
                ax.set_title(f"{density.upper()} {time_point.upper()}", fontsize=10)
                ax.set_xlabel('Predicted', fontsize=9)
                ax.set_ylabel('Actual', fontsize=9)

    # 调整布局
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    # 添加全局颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='Normalized Rate')

    # 添加图例说明
    fig.text(0.5, 0.02,
             'Matrix Format: [[TN, FP], [FN, TP]] | Values: Absolute Counts | Colors: Normalized Rates\n'
             'Densities: p0.2 (Low), p0.4 (Medium), p0.6 (High) | Time: early, mid, late burn-in steps',
             ha='center', va='center', fontsize=10, style='italic')

    return fig

def create_summary_statistics(aggregated_results):
    """创建汇总统计图"""

    densities = ['p0.2', 'p0.4', 'p0.6']
    time_points = ['early', 'mid', 'late']

    # 创建指标对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Weight vs Unweighted Performance Comparison', fontsize=16, fontweight='bold')

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 2, idx % 2]

        x = np.arange(len(densities) * len(time_points))
        width = 0.35

        unweighted_scores = []
        weighted_scores = []
        x_labels = []

        for density in densities:
            for time_point in time_points:
                setting_key = f"{density}_{time_point}"

                # 收集3×3和5×5的结果
                all_unweighted = []
                all_weighted = []

                for patch_size in [3, 5]:
                    patch_key = f"patch{patch_size}"
                    if (patch_key in aggregated_results and
                        setting_key in aggregated_results[patch_key]):

                        # 无权重结果
                        if aggregated_results[patch_key][setting_key]['unweighted']:
                            unweighted_results = aggregated_results[patch_key][setting_key]['unweighted']
                            avg_score = np.mean([r[metric] for r in unweighted_results])
                            all_unweighted.append(avg_score)

                        # 有权重结果
                        if aggregated_results[patch_key][setting_key]['weighted']:
                            weighted_results = aggregated_results[patch_key][setting_key]['weighted']
                            avg_score = np.mean([r[metric] for r in weighted_results])
                            all_weighted.append(avg_score)

                # 跨patch大小的平均
                unweighted_scores.append(np.mean(all_unweighted) if all_unweighted else 0)
                weighted_scores.append(np.mean(all_weighted) if all_weighted else 0)
                x_labels.append(f"{density}_{time_point}")

        # 绘制柱状图
        bars1 = ax.bar(x - width/2, unweighted_scores, width, label='Unweighted', alpha=0.8)
        bars2 = ax.bar(x + width/2, weighted_scores, width, label='Weighted', alpha=0.8)

        ax.set_xlabel('Setting (Density_Time)', fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f'{label} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 添加改善百分比
        for i, (uw, w) in enumerate(zip(unweighted_scores, weighted_scores)):
            if uw > 0:
                improvement = ((w - uw) / uw) * 100
                if improvement > 0:
                    ax.text(i, max(uw, w) + 0.01, f'+{improvement:.1f}%',
                            ha='center', va='bottom', fontsize=7, color='green')
                else:
                    ax.text(i, max(uw, w) + 0.01, f'{improvement:.1f}%',
                            ha='center', va='bottom', fontsize=7, color='red')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    return fig

def main():
    """主函数"""
    print("🎨 开始创建混淆矩阵可视化...")

    # 加载结果
    results = load_results()
    if not results:
        return

    # 聚合结果
    aggregated = aggregate_results_by_setting(results)

    print(f"📊 数据统计:")
    for patch_size, patch_data in aggregated.items():
        print(f"  {patch_size}: {len(patch_data)} settings")

    # 创建4张混淆矩阵图
    figures = {}

    for patch_size in [3, 5]:
        for use_weight in [False, True]:
            print(f"🎨 创建图: {patch_size}×{patch_size} {'Weighted' if use_weight else 'Unweighted'}")

            fig = create_confusion_matrix_plot(aggregated, patch_size, use_weight)
            filename = f"confusion_matrix_{patch_size}x{patch_size}_{'weighted' if use_weight else 'unweighted'}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            figures[filename] = fig

            print(f"  ✅ 保存到: {filename}")

    # 创建汇总统计图
    print(f"📈 创建汇总统计图...")
    summary_fig = create_summary_statistics(aggregated)
    summary_filename = "performance_comparison_summary.png"
    summary_fig.savefig(summary_filename, dpi=300, bbox_inches='tight')
    figures[summary_filename] = summary_fig
    print(f"  ✅ 保存到: {summary_filename}")

    # 显示所有图
    print(f"\n🖼️ 显示所有生成的图...")

    # 关闭所有图以节省内存
    for fig in figures.values():
        plt.close(fig)

    print(f"\n✅ 可视化完成!")
    print(f"📁 生成的文件:")
    for filename in figures.keys():
        print(f"  - {filename}")

    print(f"\n📋 图像说明:")
    print(f"  - confusion_matrix_3x3_unweighted.png: 3×3 patch 无权重混淆矩阵")
    print(f"  - confusion_matrix_3x3_weighted.png: 3×3 patch 有权重混淆矩阵")
    print(f"  - confusion_matrix_5x5_unweighted.png: 5×5 patch 无权重混淆矩阵")
    print(f"  - confusion_matrix_5x5_weighted.png: 5×5 patch 有权重混淆矩阵")
    print(f"  - performance_comparison_summary.png: 性能对比汇总")

if __name__ == "__main__":
    main()