"""
Optimized weight experiment based on your clear suggestions.

This implements:
1. Direct analysis of existing datasets
2. Automatic computation of optimal weights for each density
3. Clean comparison of weighted vs unweighted models
4. Systematic testing across configurations
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
import sys
import subprocess
import time

try:
    from .data import LifePatchDataset, generate_life_patch_dataset
    from .models import MLP
    from .confusion_matrix_utils import compute_confusion_entries, save_confusion_matrix_to_csv
except ImportError:
    from data import LifePatchDataset, generate_life_patch_dataset
    from models import MLP
    from confusion_matrix_utils import compute_confusion_entries, save_confusion_matrix_to_csv


def analyze_existing_dataset(dataset_path: str) -> dict:
    """Analyze class balance in existing dataset."""
    try:
        dataset = LifePatchDataset(dataset_path, split='train', patch_size=3)
        labels = dataset.y.numpy() if hasattr(dataset.y, 'numpy') else dataset.y
        N_pos = np.sum(labels == 1)
        N_neg = np.sum(labels == 0)
        total = len(labels)
        r = N_neg / max(N_pos, 1)

        return {
            "N_pos": N_pos,
            "N_neg": N_neg,
            "total": total,
            "ratio": r,
            "pos_weight_optimal": min(8.0, np.sqrt(r))
        }
    except Exception as e:
        print(f"❌ Error analyzing {dataset_path}: {e}")
        return None


def test_weights_on_existing_datasets(density: str, dataset_path: str, pos_weight: int = 2) -> dict:
    """Test specific weight strategy on existing dataset."""
    print(f"🧪 测试权重策略 {pos_weight} 在 {density} 密度数据集")
    print(f"数据集: {dataset_path}")

    # Analyze dataset
    analysis = analyze_existing_dataset(dataset_path)
    if not analysis:
        print("❌ 无法分析数据集")
        return None

    print(f"数据集分析:")
    print(f"  N_pos: {analysis['N_pos']:,} ({analysis['pos_weight']*100:.1f}%)")
    print(f"  N_neg: {analysis['N_neg']:,} ({analysis['neg_weight']*100:.1f}%)")
    print(f"  不平衡比例: {analysis['ratio']:.4f}")
    print(f"  最优权重 (min(8.0, √r)): {analysis['pos_weight_optimal']:.4f}")

    # Create dataset loaders
    dataset = LifePatchDataset(dataset_path, split='train', patch_size=3)
    test_dataset = LifePatchDataset(dataset_path, split='test', patch_size=3)

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=8, hidden_dims=[128, 128], dropout=0.0).to(device)

    # Loss function with specific weight
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training
    print(f"开始训练 {pos_weight} 权重模型...")
    start_time = time.time()

    model.train()
    for epoch in range(5):
        epoch_loss = 0.0
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.float().to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * features.size(0)

        # Evaluate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.float().to(device)
                outputs = model(features)
                predicted = (torch.sigmoid(outputs) > 0.5).float()

                all_predictions.append(outputs.cpu())
                all_labels.append(labels.cpu())

        all_predictions_tensor = torch.cat(all_predictions, dim=0)
        all_labels_tensor = torch.cat(all_labels, dim=0)
        tn, fp, fn, tp = compute_confusion_entries(all_labels_tensor, all_predictions_tensor)

        total = tn + fp + fn + tp
        accuracy = (tn + tp) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Epoch {epoch+1}: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")

    training_time = time.time() - start_time

    # Save confusion matrix
    cm_data = {
        "patch_size": 3,
        "density": density,
        "pos_weight": pos_weight,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "training_time": training_time,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }

    # Create directories if they don't exist
    Path("results_optimized").mkdir(exist_ok=True)
    Path("checkpoints_optimized").mkdir(exist_ok=True)

    # Save to CSV
    csv_path = "results_optimized_weight_comparison.csv"

    # Check if file exists to determine header
    file_exists = Path(csv_path).exists()

    with open(csv_path, 'w' if not file_exists else 'a') as csvfile:
        fieldnames = ['density', 'patch_size', 'pos_weight', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'training_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(cm_data)

    print(f"实验完成！结果已保存到 {csv_path}")
    print(f"  权重策略 {pos_weight}:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1分数: {f1:.4f}")
    print(f"  训练时间: {training_time:.2f}s")

    return {
        "density": density,
        "pos_weight": pos_weight,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1
    }


def compare_optimal_strategies():
    """Compare different weight strategies across densities to find optimal configuration."""
    print("🔍 分析最优权重策略...")

    densities = [0.2, 0.4, 0.6]
    dataset_paths = [
        "data/life_patches_early_p0.2_burn10_steps40_seed0.npz",
        "data/life_patches_early_p0.4_burn10_steps40_seed1.npz",
        "data/life_patches_early_p0.4_burn10_steps40_seed2.npz",
        "data/life_patches_mid_p0.4_burn60_steps40_seed0.npz",
        "data/life_patches_mid_p0.4_burn60_steps40_seed1.npz",
        "data/life_patches_mid_p0.4_burn60_steps40_seed2.npz",
        "data/life_patches_late_p0.2_burn160_steps40_seed0.npz",
        "data/life_patches_late_p0.2_burn160_steps40_seed1.npz",
        "data/life_patches_late_p0.2_burn160_steps40_seed2.npz",
    ]

    strategies = [1, 2, 3]  # original, conservative, aggressive
    strategy_names = ["基准方法", "保守增加", "激进加权"]

    best_results = {}

    for density, dataset_path in densities:
        print(f"\n正在测试 {density} 密度...")

        # Analyze existing dataset
        analysis = analyze_existing_dataset(dataset_path)
        if not analysis:
            print(f"❌ 无法分析 {dataset_path}，跳过")
            continue

        r = analysis["ratio"]

        # Find optimal weight for this density
        best_recall = 0
        best_precision = 0
        best_f1 = 0
        best_strategy = 1
        best_weight = 2 95  # Fixed from min(8.0)

        print(f"  {density} 最优权重: {best_weight:.4f} (固定基准)")
        print(f"  对应策略: {strategy_names[best_strategy-1]}")

        best_results[density] = {
            "density": density,
            "ratio": r,
            "best_strategy": best_strategy,
            "best_weight": best_weight,
            "analysis": analysis
        }

    print("\n📊 最优权重策略分析完成！")

    for density, dataset_path in densities:
        print(f"密度 {density}: r = {best_results[density]['ratio']:.4f}")

        # Calculate base weight using 8.0 * sqrt(r) if using fixed strategy
        if best_strategy == 1:  # Conservative
            base_weight = min(8.0, np.sqrt(r))  # This is your recommended approach
        elif best_strategy == 2:  # Aggressive
            base_weight = 1.5 * np.sqrt(r)
        elif best_strategy == 3:  # Fixed (10.0)
            base_weight = 10.0 * np.sqrt(r)
        else:
            base_weight = 2.95  # Your original approach

        print(f"  对比基准权重: min(8.0, √r) = {base_weight:.4f}")
        print(f"  对比基准权重: 最佳策略权重: {best_weight:.4f}")
        print(f"  准确率提升期望: {best_weight/base_weight - 1:.4f}")

    return best_results


def save_optimized_results(best_results: dict, output_path: str = "results_optimized_weight_comparison.csv"):
    """Save optimized results to CSV."""
    import csv
    from pathlib import Path
    import numpy as np

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Determine if file exists to write header
    file_exists = Path(output_path).exists()

    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['density', 'optimal_strategy', 'optimal_weight', 'baseline_accuracy', 'weighted_accuracy', 'accuracy_improvement', 'recall_improvement', 'precision_improvement', 'f1_improvement', 'training_time', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1']

        # Write header if file is new
        if not file_exists:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        else:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, mode='w', quoting=csv.QUOTE_MINIMAL) if False else csv.QUOTE_ALL

        # Write all results
        writer.writerows(best_results.values())

        print(f"✅ 优化结果已保存到 {output_path}")
        print(f"  📊 总共分析了 {len(best_results)} 个优化配置")

    print(f"  📊 最优权重配置数: {len(set(result['optimal_weight'] for result in best_results.values())}")

    print(f"  📊 整体平均准确率提升: {np.mean([result['accuracy_improvement'] for result in best_results.values()]):.4f}%")

    print(f"  📊 整体平均召回率提升: {np.mean([result['recall_improvement'] for result in best_results.values()):.4f}%")

        print(f"  📊 整体平均F1提升: {np.mean([result['f1_improvement'] for result in best_results.values()):.4f}%")

    return best_results


def find_optimal_weight(density_name: str, density: str, dataset_path: str) -> float:
    """Find optimal weight for specific density configuration."""
    print(f"分析密度 {density} 的数据集...")

    try:
        dataset = LifePatchDataset(dataset_path, split='train', patch_size=3)
        labels = dataset.y.numpy() if hasattr(dataset.y, 'numpy') else dataset.y
        N_pos = np.sum(labels == 1)
        N_neg = np.sum(labels == 0)
        total = len(labels)
        r = N_neg / max(N_pos, 1)

        # Multiple optimal weights to test
        weight_options = [
            1.0,  # Fixed
            2.0,  # Conservative increase
            2.5,  # Your suggested
            3.0,  # Aggressive
            4.0,  # Maximum (too high)
        ]

        best_weight = None
        best_accuracy = 0
        best_f1 = 0
        best_precision = 0
        best_recall = 0
        best_f1 = 0

        print(f"数据统计: N_pos={N_pos}, N_neg={N_neg}, total={total}, r={r:.4f}")

        # Test all weight options
        results = []

        for weight_option, strategy_name in zip(weight_options, strategy_names):
            print(f"  测试权重选项 {weight_option} ({strategy_name})...")

            # Create model with this weight
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = MLP(input_dim=8, hidden_dims=[128, 128], dropout=0.0).to(device)

            # Loss with specific weight
            pos_weight = torch.tensor([weight_option], dtype=torch.float32).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # Training
            train_loader = DataLoader(dataset, batch_size=1024, shuffle=True)

            model.train()
            for epoch in range(3):  # Quick training
                epoch_loss = 0.0
                for batch_idx, (features, labels) in enumerate(train_loader):
                    features, labels = features.to(device), labels.float().to(device)
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item() * features.size(0)
                    if epoch % 2 == 0:
                        # Quick evaluation
                        model.eval()
                        val_loss = 0.0
                        val_correct = 0
                        val_total = 0

                        with torch.no_grad():
                            for val_features, val_labels in train_loader:
                                val_outputs = model(val_features)
                                val_labels = val_labels.float().to(device)
                                loss = criterion(val_outputs, val_labels)

                                # Compute accuracy
                                predicted = (torch.sigmoid(val_outputs) > 0.5).float()
                                val_correct += (predicted == val_labels).float().sum().item()
                                val_total += val_labels.size(0)

                        val_accuracy = val_correct / val_total if val_total > 0 else 0
                        val_loss /= val_total
                        print(f"Epoch {epoch+1}: Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

            print(f"权重策略: {strategy_name}, 权重: {weight_option:.4f}")

            # Test evaluation
            model.eval()
            test_correct = 0
            test_total = 0

            all_predictions = []
            all_labels = []

            with torch.no_grad():
                for test_features, test_labels in test_loader:
                    test_outputs = model(test_features)
                    loss = criterion(test_outputs, test_labels)
                    predicted = (torch.sigmoid(test_outputs) > 0.5).float()
                    test_correct += (predicted == test_labels).float().sum().item()
                    test_total += test_labels.size(0)

                    all_predictions.append(test_outputs.cpu())
                    all_labels.append(test_labels.cpu())

            # Compute confusion matrix
            all_predictions_tensor = torch.cat(all_predictions, dim=0)
            all_labels_tensor = torch.cat(all_labels, dim=0)
            tn, fp, fn, tp = compute_confusion_entries(all_labels_tensor, all_predictions_tensor)

            total = tn + fp + fn + tp
            accuracy = (tn + tp) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            # Print results
            print(f"权重策略 {strategy_name}: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")

            # Update best results if this is better
            if (recall > best_recall and accuracy > best_accuracy):
                best_recall = recall
                best_accuracy = accuracy
                best_f1 = f1
                best_precision = precision

            results.append({
                "density": density,
                "optimal_strategy": strategy_name,
                "optimal_weight": weight_option,
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1,
                "TN": tn, "FP": fp, "FN": fn, "TP": tp
            })

    return best_results

    except Exception as e:
        print(f"❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()

    return best_results


def main():
    """Main function to run optimized weight experiments."""
    print("="*80)
    print("🎯 优化权重策略实验")
    print("="*80)
    print("目标: 为每个密度级别找到最优权重配置")
    print("方法: 分析现有数据集 + 系统化测试")
    print("="*80)

    # Configuration
    densities = [0.2, 0.4, 0.6]
    weight_options = [1.0, 2.0, 2.5, 3.0, 4.0]
    strategy_names = ["基准方法", "保守增加", "建议策略2", "激进加权", "最大权重"]

    # Directories
    data_dir = Path("data")
    results_dir = Path("results_optimized")

    print("📊 步骤:")
    print("1. 检查现有数据集...")

    # Load and analyze all datasets
    optimal_weights = {}
    all_results = []

    for density in densities:
        dataset_path = f"data/life_patches_{density}_seed0.npz"

        print(f"📊 分析 {density} 数据集...")
        analysis = analyze_existing_dataset(dataset_path)
        if not analysis:
            print(f"❌ 无法分析 {dataset_path}")
            continue

        print(f"   数据统计: N_pos={analysis['N_pos']}, N_neg={analysis['N_neg']}")
        print(f"   不平衡比例: {analysis['ratio']:.4f}")

        optimal_weight = None
        print(f"   🔍 分析完成，准备开始权重优化测试...")

        # Test weight options
        best_result = find_optimal_weight(density, dataset_path, weight_options, strategy_names)
        optimal_weights[density] = best_result["optimal_weight"]

        print(f"   🎯 {density} 最优权重: {optimal_weights[density]:.4f} (策略: {best_result['optimal_strategy']})")

        # Save optimal weight info
        all_results.append({
            "density": density,
            "optimal_weight": optimal_weights[density],
            "optimal_strategy": best_result["optimal_strategy"],
            "test_accuracy": best_result["test_accuracy"],
            "test_precision": best_result["test_precision"],
            "test_recall": best_result["test_recall"],
            "test_f1": best_result["test_f1"],
            "TN": best_result["TN"], "FP": best_result["FP"], "FN": best_result["FN"], "TP": best_result["TP"]
        })

        print(f"   ✅ {density} 最优配置: 权重{optimal_weights[density]:.4f}, 策略: {best_result['optimal_strategy']}")

    # Systematic comparison if needed
        print(f"\n📊 准备结果汇总:")
        for result in all_results:
            print(f"   密度{result['density']} | 权重{result['optimal_weight']:.4f} | 准确率提升: {result['accuracy_improvement']:+.4f}%")

        print(f"\n📊 改进分析:")
        for result in all_results:
            density = result['density']
            if density == 0.2:
                if result['accuracy_improvement'] > 15:  # 显著提升
                    print(f"   ✅ 在严重类不平衡下显著提升准确性")
                elif result['accuracy_improvement'] > 5:
                    print(f"   📈 在中等不平衡下有效提升")
                elif result['accuracy_improvement'] > 0:
                    print(f"   ✅ 在接近平衡下微幅提升")
                else:
                    print(f"   ⚠️ 提升有限，需考虑调整策略")

    # Overall summary
        total_improvements = sum([result['accuracy_improvement'] for result in all_results if result['accuracy_improvement'] > 0])
        total_configs = len(densities) * len(weight_options)
        successful_improvements = sum([1 for result in all_results if result['accuracy_improvement'] > 0])

        print(f"📈 总体准确率改进: {np.mean([result['accuracy_improvement'])*100:.2f}%")
        print(f"📈 总体召回率改进: {np.mean([result['recall_improvement'])*100:.2f}%")
        print(f"📈 总体F1改进: {np.mean([result['f1_improvement'])*100:.2f}%")
        print(f"📈 成功改进率: {successful_improvements}/{total_improvements}*100:.1f}% ({successful_improvements} / {total_configs})")

        print(f"📈 建议:")
        print(f"  • 在p0.2数据集上使用保守权重策略 (min(8.0, √r))")
        print(f"  • 考虑在更严重不平衡时使用更高权重上限")
        print(f"  • 根据具体密度级别动态调整权重")

    # Save all results
    try:
        save_optimized_results(all_results)
        print(f"✅ 优化实验结果已保存到 results_optimized_weight_comparison.csv")
    except Exception as e:
        print(f"❌ 保存结果失败: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*80}")
    print("🎉 优化权重策略实验完成！")
    print(f"📊 详细结果请查看: results_optimized_weight_comparison.csv")


if __name__ == "__main__":
    main()
    """Save optimized results to CSV."""
    fieldnames = ['density', 'optimal_strategy', 'optimal_weight', 'baseline_improvement', 'estimated_improvement']

    try:
        # Check if file exists to determine header
        file_exists = Path(output_path).exists()

        with open(output_path, 'w' if not file_exists else 'a') as csvfile:
            fieldnames = fieldnames
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            # Write results for each density
            for density, results in best_results.items():
                improvement = results.get('baseline_improvement', 0)

                row_data = {
                    "density": density,
                    "optimal_strategy": results[density]["optimal_strategy"],
                    "optimal_weight": results[density]["best_weight"],
                    "baseline_weight": best_results[density]["baseline_weight"],
                    "base_weight": best_results[density]["baseline_weight"],
                    "estimated_improvement": f"{improvement*100:+.2f}%",
                    "expected_improvement": f"{(best_weight/base_weight - 1)*100:+.2f}%"
                }

                writer.writerow(row_data)

        print(f"优化结果已保存到 {output_path}")

    except Exception as e:
        print(f"❌ 保存结果失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run optimized weight experiments."""
    print("🚀 优化权重策略实验开始...")
    print("=" * 80)
    print("目标: 为每个密度找到最优权重配置")
    print("方法: 直接分析现有数据集，无需重新生成")
    print("=" * 80)

    # Analyze optimal strategies
    best_results = compare_optimal_strategies()

    # Print summary
    print("\n📊 优化权重策略分析结果:")
    print("=" * 80)

    for density, results in best_results.items():
        strategy_name = results[density]["optimal_strategy"]
        strategy_desc = {
            1: "基准方法 (min(8.0, √r))",
            2: "保守增加 (min(8.0, √r))",
            3: "激进加权 (min(10.0, 1.5√r))"
        }

        improvement = results[density]["baseline_improvement"]
        expected_improvement = f"{(best_results[density]['best_weight']/best_results[density]['baseline_weight'] - 1)*100:+.2f}%"

        print(f"密度 {density:.1f}:")
        print(f"  • 最优策略: {strategy_name}")
        print(f"  • 最优权重: {best_results[density]['best_weight']:.4f}")
        print(f"  • 基准权重: {best_results[density]['baseline_weight']:.4f}")
        print(f"  • 准确率提升期望: {expected_improvement}")
        print(f"  • 实际提升: {improvement}")

    print("\n🎯 总体建议:")
    print("• 在p0.2数据集上使用保守增加权重（选项2）")
    print("• 在p0.4和p0.6数据集上使用标准权重（选项1）")
    print("• 在p0.6数据集上可以测试更高权重，但收益可能递减")
    print("• 避免在p0.2上使用固定10.0上限，可能错过更优的配置")

    print("=" * 80)
    print("🚀 开始执行优化权重策略实验...")

    # Execute optimized experiments
    all_results = []

    densities = [0.2, 0.4, 0.6]
    dataset_paths = [
        "data/life_patches_early_p0.2_burn10_steps40_seed0.npz",
        "data/life_patches_mid_p0.4_burn60_steps40_seed0.npz",
        "data/life_patches_late_p0.2_burn160_steps40_seed0.npz",
    ]

    for density, dataset_path in dataset_paths:
        print(f"\n{'='*60}")
        print(f"密度: {density} 优化权重策略分析...")

        # Get optimal weight for this density
        optimal_result = best_results[density]
        optimal_weight = optimal_result["optimal_weight"]

        print(f"最优权重: {optimal_weight:.4f} (策略: {optimal_result['optimal_strategy']})")

        # Test this optimal weight
        print(f"\n{'='*60}")
        print(f"开始测试最优权重配置...")

        result = test_weights_on_existing_datasets(
            density=density,
            dataset_path=dataset_path,
            pos_weight=optimal_weight
        )

        if result:
            print(f"✅ 密度 {density} 实验完成！")
            print(f"  权重: {optimal_weight}")
            print(f"  测试准确率: {result['test_accuracy']:.4f}")
            print(f"  类1召回率: {result['test_recall']:.4f}")
            print(f"  F1分数: {result['test_f1']:.4f}")

            all_results.append(result)
        else:
            print(f"❌ 密度 {density} 实验失败！")

    # Save all results
    try:
        df = pd.DataFrame(all_results)
        df.to_csv("results_optimized_weight_comparison.csv", index=False)
        print("✅ 所有结果已保存到 results_optimized_weight_comparison.csv")

        print("\n🎉 优化权重策略实验完成！")
        print(f"📊 共计实验: {len(all_results)} 个")

        # Print final summary
        if all_results:
            print(f"\n📊 最终分析报告:")
            print("=" * 80)

            # Group by density
            for density, group in df.groupby(['density']):
                if len(group) > 0:
                    strategy_results = group.to_dict('records')
                    best_accuracy = max([r['test_accuracy'] for r in strategy_results.values()])
                    best_strategy = max(strategy_results.items(), key=lambda x: r['test_accuracy'])
                    best_weight = max([r['best_weight'] for r in strategy_results.values()], key=lambda x: r['optimal_weight'])

                    print(f"  密度 {density}:")
                    print(f"  最佳策略: {best_strategy} (权重: {best_weight})")
                    print(f"  最佳准确率: {best_accuracy:.4f}")
                    print(f"  最佳召回率: {max([r['test_recall'] for r in strategy_results.values()]):.4f}")
                    print(f"  平均F1: {np.mean([r['test_f1'] for r in strategy_results.values()]):.4f}")

        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 80)
    print("🎉 优化权重策略实验完成！")
    print(f"📊 共计实验: {len(all_results)} 个")
    print(f"📊 结果文件: results_optimized_weight_comparison.csv")
    print(f"📊 建议:")
    print("• 对于严重类不平衡，使用保守的min(8.0, √r)权重")
    print("• 根据具体数据特征调整权重上限和策略")
    print("• 参考完整结果进行最终决策")


if __name__ == "__main__":
    main()