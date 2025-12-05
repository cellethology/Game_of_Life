"""
🔬 CORRECT 18-SETTING EXPERIMENT
18个设置: 3密度 × 3时间点 × 2模型尺寸
每个设置跑3个种子，对比无权重 vs 有权重策略
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import json
from collections import defaultdict

# Import locally
import sys
sys.path.append('.')
from data import LifePatchDataset
from models import MLP
from confusion_matrix_utils import compute_confusion_entries

class Correct18Experiment:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_time = datetime.now()

        # 实验设计: 3密度 × 3时间点 × 2模型尺寸
        self.densities = ["p0.2", "p0.4", "p0.6"]
        self.time_points = ["early", "mid", "late"]  # burn-in: 10, 60, 160 steps
        self.patch_sizes = [3, 5]
        self.seeds = [0, 1, 2]

        # 数据集映射
        self.dataset_map = {
            "early": {
                "p0.2": "data/life_patches_early_p0.2_burn10_steps40_seed{seed}.npz",
                "p0.4": "data/life_patches_early_p0.4_burn10_steps40_seed{seed}.npz",
                "p0.6": "data/life_patches_early_p0.6_burn10_steps40_seed{seed}.npz"
            },
            "mid": {
                "p0.2": "data/life_patches_mid_p0.2_burn60_steps40_seed{seed}.npz",
                "p0.4": "data/life_patches_mid_p0.4_burn60_steps40_seed{seed}.npz",
                "p0.6": "data/life_patches_mid_p0.6_burn60_steps40_seed{seed}.npz"
            },
            "late": {
                "p0.2": "data/life_patches_late_p0.2_burn160_steps40_seed{seed}.npz",
                "p0.4": "data/life_patches_late_p0.4_burn160_steps40_seed{seed}.npz",
                "p0.6": "data/life_patches_late_p0.6_burn160_steps40_seed{seed}.npz"
            }
        }

        self.results = defaultdict(list)  # 存储每个设置的结果

        total_experiments = len(self.densities) * len(self.time_points) * len(self.patch_sizes) * len(self.seeds) * 2  # 2权重策略
        print(f"🔬 CORRECT 18-SETTING EXPERIMENT")
        print(f"Device: {self.device}")
        print(f"Densities: {self.densities}")
        print(f"Time points: {self.time_points}")
        print(f"Patch sizes: {self.patch_sizes}")
        print(f"Seeds: {self.seeds}")
        print(f"Total experiments: {total_experiments}")
        print(f"Started at: {self.start_time}")

    def compute_optimal_weight(self, dataset_path, patch_size):
        """计算给定数据集和patch size的最优权重"""
        try:
            dataset = LifePatchDataset(dataset_path, split='train', patch_size=patch_size)
            labels = dataset.y.numpy() if hasattr(dataset.y, 'numpy') else dataset.y
            N_pos = np.sum(labels == 1)
            N_neg = np.sum(labels == 0)
            r = N_neg / max(N_pos, 1)
            # 推荐策略: min(8.0, sqrt(r))
            optimal_weight = min(8.0, np.sqrt(r))
            return optimal_weight, N_pos, N_neg, r
        except Exception as e:
            print(f"Error computing weight for {dataset_path}: {e}")
            return 1.0, 0, 0, 1.0

    def train_and_evaluate(self, dataset_path, patch_size, use_weight, experiment_id):
        """训练并评估一个设置"""
        try:
            print(f"\n🧪 {experiment_id}: {'Weighted' if use_weight else 'Unweighted'}")
            print(f"Dataset: {dataset_path}, Patch: {patch_size}×{patch_size}")

            # 创建数据集
            train_dataset = LifePatchDataset(dataset_path, split='train', patch_size=patch_size)
            test_dataset = LifePatchDataset(dataset_path, split='test', patch_size=patch_size)

            train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

            # 计算权重
            if use_weight:
                pos_weight, N_pos, N_neg, r = self.compute_optimal_weight(dataset_path, patch_size)
                print(f"Class balance: N_pos={N_pos:,}, N_neg={N_neg:,}, r={r:.2f}")
                print(f"Computed weight: {pos_weight:.4f}")
            else:
                pos_weight = 1.0
                N_pos, N_neg, r = self.compute_optimal_weight(dataset_path, patch_size)[1:]
                print(f"Class balance: N_pos={N_pos:,}, N_neg={N_neg:,}, r={r:.2f}")
                print(f"Using weight: {pos_weight:.4f}")

            # 根据patch size调整模型
            input_dim = 8 if patch_size == 3 else 24  # 3×3=8特征, 5×5=24特征
            model = MLP(input_dim=input_dim, hidden_dims=[128, 128], dropout=0.0).to(self.device)

            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(self.device)
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # 训练
            model.train()
            best_val_loss = float('inf')
            patience_counter = 0
            max_patience = 5

            # 创建验证集
            val_size = len(train_dataset) // 4
            train_size = len(train_dataset) - val_size
            train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            val_loader = DataLoader(val_subset, batch_size=1024, shuffle=False)

            for epoch in range(20):  # 最多20个epoch
                # 训练
                model.train()
                train_loss = 0.0
                for features, labels in train_loader:
                    features, labels = features.to(self.device), labels.float().to(self.device)
                    outputs = model(features)
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                # 验证
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for features, labels in val_loader:
                        features, labels = features.to(self.device), labels.float().to(self.device)
                        outputs = model(features)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # 保存最佳模型状态
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1

                if patience_counter >= max_patience:
                    print(f"  Early stopping at epoch {epoch+1} (val loss: {avg_val_loss:.4f})")
                    break

                if epoch % 5 == 0:
                    print(f"  Epoch {epoch+1:2d}/20: Train Loss={train_loss/len(train_loader):.4f}, Val Loss={avg_val_loss:.4f}")

            # 加载最佳模型
            model.load_state_dict(best_model_state)

            # 评估
            model.eval()
            all_predictions = []
            all_labels = []

            with torch.no_grad():
                for features, labels in test_loader:
                    features, labels = features.to(self.device), labels.float().to(self.device)
                    outputs = model(features)
                    predicted = (torch.sigmoid(outputs) > 0.5).float()

                    all_predictions.append(outputs.cpu())
                    all_labels.append(labels.cpu())

            # 计算混淆矩阵和指标
            all_predictions_tensor = torch.cat(all_predictions, dim=0)
            all_labels_tensor = torch.cat(all_labels, dim=0)
            tn, fp, fn, tp = compute_confusion_entries(all_labels_tensor, all_predictions_tensor)

            total = tn + fp + fn + tp
            accuracy = (tn + tp) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            result = {
                'experiment_id': experiment_id,
                'dataset_path': dataset_path,
                'patch_size': patch_size,
                'use_weight': use_weight,
                'pos_weight': pos_weight,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                'total_samples': total,
                'train_samples': len(train_dataset),
                'test_samples': len(test_dataset),
                'N_pos': N_pos, 'N_neg': N_neg, 'class_ratio': r
            }

            print(f"  Results: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")
            print(f"  Confusion: TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")

            return result

        except Exception as e:
            print(f"  ❌ Error in {experiment_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_all_experiments(self):
        """运行所有18个设置"""
        experiment_count = 0
        total_settings = len(self.densities) * len(self.time_points) * len(self.patch_sizes)
        total_experiments = total_settings * len(self.seeds) * 2  # 2权重策略

        print(f"\n🚀 STARTING ALL EXPERIMENTS")
        print(f"Total settings: {total_settings}")
        print(f"Total experiments: {total_experiments}")

        for density in self.densities:
            for time_point in self.time_points:
                for patch_size in self.patch_sizes:
                    setting_name = f"{density}_{time_point}_patch{patch_size}"
                    print(f"\n{'='*80}")
                    print(f"🔬 SETTING: {setting_name}")
                    print(f"{'='*80}")

                    dataset_path = self.dataset_map[time_point][density]

                    for seed in self.seeds:
                        current_dataset_path = dataset_path.format(seed=seed)

                        if not Path(current_dataset_path).exists():
                            print(f"❌ Dataset not found: {current_dataset_path}")
                            continue

                        experiment_count += 1
                        print(f"\n⏳ Progress: {experiment_count}/{total_experiments}")

                        # 运行无权重版本
                        unweighted_id = f"{setting_name}_seed{seed}_unweighted"
                        unweighted_result = self.train_and_evaluate(
                            current_dataset_path, patch_size, False, unweighted_id
                        )
                        if unweighted_result:
                            self.results[setting_name].append(unweighted_result)

                        # 运行有权重版本
                        weighted_id = f"{setting_name}_seed{seed}_weighted"
                        weighted_result = self.train_and_evaluate(
                            current_dataset_path, patch_size, True, weighted_id
                        )
                        if weighted_result:
                            self.results[setting_name].append(weighted_result)

        print(f"\n✅ ALL EXPERIMENTS COMPLETED!")
        print(f"Settings processed: {len(self.results)}")

    def aggregate_and_report(self):
        """聚合结果并生成报告"""
        print(f"\n📊 COMPREHENSIVE RESULTS REPORT")
        print(f"{'='*100}")

        # 创建汇总表
        summary_data = []

        print(f"\n🎯 DETAILED RESULTS BY SETTING:")
        print("-" * 120)
        print(f"{'Setting':<25} {'Weight':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'TN':<8} {'FP':<8} {'FN':<8} {'TP':<8}")
        print("-" * 120)

        for setting_name, results_list in sorted(self.results.items()):
            # 分离无权重和有权重结果
            unweighted_results = [r for r in results_list if not r['use_weight']]
            weighted_results = [r for r in results_list if r['use_weight']]

            # 聚合无权重结果（如果有多个种子）
            if unweighted_results:
                avg_acc = np.mean([r['accuracy'] for r in unweighted_results])
                avg_prec = np.mean([r['precision'] for r in unweighted_results])
                avg_rec = np.mean([r['recall'] for r in unweighted_results])
                avg_f1 = np.mean([r['f1'] for r in unweighted_results])
                sum_tn = sum([r['tn'] for r in unweighted_results])
                sum_fp = sum([r['fp'] for r in unweighted_results])
                sum_fn = sum([r['fn'] for r in unweighted_results])
                sum_tp = sum([r['tp'] for r in unweighted_results])

                print(f"{setting_name:<25} {'None':<10} {avg_acc:<10.4f} {avg_prec:<10.4f} {avg_rec:<10.4f} {avg_f1:<10.4f} {sum_tn:<8} {sum_fp:<8} {sum_fn:<8} {sum_tp:<8}")

                summary_data.append({
                    'Setting': setting_name,
                    'Weight': 'None',
                    'Accuracy': avg_acc,
                    'Precision': avg_prec,
                    'Recall': avg_rec,
                    'F1': avg_f1,
                    'TN': sum_tn, 'FP': sum_fp, 'FN': sum_fn, 'TP': sum_tp,
                    'Num_Seeds': len(unweighted_results)
                })

            # 聚合有权重结果（如果有多个种子）
            if weighted_results:
                avg_acc = np.mean([r['accuracy'] for r in weighted_results])
                avg_prec = np.mean([r['precision'] for r in weighted_results])
                avg_rec = np.mean([r['recall'] for r in weighted_results])
                avg_f1 = np.mean([r['f1'] for r in weighted_results])
                sum_tn = sum([r['tn'] for r in weighted_results])
                sum_fp = sum([r['fp'] for r in weighted_results])
                sum_fn = sum([r['fn'] for r in weighted_results])
                sum_tp = sum([r['tp'] for r in weighted_results])
                avg_weight = np.mean([r['pos_weight'] for r in weighted_results])

                print(f"{setting_name:<25} {avg_weight:<10.3f} {avg_acc:<10.4f} {avg_prec:<10.4f} {avg_rec:<10.4f} {avg_f1:<10.4f} {sum_tn:<8} {sum_fp:<8} {sum_fn:<8} {sum_tp:<8}")

                summary_data.append({
                    'Setting': setting_name,
                    'Weight': f"{avg_weight:.3f}",
                    'Accuracy': avg_acc,
                    'Precision': avg_prec,
                    'Recall': avg_rec,
                    'F1': avg_f1,
                    'TN': sum_tn, 'FP': sum_fp, 'FN': sum_fn, 'TP': sum_tp,
                    'Num_Seeds': len(weighted_results)
                })

        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存CSV
        df = pd.DataFrame(summary_data)
        csv_file = f"results_18_settings_{timestamp}.csv"
        df.to_csv(csv_file, index=False)

        # 保存JSON
        json_file = f"results_18_settings_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n💾 Results saved to:")
        print(f"  CSV: {csv_file}")
        print(f"  JSON: {json_file}")

        # 生成对比分析
        self.generate_comparison_analysis(summary_data)

        return csv_file, json_file

    def generate_comparison_analysis(self, summary_data):
        """生成权重vs无权重的对比分析"""
        print(f"\n📈 WEIGHT VS NO-WEIGHT COMPARISON")
        print("=" * 80)

        # 按设置分组
        settings = {}
        for row in summary_data:
            setting = row['Setting']
            if setting not in settings:
                settings[setting] = {}
            if row['Weight'] == 'None':
                settings[setting]['unweighted'] = row
            else:
                settings[setting]['weighted'] = row

        print(f"{'Setting':<25} {'Unweighted':<10} {'Weighted':<10} {'ΔRecall':<10} {'ΔAccuracy':<10} {'Weight':<10}")
        print("-" * 80)

        total_recall_improvement = 0
        total_accuracy_improvement = 0
        valid_comparisons = 0

        for setting_name, results in sorted(settings.items()):
            if 'unweighted' in results and 'weighted' in results:
                unweighted = results['unweighted']
                weighted = results['weighted']

                recall_improvement = weighted['Recall'] - unweighted['Recall']
                accuracy_improvement = weighted['Accuracy'] - unweighted['Accuracy']

                print(f"{setting_name:<25} {unweighted['Recall']:<10.4f} {weighted['Recall']:<10.4f} {recall_improvement:<+10.4f} {accuracy_improvement:<+10.4f} {weighted['Weight']:<10}")

                total_recall_improvement += recall_improvement
                total_accuracy_improvement += accuracy_improvement
                valid_comparisons += 1

        if valid_comparisons > 0:
            avg_recall_improvement = total_recall_improvement / valid_comparisons
            avg_accuracy_improvement = total_accuracy_improvement / valid_comparisons

            print("-" * 80)
            print(f"{'AVERAGE':<25} {'':<10} {'':<10} {avg_recall_improvement:<+10.4f} {avg_accuracy_improvement:<+10.4f} {''}")

            print(f"\n🏆 KEY FINDINGS:")
            print(f"• Valid comparisons: {valid_comparisons}")
            print(f"• Average recall improvement: {avg_recall_improvement:+.4f} ({avg_recall_improvement*100:+.2f}%)")
            print(f"• Average accuracy improvement: {avg_accuracy_improvement:+.4f} ({avg_accuracy_improvement*100:+.2f}%)")

            if avg_recall_improvement > 0:
                print(f"✅ Weighted strategy consistently improves recall!")
            else:
                print(f"⚠️ Weighted strategy shows mixed results")

        # 按密度和时间点分析
        print(f"\n📊 ANALYSIS BY DENSITY AND TIME:")
        print("-" * 60)

        density_time_analysis = {}

        for setting_name, results in settings.items():
            if 'unweighted' in results and 'weighted' in results:
                parts = setting_name.split('_')
                density = parts[0]
                time_point = parts[1]
                patch_size = parts[2]

                key = f"{density}_{time_point}"
                if key not in density_time_analysis:
                    density_time_analysis[key] = []

                recall_improvement = results['weighted']['Recall'] - results['unweighted']['Recall']
                density_time_analysis[key].append(recall_improvement)

        print(f"{'Density_Time':<15} {'Avg ΔRecall':<12} {'Min':<8} {'Max':<8} {'Count':<8}")
        print("-" * 50)

        for key, improvements in sorted(density_time_analysis.items()):
            avg_imp = np.mean(improvements)
            min_imp = np.min(improvements)
            max_imp = np.max(improvements)
            count = len(improvements)
            print(f"{key:<15} {avg_imp:<+12.4f} {min_imp:<+8.4f} {max_imp:<+8.4f} {count:<8}")

def main():
    """主函数"""
    experiment = Correct18Experiment()

    try:
        # 运行所有实验
        experiment.run_all_experiments()

        # 聚合结果并生成报告
        csv_file, json_file = experiment.aggregate_and_report()

        end_time = datetime.now()
        duration = end_time - experiment.start_time

        print(f"\n🎉 EXPERIMENT COMPLETED!")
        print(f"{'='*80}")
        print(f"Total duration: {duration}")
        print(f"Results saved to: {csv_file}")
        print(f"JSON available at: {json_file}")
        print(f"Settings processed: {len(experiment.results)}")

    except KeyboardInterrupt:
        print(f"\n⚠️ Experiment interrupted by user")
        if experiment.results:
            print(f"Saving partial results...")
            experiment.aggregate_and_report()
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        if experiment.results:
            print(f"Saving partial results...")
            experiment.aggregate_and_report()

if __name__ == "__main__":
    main()