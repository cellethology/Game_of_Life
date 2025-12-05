"""
🔬 COMPREHENSIVE 27-SETTING EXPERIMENT
完整的 27 个设置实验: 3密度 × 3时间点 × 3种子
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

# Import locally
import sys
sys.path.append('.')
from data import LifePatchDataset
from models import MLP
from confusion_matrix_utils import compute_confusion_entries

class ComprehensiveExperiment:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = []
        self.start_time = datetime.now()

        # 实验设计
        self.densities = ["p0.2", "p0.4", "p0.6"]
        self.time_points = ["early", "mid", "late"]  # 对应不同的burn-in steps
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

        print(f"🔬 COMPREHENSIVE 27-SETTING EXPERIMENT")
        print(f"Device: {self.device}")
        print(f"Total experiments: {len(self.densities) * len(self.time_points) * len(self.seeds)}")
        print(f"Densities: {self.densities}")
        print(f"Time points: {self.time_points}")
        print(f"Seeds: {self.seeds}")
        print(f"Started at: {self.start_time}")

    def compute_optimal_weight(self, dataset_path):
        """计算给定数据集的最优权重"""
        try:
            dataset = LifePatchDataset(dataset_path, split='train', patch_size=3)
            labels = dataset.y.numpy() if hasattr(dataset.y, 'numpy') else dataset.y
            N_pos = np.sum(labels == 1)
            N_neg = np.sum(labels == 0)
            r = N_neg / max(N_pos, 1)
            optimal_weight = np.sqrt(r)
            return optimal_weight, N_pos, N_neg, r
        except Exception as e:
            print(f"Error computing weight for {dataset_path}: {e}")
            return 1.0, 0, 0, 1.0

    def train_and_evaluate(self, dataset_path, weight_strategy, experiment_id):
        """训练并评估一个设置"""
        try:
            print(f"\n🧪 Experiment {experiment_id}: {weight_strategy['name']}")
            print(f"Dataset: {dataset_path}")

            # 计算权重
            if weight_strategy['type'] == 'unweighted':
                pos_weight = 1.0
            else:
                pos_weight, N_pos, N_neg, r = self.compute_optimal_weight(dataset_path)
                if weight_strategy['type'] == 'conservative':
                    pos_weight = min(8.0, pos_weight)
                elif weight_strategy['type'] == 'aggressive':
                    pos_weight = min(10.0, 1.5 * pos_weight)
                # 'optimal' uses the computed sqrt(r) directly

            print(f"Computed weight: {pos_weight:.4f}")

            # 创建数据集
            train_dataset = LifePatchDataset(dataset_path, split='train', patch_size=3)
            test_dataset = LifePatchDataset(dataset_path, split='test', patch_size=3)

            train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

            # 模型和训练设置
            model = MLP(input_dim=8, hidden_dims=[128, 128], dropout=0.0).to(self.device)
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(self.device)
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # 快速训练（减少epoch数以加快实验）
            model.train()
            for epoch in range(8):  # 减少到8个epoch以加快速度
                epoch_loss = 0.0
                for features, labels in train_loader:
                    features, labels = features.to(self.device), labels.float().to(self.device)

                    outputs = model(features)
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                if epoch % 4 == 0:  # 每4个epoch打印一次
                    avg_loss = epoch_loss / len(train_loader)
                    print(f"  Epoch {epoch+1}/8: Loss = {avg_loss:.4f}")

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
                'weight_strategy': weight_strategy['name'],
                'weight_type': weight_strategy['type'],
                'pos_weight': pos_weight,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                'total_samples': total,
                'train_samples': len(train_dataset),
                'test_samples': len(test_dataset)
            }

            print(f"  Results: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")
            print(f"  Confusion: TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")

            return result

        except Exception as e:
            print(f"  ❌ Error in experiment {experiment_id}: {e}")
            return None

    def run_single_density_complete(self, density, time_point, seed):
        """运行单个密度的完整实验（包含所有权重策略）"""
        dataset_path = self.dataset_map[time_point][density].format(seed=seed)

        if not Path(dataset_path).exists():
            print(f"❌ Dataset not found: {dataset_path}")
            return []

        print(f"\n{'='*80}")
        print(f"🔬 {density.upper()} {time_point.upper()} SEED{seed}")
        print(f"Dataset: {dataset_path}")
        print(f"{'='*80}")

        # 权重策略
        weight_strategies = [
            {'name': 'Unweighted', 'type': 'unweighted'},
            {'name': 'Optimal', 'type': 'optimal'},
            {'name': 'Conservative', 'type': 'conservative'},
            {'name': 'Aggressive', 'type': 'aggressive'}
        ]

        results = []
        for i, strategy in enumerate(weight_strategies):
            experiment_id = f"{density}_{time_point}_seed{seed}_{strategy['type']}"
            result = self.train_and_evaluate(dataset_path, strategy, experiment_id)
            if result:
                results.append(result)

        return results

    def run_all_experiments(self):
        """运行所有27个设置"""
        print(f"\n🚀 STARTING ALL 27 EXPERIMENTS")
        print(f"Expected total time: ~{27 * 5} minutes (estimated)")

        experiment_count = 0
        total_experiments = len(self.densities) * len(self.time_points) * len(self.seeds)

        for density in self.densities:
            for time_point in self.time_points:
                for seed in self.seeds:
                    experiment_count += 1
                    print(f"\n⏳ Progress: {experiment_count}/{total_experiments}")

                    results = self.run_single_density_complete(density, time_point, seed)
                    self.results.extend(results)

        print(f"\n✅ ALL EXPERIMENTS COMPLETED!")
        print(f"Total results collected: {len(self.results)}")

    def save_results(self):
        """保存结果到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存为JSON
        json_file = f"results_27_experiments_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        # 保存为CSV
        df = pd.DataFrame(self.results)
        csv_file = f"results_27_experiments_{timestamp}.csv"
        df.to_csv(csv_file, index=False)

        print(f"💾 Results saved to:")
        print(f"  JSON: {json_file}")
        print(f"  CSV: {csv_file}")

        return json_file, csv_file

    def analyze_results(self):
        """分析实验结果"""
        if not self.results:
            print("❌ No results to analyze!")
            return

        print(f"\n📊 COMPREHENSIVE ANALYSIS OF 27 EXPERIMENTS")
        print(f"{'='*80}")

        df = pd.DataFrame(self.results)

        # 按密度分析
        print(f"\n🎯 ANALYSIS BY DENSITY:")
        print("-" * 50)
        for density in self.densities:
            density_results = df[df['experiment_id'].str.contains(density)]
            print(f"\n{density.upper()} ({len(density_results)} results):")

            # 按权重类型分组
            for weight_type in ['unweighted', 'optimal', 'conservative', 'aggressive']:
                type_results = density_results[density_results['weight_type'] == weight_type]
                if not type_results.empty:
                    avg_recall = type_results['recall'].mean()
                    avg_precision = type_results['precision'].mean()
                    avg_f1 = type_results['f1'].mean()
                    print(f"  {weight_type:12}: Recall={avg_recall:.4f}, Precision={avg_precision:.4f}, F1={avg_f1:.4f}")

        # 按时间点分析
        print(f"\n⏰ ANALYSIS BY TIME POINT:")
        print("-" * 50)
        for time_point in self.time_points:
            time_results = df[df['experiment_id'].str.contains(time_point)]
            print(f"\n{time_point.upper()} ({len(time_results)} results):")

            for weight_type in ['unweighted', 'optimal', 'conservative', 'aggressive']:
                type_results = time_results[time_results['weight_type'] == weight_type]
                if not type_results.empty:
                    avg_recall = type_results['recall'].mean()
                    print(f"  {weight_type:12}: Recall={avg_recall:.4f}")

def main():
    """主函数"""
    experiment = ComprehensiveExperiment()

    try:
        # 运行所有实验
        experiment.run_all_experiments()

        # 保存结果
        json_file, csv_file = experiment.save_results()

        # 分析结果
        experiment.analyze_results()

        end_time = datetime.now()
        duration = end_time - experiment.start_time

        print(f"\n🎉 EXPERIMENT COMPLETED!")
        print(f"{'='*80}")
        print(f"Total duration: {duration}")
        print(f"Results saved to: {json_file}")
        print(f"CSV available at: {csv_file}")
        print(f"Total experiments: {len(experiment.results)}")

    except KeyboardInterrupt:
        print(f"\n⚠️ Experiment interrupted by user")
        if experiment.results:
            print(f"Saving partial results...")
            experiment.save_results()
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        if experiment.results:
            print(f"Saving partial results...")
            experiment.save_results()

if __name__ == "__main__":
    main()