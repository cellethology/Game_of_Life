"""
Quick weight test on existing datasets.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

# Import locally
import sys
sys.path.append('.')
from data import LifePatchDataset
from models import MLP
from confusion_matrix_utils import compute_confusion_entries


def quick_test(dataset_path, weight_option=1):
    """Quick test with specific weight option."""
    print(f"\n🧪 Testing on {Path(dataset_path).name}")

    try:
        dataset = LifePatchDataset(dataset_path, split='train', patch_size=3)
        labels = dataset.y.numpy()
        N_pos = np.sum(labels == 1)
        N_neg = np.sum(labels == 0)
        r = N_neg / max(N_pos, 1)

        if weight_option == 1:
            pos_weight = min(5.0, np.sqrt(r))
            name = "Original"
        elif weight_option == 2:
            pos_weight = min(8.0, np.sqrt(r))
            name = "Option A"
        elif weight_option == 3:
            pos_weight = min(10.0, 1.5 * np.sqrt(r))
            name = "Option B"
        else:
            pos_weight = 1.0
            name = "No weight"

        print(f"  Data: N_pos={N_pos}, N_neg={N_neg}, r={r:.2f}")
        print(f"  Weight {name}: {pos_weight:.3f}")

        # Quick train
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLP(input_dim=8, hidden_dims=[128, 128], dropout=0.0).to(device)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_loader = DataLoader(dataset, batch_size=1024, shuffle=True)

        # 3 epochs quick test
        model.train()
        for epoch in range(3):
            for features, labels in train_loader:
                features, labels = features.to(device), labels.float().to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Test
        test_dataset = LifePatchDataset(dataset_path, split='test', patch_size=3)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

        model.eval()
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
        accuracy = (tn + tp) / total
        recall = tp / (tp + fn)

        print(f"  Results: Acc={accuracy:.3f}, Rec={recall:.3f}")
        print(f"  Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        return {"name": name, "weight": pos_weight, "accuracy": accuracy, "recall": recall}

    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    """Quick test on multiple datasets."""
    print("🧪 QUICK WEIGHT TEST ON EXISTING DATASETS")

    # Test on 3 datasets (using existing files with seed0)
    datasets = [
        ("data/life_patches_early_p0.2-burn10_steps40_seed0.npz", "p0.2"),
        ("data/life_patches_early_p0.4-burn10_steps40_seed0.npz", "p0.4"),
        ("data/life_patches_early_p0.6-burn10_steps40_seed0.npz", "p0.6"),
    ]

    all_results = {}

    for dataset_path, density in datasets:
        print(f"\n{'='*60}")
        print(f"DENSITY: {density}")
        print(f"{'='*60}")

        if not Path(dataset_path).exists():
            print(f"❌ Dataset not found: {dataset_path}")
            continue

        density_results = []
        for option in [1, 2, 3]:
            result = quick_test(dataset_path, option)
            if result:
                density_results.append(result)

        all_results[density] = density_results

        # Show comparison
        baseline = density_results[0]  # Original as baseline
        print(f"\n📊 Comparison for {density}:")
        print(f"{'Option':<10} {'Weight':<8} {'Accuracy':<8} {'Recall':<8}")
        print("-" * 50)

        for result in density_results:
            name = result["name"]
            weight = result["weight"]
            acc = result["accuracy"]
            rec = result["recall"]

            acc_change = acc - baseline["accuracy"] if baseline else 0
            rec_change = rec - baseline["recall"] if baseline else 0

            print(f"{name:<10} {weight:<8.3f} {acc:<8.3f} {rec:<8.3f}")
            if baseline:
                print(f"{'':10} {'':8} ΔAcc={acc_change:+.3f} ΔRec={rec_change:+.3f}")

    # Overall summary
    print(f"\n{'='*80}")
    print("🏆 OVERALL SUMMARY")
    print(f"{'='*80}")

    for density, results in all_results.items():
        if len(results) >= 3:
            best_by_acc = max(results, key=lambda x: x["accuracy"])
            best_by_rec = max(results, key=lambda x: x["recall"])
            print(f"\n{density}:")
            print(f"  Best accuracy: {best_by_acc['name']} ({best_by_acc['weight']:.3f}) = {best_by_acc['accuracy']:.3f}")
            print(f"  Best recall:   {best_by_rec['name']} ({best_by_rec['weight']:.3f}) = {best_by_rec['recall']:.3f}")


if __name__ == "__main__":
    main()