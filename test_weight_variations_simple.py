"""
Simplified test for different weight variations.

This script directly tests three weighting strategies on the same dataset
with minimal complexity to focus on the core question: which weighting works best?
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

try:
    from .data import LifePatchDataset
    from .models import MLP
    from .confusion_matrix_utils import compute_confusion_entries
except ImportError:
    from data import LifePatchDataset
    from models import MLP
    from confusion_matrix_utils import compute_confusion_entries


def compute_class_weight_option(N_pos: int, N_neg: int, option: int) -> float:
    """Compute class weight for different options."""
    r = N_neg / max(N_pos, 1)

    if option == 1:
        # Original: min(5.0, sqrt(r))
        return min(5.0, np.sqrt(r))
    elif option == 2:
        # Option A: Increase upper limit to 8.0
        return min(8.0, np.sqrt(r))
    elif option == 3:
        # Option B: Use 1.5x sqrt(r) with upper limit 10.0
        return min(10.0, 1.5 * np.sqrt(r))
    else:
        return 1.0


def test_weight_options_simple():
    """Test three weight options quickly."""
    print("="*70)
    print("🔬 TESTING CLASS WEIGHT VARIATIONS - SIMPLIFIED")
    print("="*70)

    # Load dataset
    data_path = "data/life_patches.npz"
    dataset = LifePatchDataset(data_path, split="train", patch_size=3)
    test_dataset = LifePatchDataset(data_path, split="test", patch_size=3)

    # Count classes
    labels = dataset.y.numpy() if hasattr(dataset.y, 'numpy') else dataset.y
    N_pos = np.sum(labels == 1)
    N_neg = np.sum(labels == 0)
    r = N_neg / max(N_pos, 1)

    print(f"Dataset Analysis:")
    print(f"  N_pos: {N_pos:,}")
    print(f"  N_neg: {N_neg:,}")
    print(f"  Ratio r: {r:.4f}")
    print(f"  sqrt(r): {np.sqrt(r):.4f}")

    # Calculate weights for each option
    options = {
        1: {"name": "Original min(5.0, √r)", "weight": compute_class_weight_option(N_pos, N_neg, 1)},
        2: {"name": "Option A: min(8.0, √r)", "weight": compute_class_weight_option(N_pos, N_neg, 2)},
        3: {"name": "Option B: min(10.0, 1.5√r)", "weight": compute_class_weight_option(N_pos, N_neg, 3)}
    }

    print(f"\nWeight Options:")
    for option_id, option_info in options.items():
        print(f"  Option {option_id}: {option_info['name']} = {option_info['weight']:.4f}")

    # Test each option with a quick training run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    print(f"\n{'='*70}")
    print("TRAINING COMPARISON - QUICK TEST (reduced epochs)")
    print("="*70)

    results = []

    for option_id, option_info in options.items():
        print(f"\n🔬 Testing Option {option_id}: {option_info['name']}")
        print(f"   Weight: {option_info['weight']:.4f}")

        # Initialize model
        model = MLP(input_dim=8, hidden_dims=[128, 128], dropout=0.0).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([option_info['weight']], device=device))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Quick training (5 epochs max)
        model.train()
        for epoch in range(5):
            epoch_loss = 0.0
            for features, labels in train_loader:
                features, labels = features.to(device), labels.float().to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        # Evaluate
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

        # Compute confusion matrix and metrics
        all_predictions_tensor = torch.cat(all_predictions, dim=0)
        all_labels_tensor = torch.cat(all_labels, dim=0)
        tn, fp, fn, tp = compute_confusion_entries(all_labels_tensor, all_predictions_tensor)

        # Calculate metrics
        total = tn + fp + fn + tp
        accuracy = (tn + tp) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        result = {
            "option": option_id,
            "name": option_info['name'],
            "weight": option_info['weight'],
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp
        }

        results.append(result)

        print(f"   Results: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")
        print(f"   Confusion: TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")

    return results


def print_comparison_results(results):
    """Print clean comparison of weight options."""
    print(f"\n{'='*80}")
    print("📊 WEIGHT VARIATIONS COMPARISON RESULTS")
    print("="*80)

    print(f"{'Option':<6} {'Weight':<8} {'Name':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 88)

    baseline_result = None
    for result in results:
        if result["option"] == 1:  # Original method
            baseline_result = result
            break

    for result in results:
        name_short = result["name"].replace("min(", "").replace(")", "").replace("√", "sqrt")
        acc = result["accuracy"]
        prec = result["precision"]
        rec = result["recall"]
        f1 = result["f1"]

        print(f"{result['option']:<6} {result['weight']:<8.4f} {name_short:<30} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")

        if baseline_result:
            acc_change = acc - baseline_result["accuracy"]
            rec_change = rec - baseline_result["recall"]
            f1_change = f1 - baseline_result["f1"]

            if result["option"] != 1:
                print(f"{'':6} {'':8} {'':30} ΔAcc={acc_change:+.4f} ΔRec={rec_change:+.4f} ΔF1={f1_change:+.4f}")

    # Find best for each metric
    best_acc = max(results, key=lambda x: x["accuracy"])
    best_rec = max(results, key=lambda x: x["recall"])
    best_f1 = max(results, key=lambda x: x["f1"])

    print(f"\n🏆 BEST OPTIONS:")
    print(f"   Best Accuracy: Option {best_acc['option']} ({best_acc['name']}) = {best_acc['accuracy']:.4f}")
    print(f"   Best Recall:   Option {best_rec['option']} ({best_rec['name']}) = {best_rec['recall']:.4f}")
    print(f"   Best F1 Score: Option {best_f1['option']} ({best_f1['name']}) = {best_f1['f1']:.4f}")

    # Recommendation
    print(f"\n💡 RECOMMENDATION:")
    if best_rec["option"] == 1:
        print("   Original weighting (min(5.0, √r)) performs best for recall")
    elif best_rec["option"] == 2:
        print("   Option A (min(8.0, √r)) performs best for recall")
    elif best_rec["option"] == 3:
        print("   Option B (min(10.0, 1.5√r)) performs best for recall")

    print(f"   Recommended for Class 1 (Alive Cell) Detection: Option {best_rec['option']}")


def main():
    """Main function to test weight variations."""
    # Check dataset exists
    if not Path("data/life_patches.npz").exists():
        print("Dataset not found! Please run: python -m train_weighted")
        return

    # Run weight variation tests
    results = test_weight_options_simple()

    if results:
        # Print comparison
        print_comparison_results(results)

        print(f"\n{'='*80}")
        print("🎉 WEIGHT VARIATIONS TEST COMPLETED!")
        print("="*80)
    else:
        print("❌ No weight variation tests completed!")


if __name__ == "__main__":
    main()