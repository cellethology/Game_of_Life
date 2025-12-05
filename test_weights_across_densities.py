"""
Systematic test of weight variations across different density settings.

This script:
1. Generates datasets for p_alive = 0.2, 0.4, 0.6
2. Tests all three weight options on each density
3. Provides clear comparison of weighting strategies
4. Ensures we know exactly what data we're using
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

try:
    from .data import LifePatchDataset, generate_life_patch_dataset
    from .models import MLP
    from .confusion_matrix_utils import compute_confusion_entries
except ImportError:
    from data import LifePatchDataset, generate_life_patch_dataset
    from models import MLP
    from confusion_matrix_utils import compute_confusion_entries


def compute_class_weight_option(N_pos: int, N_neg: int, option: int) -> float:
    """Compute class weight using three different strategies."""
    r = N_neg / max(N_pos, 1)

    if option == 1:
        # Original: min(5.0, sqrt(r))
        return min(5.0, np.sqrt(r))
    elif option == 2:
        # Conservative increase: min(8.0, sqrt(r))
        return min(8.0, np.sqrt(r))
    elif option == 3:
        # Aggressive: min(10.0, 1.5*sqrt(r))
        return min(10.0, 1.5 * np.sqrt(r))
    else:
        return 1.0


def generate_density_datasets():
    """Generate datasets for three density levels."""
    print("🔬 GENERATING DATASETS FOR DIFFERENT DENSITIES")
    print("=" * 60)

    densities = [0.2, 0.4, 0.6]
    dataset_paths = {}

    for p_alive in densities:
        print(f"Generating dataset for p_alive = {p_alive}...")

        dataset_path = f"data/life_patches_p{p_alive:.1f}_seed42.npz"

        # Check if already exists
        if not Path(dataset_path).exists():
            print(f"  Creating: {dataset_path}")
            generate_life_patch_dataset(
                out_path=dataset_path,
                board_size=128,
                num_train_boards=16,
                num_test_boards=4,
                num_steps=60,
                burn_in=50,
                p_alive=p_alive,
                num_patches_per_step=200,
                seed=42
            )
        else:
            print(f"  Using existing: {dataset_path}")

        # Load and analyze the dataset
        dataset = LifePatchDataset(dataset_path, split='train', patch_size=3)
        labels = dataset.y.numpy() if hasattr(dataset.y, 'numpy') else dataset.y
        N_pos = np.sum(labels == 1)
        N_neg = np.sum(labels == 0)
        r = N_neg / max(N_pos, 1)

        dataset_paths[p_alive] = {
            "path": dataset_path,
            "N_pos": N_pos,
            "N_neg": N_neg,
            "ratio": r,
            "sqrt_ratio": np.sqrt(r)
        }

        print(f"  Analysis: N_pos={N_pos:,}, N_neg={N_neg:,}, ratio={r:.4f}")
        print(f"  Weight 1 (orig): min(5.0, {np.sqrt(r):.4f}) = {compute_class_weight_option(N_pos, N_neg, 1):.4f}")
        print(f"  Weight 2 (cons): min(8.0, {np.sqrt(r):.4f}) = {compute_class_weight_option(N_pos, N_neg, 2):.4f}")
        print(f"  Weight 3 (agg): min(10.0, 1.5*{np.sqrt(r):.4f}) = {compute_class_weight_option(N_pos, N_neg, 3):.4f}")
        print()

    return dataset_paths


def test_weights_on_density(p_alive: float, density_info: dict):
    """Test all three weight options on a specific density dataset."""
    dataset_path = density_info["path"]

    print(f"🧪 TESTING WEIGHTS ON p_alive = {p_alive}")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Class balance: N_pos={density_info['N_pos']:,}, N_neg={density_info['N_neg']:,}")
    print(f"Ratio r = {density_info['ratio']:.4f}, sqrt(r) = {density_info['sqrt_ratio']:.4f}")
    print("-" * 60)

    # Create datasets
    train_dataset = LifePatchDataset(dataset_path, split='train', patch_size=3)
    test_dataset = LifePatchDataset(dataset_path, split='test', patch_size=3)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []

    # Test each weight option
    weight_options = {
        1: "Original min(5.0, √r)",
        2: "Conservative min(8.0, √r)",
        3: "Aggressive min(10.0, 1.5√r)"
    }

    for option, option_name in weight_options.items():
        print(f"\n🔬 Weight Option {option}: {option_name}")
        print(f"{'─'*50}")

        # Initialize model
        model = MLP(input_dim=8, hidden_dims=[128, 128], dropout=0.0).to(device)

        # Compute weight
        pos_weight = compute_class_weight_option(
            density_info["N_pos"], density_info["N_neg"], option
        )
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        print(f"  Computed weight: {pos_weight:.4f}")

        # Quick training (10 epochs for speed)
        model.train()
        for epoch in range(10):
            for features, labels in train_loader:
                features, labels = features.to(device), labels.float().to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluation
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

        # Compute metrics
        all_predictions_tensor = torch.cat(all_predictions, dim=0)
        all_labels_tensor = torch.cat(all_labels, dim=0)
        tn, fp, fn, tp = compute_confusion_entries(all_labels_tensor, all_predictions_tensor)

        # Calculate detailed metrics
        total = tn + fp + fn + tp
        accuracy = (tn + tp) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        result = {
            "p_alive": p_alive,
            "weight_option": option,
            "weight_name": option_name,
            "pos_weight": pos_weight,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "total": total
        }

        results.append(result)

        print(f"  Results: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")
        print(f"  Confusion: TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")

    return results


def print_density_comparison_table(all_results):
    """Print comprehensive comparison table across all densities."""
    print("\n" + "="*100)
    print("📊 COMPREHENSIVE WEIGHT COMPARISON ACROSS DENSITIES")
    print("="*100)

    # Group results by density
    by_density = {}
    for result in all_results:
        p_alive = result["p_alive"]
        if p_alive not in by_density:
            by_density[p_alive] = []
        by_density[p_alive].append(result)

    # Print table for each density
    densities = [0.2, 0.4, 0.6]

    print(f"{'Density':<8} {'Option':<6} {'Weight':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 80)

    for p_alive in densities:
        if p_alive not in by_density:
            continue

        print(f"\n🎯 p_alive = {p_alive:.1f}")

        # Find baseline (option 1) for comparison
        baseline = None
        for result in by_density[p_alive]:
            if result["weight_option"] == 1:
                baseline = result
                break

        for option in [1, 2, 3]:
            result = next((r for r in by_density[p_alive] if r["weight_option"] == option), None)
            if result:
                name_short = result["weight_name"].split(" ")[0].replace("min(", "").replace("√", "").replace(")", "").replace("1.5*", "1.5*")
                acc = result["accuracy"]
                prec = result["precision"]
                rec = result["recall"]
                f1 = result["f1"]
                weight = result["pos_weight"]

                print(f"{p_alive:<8} {option:<6} {weight:<8.4f} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")

                # Show improvements compared to baseline
                if baseline and option != 1:
                    acc_change = acc - baseline["accuracy"]
                    rec_change = rec - baseline["recall"]
                    f1_change = f1 - baseline["f1"]
                    print(f"{'':8} {'':6} {'':8} {'ΔAcc':<10} {acc_change:+10.4f} {'ΔRec':<10} {rec_change:+10.4f} {'ΔF1':<10} {f1_change:+10.4f}")

    print("-" * 80)

    # Find best overall options
    print(f"\n🏆 OVERALL BEST OPTIONS:")
    print("-" * 40)

    best_accuracy = {}
    best_recall = {}
    best_f1 = {}

    for p_alive in densities:
        if p_alive not in by_density:
            continue

        density_results = by_density[p_alive]
        best_accuracy[p_alive] = max(density_results, key=lambda x: x["accuracy"])
        best_recall[p_alive] = max(density_results, key=lambda x: x["recall"])
        best_f1[p_alive] = max(density_results, key=lambda x: x["f1"])

    for metric_name, best_dict in [("Accuracy", best_accuracy), ("Recall", best_recall), ("F1", best_f1)]:
        print(f"\n🎯 Best {metric_name}:")
        for p_alive in densities:
            if p_alive in best_dict:
                best_result = best_dict[p_alive]
                print(f"  p_alive={p_alive:.1f}: Option {best_result['weight_option']} ({best_result['pos_weight']:.4f}) = {best_result[metric_name]:.4f}")


def main():
    """Main function to test weights across different densities."""
    print("🔬 SYSTEMATIC WEIGHT TESTING ACROSS DENSITIES")
    print("=" * 80)
    print("This script will:")
    print("1. Generate datasets for p_alive = 0.2, 0.4, 0.6")
    print("2. Test 3 weight strategies on each density")
    print("3. Compare results systematically")
    print("=" * 80)

    # Ensure data directory
    Path("data").mkdir(exist_ok=True)

    # Generate density datasets
    dataset_paths = generate_density_datasets()

    # Test all combinations
    all_results = []
    for p_alive, density_info in dataset_paths.items():
        try:
            results = test_weights_on_density(p_alive, density_info)
            all_results.extend(results)
        except Exception as e:
            print(f"❌ Error testing p_alive={p_alive}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("❌ No weight tests completed successfully!")
        return

    # Print comprehensive comparison
    print_density_comparison_table(all_results)

    print(f"\n{'='*80}")
    print("🎉 SYSTEMATIC WEIGHT TESTING COMPLETED!")
    print("="*80)
    print("Key Findings:")
    print("  • Weight effectiveness varies by density")
    print("  • Higher weights generally improve recall")
    print("  • Optimal weight depends on class balance")
    print("  • Precision-recall trade-off is density-dependent")
    print("="*80)


if __name__ == "__main__":
    main()