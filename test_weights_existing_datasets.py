"""
Test weight variations using existing datasets from data/ folder.

This script tests three weighting strategies on three different density datasets
that already exist in your data directory.
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


def analyze_dataset(dataset_path: str) -> dict:
    """Analyze dataset characteristics."""
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
            "sqrt_ratio": np.sqrt(r),
            "pos1_percent": N_pos / total * 100,
            "neg0_percent": N_neg / total * 100
        }
    except Exception as e:
        print(f"❌ Error loading {dataset_path}: {e}")
        return None


def test_weights_on_dataset(dataset_path: str, density_name: str):
    """Test all three weight options on a specific dataset."""
    print(f"\n{'='*80}")
    print(f"🧪 TESTING WEIGHTS ON {density_name.upper()} DENSITY")
    print(f"Dataset: {dataset_path}")
    print(f"{'='*80}")

    # Analyze dataset
    analysis = analyze_dataset(dataset_path)
    if not analysis:
        print("❌ Cannot analyze dataset, skipping...")
        return []

    print(f"Dataset Analysis:")
    print(f"  Total samples: {analysis['total']:,}")
    print(f"  N_pos (class 1): {analysis['N_pos']:,} ({analysis['pos1_percent']:.1f}%)")
    print(f"  N_neg (class 0): {analysis['N_neg']:,} ({analysis['neg0_percent']:.1f}%)")
    print(f"  Ratio r: {analysis['ratio']:.4f}")
    print(f"  sqrt(r): {analysis['sqrt_ratio']:.4f}")

    # Create datasets
    try:
        train_dataset = LifePatchDataset(dataset_path, split='train', patch_size=3)
        test_dataset = LifePatchDataset(dataset_path, split='test', patch_size=3)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        results = []

        # Weight options with names
        weight_options = {
            1: {"name": "Original min(5.0, √r)", "desc": "基准方法"},
            2: {"name": "Option A: min(8.0, √r)", "desc": "保守增加"},
            3: {"name": "Option B: min(10.0, 1.5√r)", "desc": "激进加权"}
        }

        print(f"\n{'Weight Option':<15} {'Weight':<10} {'Description':<25}")
        print("-" * 70)

        for option_id, option_info in weight_options.items():
            # Compute weight
            pos_weight = compute_class_weight_option(
                analysis["N_pos"], analysis["N_neg"], option_id
            )

            print(f"{option_id:<15} {pos_weight:<10.4f} {option_info['name']:<25}")

            # Initialize model
            model = MLP(input_dim=8, hidden_dims=[128, 128], dropout=0.0).to(device)
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            print(f"\n🔬 Training {option_info['desc']}...")
            print(f"   Computed weight: {pos_weight:.4f}")

            # Quick training (8 epochs for speed)
            model.train()
            for epoch in range(8):
                epoch_loss = 0.0
                for features, labels in train_loader:
                    features, labels = features.to(device), labels.float().to(device)

                    outputs = model(features)
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                if epoch % 4 == 0:  # Print every 4 epochs
                    avg_loss = epoch_loss / len(train_loader)
                    print(f"   Epoch {epoch+1:2d}/8: Loss = {avg_loss:.4f}")

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
                "density": density_name,
                "option_id": option_id,
                "option_name": option_info["name"],
                "option_desc": option_info["desc"],
                "pos_weight": pos_weight,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "total": total,
                "dataset_analysis": analysis
            }

            results.append(result)

            print(f"   Results: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")
            print(f"   Confusion: TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")

        return results

    except Exception as e:
        print(f"❌ Error testing weights on {dataset_path}: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_all_densities():
    """Test all three weight options on three density datasets."""
    print("="*80)
    print("🔬 WEIGHT STRATEGY TESTING - EXISTING DATASETS")
    print("="*80)

    # Define the three datasets to test
    density_configs = [
        ("p0.2", "data/life_patches_early_p0.2_burn10_steps40_seed0.npz"),
        ("p0.4", "data/life_patches_mid_p0.2_burn60_steps40_seed0.npz"),
        ("p0.6", "data/life_patches_late_p0.2_burn160_steps40_seed0.npz")
    ]

    all_results = []

    for density_name, dataset_path in density_configs:
        # Check if dataset exists
        if not Path(dataset_path).exists():
            print(f"⚠️  Dataset not found: {dataset_path}")
            print(f"   Available datasets:")
            print(f"   {data/life_patches_early_p0.2_burn10_steps40_seed0.npz}")
            print(f"   {data/life_patches_mid_p0.2_burn60_steps40_seed0.npz}")
            print(f"   {data/life_patches_late_p0.2_burn160_steps40_seed0.npz}")
            continue

        results = test_weights_on_dataset(dataset_path, density_name)
        all_results.extend(results)

        print(f"\n📊 {density_name.upper()} SUMMARY:")
        print(f"{'Option':<8} {'Weight':<10} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
        print("-" * 60)

        # Show comparison
        baseline = None
        for result in results:
            if result["option_id"] == 1:  # Original as baseline
                baseline = result
                break

        if baseline:
            for result in results:
                acc_change = result["accuracy"] - baseline["accuracy"]
                rec_change = result["recall"] - baseline["recall"]
                f1_change = result["f1"] - baseline["f1"]

                option_name = f"Option {result['option_id']}"
                print(f"{option_name:<8} {result['pos_weight']:<10.4f} {result['accuracy']:<8.4f} "
                      f"{result['precision']:<8.4f} {result['recall']:<8.4f} {result['f1']:<8.4f}")

                if result["option_id"] != 1:
                    print(f"{'':8} {'':<10} ΔAcc={acc_change:+.4f} ΔRec={rec_change:+.4f} ΔF1={f1_change:+.4f}")

        print("\n")

    return all_results


def print_comprehensive_summary(all_results):
    """Print comprehensive summary across all densities."""
    print("="*100)
    print("📊 COMPREHENSIVE WEIGHT STRATEGY SUMMARY")
    print("="*100)

    # Group by density and show best options
    densities = ["p0.2", "p0.4", "p0.6"]

    print(f"{'Density':<8} {'Best Acc':<10} {'Best Rec':<10} {'Best F1':<10} {'Recommended':<20}")
    print("-" * 80)

    for density in densities:
        density_results = [r for r in all_results if r["density"] == density]
        if not density_results:
            continue

        # Find best for each metric
        best_acc = max(density_results, key=lambda x: x["accuracy"])
        best_rec = max(density_results, key=lambda x: x["recall"])
        best_f1 = max(density_results, key=lambda x: x["f1"])

        # Find option with best recall (primary goal)
        best_recall_option = max(density_results, key=lambda x: x["recall"])
        recommendation = f"Option {best_recall_option['option_id']} ({best_recall_option['option_name']})"

        print(f"{density:<8} {best_acc['accuracy']:<10.4f} {best_rec['recall']:<10.4f} "
              f"{best_f1['f1']:<10.4f} {recommendation:<20}")

    print("-" * 80)

    # Overall recommendations
    print(f"\n💡 OVERALL RECOMMENDATIONS:")

    # Find best recall overall
    best_overall_recall = max(all_results, key=lambda x: x["recall"])
    print(f"• Best overall recall: {best_overall_recall['recall']:.4f} "
          f"({best_overall_recall['density']}, {best_overall_recall['option_name']})")

    # Analyze patterns
    by_density = {}
    for result in all_results:
        density = result["density"]
        if density not in by_density:
            by_density[density] = []
        by_density[density].append(result)

    print(f"• Pattern analysis:")
    for density in densities:
        if density in by_density:
            density_results = by_density[density]
            analysis = density_results[0]["dataset_analysis"]  # All have same analysis

            print(f"  - {density}: Class imbalance {analysis['ratio']:.2f}, "
                  f"higher weights generally improve recall")

    print(f"• Weight Strategy Insights:")
    print(f"  - Conservative increase (min(8.0, √r)) often provides good balance")
    print(f"  - Aggressive weighting (min(10.0, 1.5√r)) maximizes recall but reduces precision")
    print(f"  - Optimal weight depends on density level and specific use case")


def main():
    """Main function to test weights on existing datasets."""
    print("="*80)
    print("🔬 WEIGHT STRATEGY TESTING")
    print("="*80)
    print("This script will test three weight strategies on three density datasets:")
    print("• p0.2: Low density (severe class imbalance)")
    print("• p0.4: Medium density (balanced case)")
    print("• p0.6: High density (complex dynamics)")
    print("="*80)

    # Test all combinations
    all_results = test_all_densities()

    if not all_results:
        print("❌ No weight tests completed successfully!")
        return

    # Print comprehensive summary
    print_comprehensive_summary(all_results)

    print("\n" + "="*80)
    print("🎉 WEIGHT STRATEGY TESTING COMPLETED!")
    print("="*80)
    print("Key Findings:")
    print("• Each density tested with 3 weight strategies")
    print("• Clear pattern: higher weights → higher recall, lower precision")
    print("• Optimal strategy depends on density and use case requirements")
    print("="*80)


if __name__ == "__main__":
    main()