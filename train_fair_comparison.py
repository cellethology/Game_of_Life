"""Fair comparison training script for Conway's Game of Life MLP prediction.

This script provides the SAME early stopping mechanism as the weighted version
but WITHOUT class weighting to enable fair comparison.

Key differences from train_weighted.py:
- NO class weighting (pos_weight = 1.0)
- SAME early stopping (patience=5, max_epochs=30, val_split=0.2)
- SAME validation split and epsilon
- SAME checkpoint directory structure (for comparison)
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


def create_train_val_split(
    dataset: LifePatchDataset,
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[LifePatchDataset, LifePatchDataset]:
    """
    Split dataset into training and validation sets (SAME as weighted version).

    Args:
        dataset: Original dataset to split
        val_split: Fraction of data for validation (default: 0.2)
        seed: Random seed for reproducible splits

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    torch.manual_seed(seed)

    # Calculate split sizes (SAME as weighted version)
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    # Split dataset
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"Dataset split: {train_size:,} training, {val_size:,} validation samples")
    return train_dataset, val_dataset


def train_one_model_fair_comparison(
    patch_size: int,
    npz_path: str = "data/life_patches.npz",
    batch_size: int = 1024,
    max_epochs: int = 30,          # SAME as weighted
    learning_rate: float = 1e-3,    # SAME as weighted
    device: torch.device | None = None,
    patience: int = 5,              # SAME as weighted
    val_split: float = 0.2,        # SAME as weighted
    epsilon: float = 1e-4,          # SAME as weighted
    checkpoint_dir: str = "checkpoints_fair_comparison",  # Different directory for comparison
    seed: int = 42
) -> Dict[str, Any]:
    """
    Train and evaluate an MLP with SAME early stopping as weighted version
    but WITHOUT class weighting.

    Args:
        patch_size: Size of the neighborhood (3 or 5)
        npz_path: Path to the dataset file
        batch_size: Batch size for training
        max_epochs: Maximum number of training epochs (SAME as weighted)
        learning_rate: Learning rate for optimizer (SAME as weighted)
        device: Device to use (if None, will auto-detect)
        patience: Number of epochs to wait for improvement before early stopping (SAME as weighted)
        val_split: Fraction of training data for validation (SAME as weighted)
        epsilon: Minimum improvement to reset early stopping counter (SAME as weighted)
        checkpoint_dir: Directory to save model checkpoints (different from weighted)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with training metrics and confusion matrix
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Training {patch_size}x{patch_size} model with FAIR COMPARISON")
    print(f"Device: {device}")
    print(f"Max epochs: {max_epochs}, Patience: {patience}")
    print(f"⚖️  NO class weighting - FAIR COMPARISON")
    print(f"{'='*60}")

    # Create full training dataset
    full_train_dataset = LifePatchDataset(npz_path, split="train", patch_size=patch_size)
    test_dataset = LifePatchDataset(npz_path, split="test", patch_size=patch_size)

    print(f"Full training samples: {len(full_train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")

    # Count class balance for reference (but don't use for weighting)
    labels = full_train_dataset.y.numpy() if hasattr(full_train_dataset.y, 'numpy') else full_train_dataset.y
    N_pos = np.sum(labels == 1)
    N_neg = np.sum(labels == 0)

    print(f"Class balance (for reference):")
    print(f"  N_pos (class 1): {N_pos:,}")
    print(f"  N_neg (class 0): {N_neg:,}")
    print(f"  Ratio (neg/pos): {N_neg/max(N_pos,1):.4f}")
    print(f"  🚫 No class weighting applied")

    # Create train/validation split (SAME as weighted)
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

    # Loss function WITHOUT class weighting
    criterion = nn.BCEWithLogitsLoss()  # pos_weight = 1.0 (default)

    # Optimizer (SAME as weighted)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping initialization (SAME as weighted)
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    best_epoch = 0

    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print(f"\nStarting training WITHOUT class weighting...")
    print(f"Loss function: BCEWithLogitsLoss(pos_weight=1.0)")

    # Training loop with validation-based early stopping (SAME as weighted)
    for epoch in range(max_epochs):
        # === TRAINING PHASE ===
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.float().to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)  # NO weighting

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            train_loss += loss.item() * features.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        # Calculate training metrics
        avg_train_loss = train_loss / train_total
        train_accuracy = train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # === VALIDATION PHASE ===
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.float().to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)  # NO weighting

                val_loss += loss.item() * features.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        # Calculate validation metrics
        avg_val_loss = val_loss / val_total
        val_accuracy = val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        # === EARLY STOPPING CHECK (SAME as weighted) ===
        # Check if current validation loss is better (accounting for epsilon)
        if avg_val_loss <= best_val_loss + epsilon:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            patience_counter = 0
            improvement_marker = " ✓ NEW BEST"
        else:
            patience_counter += 1
            improvement_marker = f" (patience: {patience_counter}/{patience})"

        # Print progress
        print(f"Epoch {epoch+1:2d}/{max_epochs:2d} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f} | "
              f"Best Val: {best_val_loss:.4f}{improvement_marker}")

        # Check early stopping condition (SAME as weighted)
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            print(f"   Best validation loss: {best_val_loss:.4f} (epoch {best_epoch + 1})")
            break

    # Load best model state for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Loaded best model from epoch {best_epoch + 1}")

    # === FINAL TEST EVALUATION ===
    print(f"\n{'='*50}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*50}")

    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    # Store all predictions and labels for confusion matrix
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

            # Store for confusion matrix calculation
            all_predictions.append(outputs.cpu())
            all_labels.append(labels.cpu())

    # Calculate final test metrics
    avg_test_loss = test_loss / test_total
    test_accuracy = test_correct / test_total

    # Compute confusion matrix
    all_predictions_tensor = torch.cat(all_predictions, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    tn, fp, fn, tp = compute_confusion_entries(all_labels_tensor, all_predictions_tensor)

    # Save best model checkpoint (for fair comparison)
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Create descriptive filename
    config_info = Path(npz_path).stem.replace('life_patches_', '').replace('_seed', '')
    model_filename = f"best_model_patch{patch_size}_{config_info}_epoch{best_epoch+1}_FAIR.pth"
    model_save_path = checkpoint_path / model_filename

    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'patch_size': patch_size,
        'config': config_info,
        'weighted': False,  # Clearly mark as unweighted
        'fair_comparison': True
    }, model_save_path)

    print(f"✅ Model checkpoint saved: {model_save_path}")

    # Return comprehensive results
    return {
        "train_loss": avg_train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": avg_val_loss,
        "val_accuracy": val_accuracy,
        "test_loss": avg_test_loss,
        "test_accuracy": test_accuracy,
        "confusion_matrix": (tn, fp, fn, tp),
        "epochs_trained": best_epoch + 1,
        "best_val_loss": best_val_loss,
        "pos_weight": 1.0,  # Explicitly mark as 1.0 (no weighting)
        "class_balance": {"N_pos": N_pos, "N_neg": N_neg},
        "checkpoint_path": str(model_save_path),
        "weighted": False
    }


def main():
    """Main training function for fair comparison (no weighting)."""
    # Set up data path
    data_path = Path("data/life_patches.npz")
    data_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate dataset if it doesn't exist
    if not data_path.exists():
        print(f"Dataset not found at {data_path}, generating...")
        generate_life_patch_dataset(str(data_path))

    print(f"\n{'='*80}")
    print("🔬 TRAINING MODELS FOR FAIR COMPARISON (NO CLASS WEIGHTING)")
    print(f"{'='*80}")

    # Train both models with fair comparison (same early stopping, no weighting)
    print(f"\n{'='*60}")
    print("Training model with 3x3 neighborhoods (FAIR COMPARISON - NO WEIGHTING)...")
    results_3 = train_one_model_fair_comparison(patch_size=3, npz_path=str(data_path))

    print(f"\n{'='*60}")
    print("Training model with 5x5 neighborhoods (FAIR COMPARISON - NO WEIGHTING)...")
    results_5 = train_one_model_fair_comparison(patch_size=5, npz_path=str(data_path))

    # Print comparison summary
    print(f"\n{'='*80}")
    print("📊 FAIR COMPARISON RESULTS SUMMARY (NO WEIGHTING)")
    print(f"{'='*80}")

    print(f"\n3×3 Model Results (FAIR COMPARISON):")
    print(f"  Test Accuracy: {results_3['test_accuracy']:.4f}")
    print(f"  Test Loss: {results_3['test_loss']:.4f}")
    print(f"  Positive Class Weight: {results_3['pos_weight']:.4f} (NO WEIGHTING)")
    print(f"  Class Balance: N_pos={results_3['class_balance']['N_pos']:,}, "
          f"N_neg={results_3['class_balance']['N_neg']:,}")
    print(f"  Epochs Trained: {results_3['epochs_trained']}")
    print(f"  Best Val Loss: {results_3['best_val_loss']:.4f}")
    print(f"  Confusion Matrix: TN={results_3['confusion_matrix'][0]}, "
          f"FP={results_3['confusion_matrix'][1]}, "
          f"FN={results_3['confusion_matrix'][2]}, "
          f"TP={results_3['confusion_matrix'][3]}")

    print(f"\n5×5 Model Results (FAIR COMPARISON):")
    print(f"  Test Accuracy: {results_5['test_accuracy']:.4f}")
    print(f"  Test Loss: {results_5['test_loss']:.4f}")
    print(f"  Positive Class Weight: {results_5['pos_weight']:.4f} (NO WEIGHTING)")
    print(f"  Class Balance: N_pos={results_5['class_balance']['N_pos']:,}, "
          f"N_neg={results_5['class_balance']['N_neg']:,}")
    print(f"  Epochs Trained: {results_5['epochs_trained']}")
    print(f"  Best Val Loss: {results_5['best_val_loss']:.4f}")
    print(f"  Confusion Matrix: TN={results_5['confusion_matrix'][0]}, "
          f"FP={results_5['confusion_matrix'][1]}, "
          f"FN={results_5['confusion_matrix'][2]}, "
          f"TP={results_5['confusion_matrix'][3]}")

    print(f"\n🔍 Comparison (FAIR COMPARISON):")
    improvement = results_5['test_accuracy'] - results_3['test_accuracy']
    print(f"  Accuracy Improvement (5×5 vs 3×3): {improvement:.4f}")

    # Compute class-specific metrics for comparison
    def compute_metrics(cm):
        tn, fp, fn, tp = cm
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    prec3, rec3, f13 = compute_metrics(results_3['confusion_matrix'])
    prec5, rec5, f15 = compute_metrics(results_5['confusion_matrix'])

    print(f"\n📈 Class-specific Performance (Class 1 - Alive Cells) - NO WEIGHTING:")
    print(f"  3×3: Precision={prec3:.4f}, Recall={rec3:.4f}, F1={f13:.4f}")
    print(f"  5×5: Precision={prec5:.4f}, Recall={rec5:.4f}, F1={f15:.4f}")
    print(f"  Recall Improvement: {rec5 - rec3:.4f}")

    print(f"\n📊 Ready for weighted vs unweighted comparison!")
    print(f"  Next step: Compare these results with train_weighted.py outputs")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()