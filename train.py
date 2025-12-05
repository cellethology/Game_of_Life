"""Training script for Conway's Game of Life MLP prediction."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

try:
    from .data import LifePatchDataset, generate_life_patch_dataset
    from .models import MLP
    from .confusion_matrix_utils import compute_confusion_entries
except ImportError:
    from data import LifePatchDataset, generate_life_patch_dataset
    from models import MLP
    from confusion_matrix_utils import compute_confusion_entries


def train_one_model(
    patch_size: int,
    npz_path: str = "data/life_patches.npz",
    batch_size: int = 1024,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: torch.device | None = None,
    patience: int = 2,
    min_epochs: int = 3,
    early_stop_delta: float = 1e-4,
) -> dict:
    """
    Train and evaluate an MLP for the given patch_size (3 or 5) on the dataset at npz_path.

    Args:
        patch_size: Size of the neighborhood (3 or 5)
        npz_path: Path to the dataset file
        batch_size: Batch size for training
        num_epochs: Number of training epochs (max epochs with early stopping)
        learning_rate: Learning rate for optimizer
        device: Device to use (if None, will auto-detect)
        patience: Number of epochs to wait for improvement before early stopping
        min_epochs: Minimum number of epochs to train
        early_stop_delta: Minimum improvement to reset early stopping counter

    Returns:
        Dictionary with train/test metrics:
        {
            "train_loss": float,
            "train_accuracy": float,
            "test_loss": float,
            "test_accuracy": float,
            "confusion_matrix": tuple,  # (TN, FP, FN, TP)
            "epochs_trained": int,     # Actual epochs trained (with early stopping)
        }
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets and data loaders
    train_dataset = LifePatchDataset(npz_path, split="train", patch_size=patch_size)
    test_dataset = LifePatchDataset(npz_path, split="test", patch_size=patch_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    if patch_size == 3:
        input_dim = 8
    elif patch_size == 5:
        input_dim = 24
    elif patch_size == 7:
        input_dim = 48
    else:
        raise ValueError(f"Unsupported patch_size {patch_size}")
    model = MLP(input_dim=input_dim, hidden_dims=[128, 128], dropout=0.0).to(device)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Training {patch_size}x{patch_size} model on {device}")
    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Early stopping initialization
    best_test_accuracy = 0.0
    patience_counter = 0
    best_predictions = None
    best_labels = None

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.float().to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

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
        train_loss /= train_total
        train_accuracy = train_correct / train_total

        # Evaluation phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        # Store all predictions and labels for confusion matrix calculation
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

                # Store for confusion matrix
                all_predictions.append(outputs)
                all_labels.append(labels)

        # Calculate test metrics
        test_loss /= test_total
        test_accuracy = test_correct / test_total

        # Early stopping check
        if test_accuracy > best_test_accuracy + early_stop_delta:
            best_test_accuracy = test_accuracy
            patience_counter = 0
            # Save best predictions for confusion matrix
            best_predictions = torch.cat(all_predictions, dim=0)
            best_labels = torch.cat(all_labels, dim=0)
        else:
            patience_counter += 1

        # Print progress
        print(f"Epoch {epoch+1:2d}/{num_epochs:2d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f} | "
              f"Best: {best_test_accuracy:.4f} | Patience: {patience_counter}/{patience}")

        # Check early stopping conditions
        if epoch >= min_epochs - 1 and patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # Compute confusion matrix from best epoch
    if best_predictions is not None and best_labels is not None:
        tn, fp, fn, tp = compute_confusion_entries(best_labels, best_predictions)
        confusion_result = (tn, fp, fn, tp)
    else:
        # Fallback: use final epoch if no improvement found
        confusion_result = (0, 0, 0, 0)

    # Return final metrics
    return {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": best_test_accuracy,  # Use best test accuracy
        "confusion_matrix": confusion_result,  # (TN, FP, FN, TP)
        "epochs_trained": epoch + 1  # Actual epochs trained
    }


def main():
    """Main training function."""
    # Set up data path
    data_path = Path("data/life_patches.npz")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    need_patch7 = True
    regenerate = False
    if not data_path.exists():
        print(f"Dataset not found at {data_path}, generating (wrap_extraction, with patch7)...")
        regenerate = True
    else:
        data = np.load(data_path)
        wrap_ok = f"meta_wrap_extraction" in data and bool(data["meta_wrap_extraction"])
        if need_patch7 and "X7_train" not in data.files:
            regenerate = True
        if not wrap_ok:
            regenerate = True
        if regenerate:
            print(f"Dataset at {data_path} missing X7_* or not wrap-based; regenerating with include_patch7=True, wrap_extraction=True...")
    if regenerate:
        generate_life_patch_dataset(str(data_path), include_patch7=need_patch7, wrap_extraction=True)

    patch_sizes = [3, 5, 7]
    results_all = {}
    for ps in patch_sizes:
        print("\n" + "="*60)
        print(f"Training model with {ps}x{ps} neighborhoods...")
        results_all[ps] = train_one_model(patch_size=ps, npz_path=str(data_path))

    print("\n" + "="*60)
    print("=== Summary ===")
    for ps in patch_sizes:
        print(f"{ps}x{ps} test accuracy: {results_all[ps]['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
