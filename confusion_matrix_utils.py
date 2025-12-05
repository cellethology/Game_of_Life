"""Confusion matrix utilities for Conway's Game of Life MLP prediction."""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import torch

# Try to import optional dependencies
try:
    from sklearn.metrics import confusion_matrix
except ImportError:
    confusion_matrix = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None


def compute_confusion_entries(y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float = 0.5) -> Tuple[int, int, int, int]:
    """
    Compute confusion matrix entries (TN, FP, FN, TP) for binary classification.

    Args:
        y_true: Ground truth labels (tensor of 0s and 1s)
        y_pred: Predicted logits (tensor)
        threshold: Classification threshold (default: 0.5)

    Returns:
        Tuple of (TN, FP, FN, TP) counts
    """
    # Convert logits to binary predictions using threshold
    y_pred_class = (torch.sigmoid(y_pred) > threshold).float()

    # Convert to numpy arrays
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred_class.cpu().numpy()

    # Use sklearn if available, otherwise manual implementation
    if confusion_matrix is not None:
        # sklearn returns [[TN, FP], [FN, TP]] for binary classification
        cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1])

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return int(tn), int(fp), int(fn), int(tp)
        else:
            # Handle edge case where all predictions are the same class
            if len(np.unique(y_true_np)) == 1:
                if y_true_np[0] == 0:  # All negatives
                    tn = len(y_true_np[y_pred_np == 0])
                    fp = len(y_true_np[y_pred_np == 1])
                    return tn, fp, 0, 0
                else:  # All positives
                    fn = len(y_true_np[y_pred_np == 0])
                    tp = len(y_true_np[y_pred_np == 1])
                    return 0, 0, fn, tp
            # Default fallback
            return 0, 0, 0, 0
    else:
        # Manual implementation without sklearn
        tn = fp = fn = tp = 0

        for true_label, pred_label in zip(y_true_np, y_pred_np):
            if true_label == 0 and pred_label == 0:
                tn += 1
            elif true_label == 0 and pred_label == 1:
                fp += 1
            elif true_label == 1 and pred_label == 0:
                fn += 1
            elif true_label == 1 and pred_label == 1:
                tp += 1

        return int(tn), int(fp), int(fn), int(tp)


def plot_confusion_matrix(cm_dict: Dict[str, int], patch_size: int, regime: str, density: float, save_path: str) -> None:
    """
    Plot confusion matrix heatmap and save to file.

    Args:
        cm_dict: Dictionary with keys ['TN', 'FP', 'FN', 'TP']
        patch_size: Size of the patch (3 or 5)
        regime: Time regime ('early', 'mid', or 'late')
        density: Initial cell density (0.2, 0.4, or 0.6)
        save_path: Path to save the figure
    """
    # Check if matplotlib and seaborn are available
    if plt is None or sns is None:
        print(f"⚠️  Matplotlib/Seaborn not available. Skipping confusion matrix plot for patch={patch_size}, regime={regime}, density={density}")
        print(f"   Confusion matrix data: TN={cm_dict['TN']}, FP={cm_dict['FP']}, FN={cm_dict['FN']}, TP={cm_dict['TP']}")
        return

    # Create confusion matrix from aggregated counts
    # Format: [[TN, FP], [FN, TP]]
    cm_matrix = np.array([
        [cm_dict['TN'], cm_dict['FP']],
        [cm_dict['FN'], cm_dict['TP']]
    ])

    # Create figure with appropriate size for PDF
    plt.figure(figsize=(6, 5))

    # Create heatmap with annotations
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'],
                cbar_kws={'label': 'Count'})

    # Add title with parameters
    plt.title(f'Confusion Matrix — patch={patch_size} regime={regime} density={density}',
              fontsize=12, pad=20)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Ensure save directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory

    print(f"Confusion matrix plot saved to {save_path}")


def aggregate_confusion_matrices(cm_list: list) -> Dict[str, int]:
    """
    Aggregate confusion matrices from multiple seeds.

    Args:
        cm_list: List of dictionaries with keys ['TN', 'FP', 'FN', 'TP']

    Returns:
        Aggregated dictionary with summed counts
    """
    if not cm_list:
        return {'TN': 0, 'FP': 0, 'FN': 0, 'TP': 0}

    aggregated = {'TN': 0, 'FP': 0, 'FN': 0, 'TP': 0}

    for cm in cm_list:
        for key in aggregated.keys():
            aggregated[key] += cm.get(key, 0)

    return aggregated


def save_confusion_matrix_to_csv(cm_data: Dict[str, Any], csv_path: str) -> None:
    """
    Save confusion matrix results to CSV file.

    Args:
        cm_data: Dictionary with confusion matrix data and metadata
        csv_path: Path to save the CSV file
    """
    # Ensure CSV directory exists
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists to determine if we need to write header
    file_exists = Path(csv_path).exists()

    # Prepare row data
    row_data = {
        'patch_size': cm_data['patch_size'],
        'regime': cm_data['regime'],
        'density': cm_data['density'],
        'seed': cm_data['seed'],
        'TN': cm_data['TN'],
        'FP': cm_data['FP'],
        'FN': cm_data['FN'],
        'TP': cm_data['TP']
    }

    # Write to CSV (append mode)
    import csv
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['patch_size', 'regime', 'density', 'seed', 'TN', 'FP', 'FN', 'TP']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file is new
        if not file_exists:
            writer.writeheader()

        # Write data row
        writer.writerow(row_data)

    print(f"Confusion matrix data appended to {csv_path}")