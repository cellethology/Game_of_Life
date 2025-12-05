"""
Systematic comparison of weighted vs unweighted MLP training across different density settings.

This script:
1. Generates datasets for three density levels (p_alive = 0.2, 0.4, 0.6)
2. Trains both weighted and unweighted models for each density and patch size
3. Records confusion matrices and metrics
4. Saves results to CSV and generates comparison plots
"""

import csv
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

try:
    from .data import LifePatchDataset, generate_life_patch_dataset
    from .models import MLP
    from .confusion_matrix_utils import compute_confusion_entries, save_confusion_matrix_to_csv
    from .confusion_matrix_visualization import create_grid_plot, create_individual_plot
except ImportError:
    from data import LifePatchDataset, generate_life_patch_dataset
    from models import MLP
    from confusion_matrix_utils import compute_confusion_entries, save_confusion_matrix_to_csv
    from confusion_matrix_visualization import create_grid_plot, create_individual_plot


def compute_class_weight(N_pos: int, N_neg: int, option: int = 2) -> float:
    """
    Compute class weight for weighted training.

    Args:
        N_pos: Number of positive samples
        N_neg: Number of negative samples
        option: Weight strategy option (1=original, 2=conservative, 3=aggressive)

    Returns:
        Positive class weight for BCEWithLogitsLoss
    """
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


def train_one_model(
    dataset_path: str,
    patch_size: int,
    weighted: bool = True,
    weight_option: int = 2,
    batch_size: int = 1024,
    max_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: torch.device = None,
    seed: int = 42
) -> dict:
    """
    Train a single model with specified weighting strategy.

    Args:
        dataset_path: Path to dataset
        patch_size: Size of neighborhood (3 or 5)
        weighted: Whether to use class weighting
        weight_option: Weight strategy option
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        learning_rate: Learning rate
        device: Training device (auto-detect if None)
        seed: Random seed

    Returns:
        Dictionary with training metrics and confusion matrix
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\\n{'='*60}")
    print(f"Training {'weighted' if weighted else 'unweighted'} {patch_size}×{patch_size} model")
    print(f"Dataset: {dataset_path}")
    print(f"Device: {device}")
    print(f"Weighted: {weighted}")
    if weighted:
        print(f"Weight option: {weight_option}")
    print(f"{'='*60}")

    # Create datasets
    train_dataset = LifePatchDataset(dataset_path, split='train', patch_size=patch_size)
    test_dataset = LifePatchDataset(dataset_path, split='test', patch_size=patch_size)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Compute class weights if weighted
    if weighted:
        labels = train_dataset.y.numpy()
        N_pos = np.sum(labels == 1)
        N_neg = np.sum(labels == 0)
        pos_weight = compute_class_weight(N_pos, N_neg, weight_option)
        print(f"Class balance: N_pos={N_pos}, N_neg={N_neg}")
        print(f"Weight computation: r={N_neg/max(N_pos,1):.2f}, pos_weight={pos_weight:.4f}")
    else:
        pos_weight = 1.0
        print("Unweighted training (pos_weight=1.0)")

    # Initialize model
    input_dim = 8 if patch_size == 3 else 24
    model = MLP(input_dim=input_dim, hidden_dims=[128, 128], dropout=0.0).to(device)

    # Loss function
    if weighted:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
        )
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3
    epsilon = 1e-4
    best_epoch = 0

    print(f"Training parameters: epochs={max_epochs}, lr={learning_rate}, patience={patience}")

    start_time = time.time()

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

        train_accuracy = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in DataLoader(train_dataset, batch_size=batch_size, shuffle=False):
                features, labels = features.to(device), labels.float().to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * features.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / val_total

        print(f"Epoch {epoch+1:2d}/{max_epochs:2d} | "
              f"Train Loss: {train_loss/train_total:.4f} | Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss/val_total:.4f} | Val Acc: {val_accuracy:.4f}")

        # Early stopping check
        if val_loss <= best_val_loss + epsilon:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            improvement_marker = " ✓ NEW BEST"
        else:
            patience_counter += 1
            improvement_marker = f" (patience: {patience_counter}/{patience})"

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch+1})")
            break

    training_time = time.time() - start_time

    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model from epoch {best_epoch+1}")

    # Final test evaluation
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

    test_accuracy = test_correct / test_total

    # Compute confusion matrix
    all_predictions_tensor = torch.cat(all_predictions, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    tn, fp, fn, tp = compute_confusion_entries(all_labels_tensor, all_predictions_tensor)

    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Final test evaluation:")
    print(f"  Test Loss: {test_loss/test_total:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Test Precision: {precision:.4f}")
    print(f"  Test Recall: {recall:.4f}")
    print(f"  Test F1: {f1:.4f}")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  Training time: {training_time:.1f}s")

    return {
        "patch_size": patch_size,
        "weighted": weighted,
        "weight_option": weight_option if weighted else None,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": (tn, fp, fn, tp),
        "epochs_trained": best_epoch + 1,
        "training_time": training_time
    }


def generate_datasets():
    """Generate datasets for three density levels."""
    densities = [0.2, 0.4, 0.6]
    base_path = "data/life_patches"

    print("Generating datasets for three density levels...")

    for p_alive in densities:
        for seed in [0, 1, 2]:
            dataset_path = f"{base_path}_p{p_alive:.1f}_seed{seed}.npz"

            if not Path(dataset_path).exists():
                print(f"Generating {dataset_path}...")
                generate_life_patch_dataset(
                    out_path=dataset_path,
                    board_size=128,
                    num_train_boards=16,
                    num_test_boards=4,
                    num_steps=60,
                    burn_in=50,
                    p_alive=p_alive,
                    num_patches_per_step=200,
                    seed=seed
                )
            else:
                print(f"Using existing {dataset_path}")

    print("Dataset generation completed!")


def run_experiments():
    """Run systematic experiments across densities, patch sizes, and weighting methods."""
    print("Starting systematic experiments...")

    # Configuration
    densities = [0.2, 0.4, 0.6]
    patch_sizes = [3, 5]
    weight_options = [None, 2]  # None for unweighted, 2 for conservative weighting
    seeds = [0, 1, 2]

    # Create output directories
    Path("results_weighted_unweighted").mkdir(exist_ok=True)
    Path("checkpoints_unweighted").mkdir(exist_ok=True)
    Path("checkpoints_weighted").mkdir(exist_ok=True)
    Path("figures_weighted_unweighted").mkdir(exist_ok=True)

    all_results = []

    total_experiments = len(densities) * len(patch_sizes) * len(weight_options) * len(seeds)
    completed_experiments = 0

    for density in densities:
        for seed in seeds:
            for patch_size in patch_sizes:
                for weighted in weight_options:
                    completed_experiments += 1
                    exp_id = f"p{density:.1f}_s{seed}_patch{patch_size}_{'_weighted' if weighted else '_unweighted'}"

                    print(f"\\n{'='*80}")
                    print(f"Experiment {completed_experiments}/{total_experiments} ({exp_id})")
                    print(f"Density: {density}, Seed: {seed}, Patch: {patch_size}×{patch_size}, Weighted: {weighted}")

                    try:
                        # Determine dataset path
                        dataset_path = f"data/life_patches_p{density:.1f}_s{seed}.npz"

                        # Train model
                        result = train_one_model(
                            dataset_path=dataset_path,
                            patch_size=patch_size,
                            weighted=weighted,
                            weight_option=2,
                            max_epochs=15,
                            seed=seed
                        )

                        # Add experiment info
                        result.update({
                            "density": density,
                            "seed": seed,
                            "patch_size": patch_size,
                            "weighted": weighted,
                            "weight_option": 2 if weighted else None
                        })

                        all_results.append(result)

                        # Save confusion matrix to CSV
                        cm_data = {
                            "density": density,
                            "seed": seed,
                            "patch_size": patch_size,
                            "weighted": weighted,
                            "weight_option": 2 if weighted else None,
                            "TN": result["confusion_matrix"][0],
                            "FP": result["confusion_matrix"][1],
                            "FN": result["confusion_matrix"][2],
                            "TP": result["confusion_matrix"][3],
                            "test_accuracy": result["test_accuracy"],
                            "precision": result["precision"],
                            "recall": result["recall"],
                            "f1": result["f1"]
                        }
                        save_confusion_matrix_to_csv(cm_data, "results_weighted_unweighted.csv")

                        print(f"Saved confusion matrix to CSV")

                    except Exception as e:
                        print(f"Error in {exp_id}: {e}")
                        import traceback
                        traceback.print_exc()

    # Save detailed results
    with open("results_weighted_unweighted_detailed.csv", "w", newline="") as csvfile:
        fieldnames = [
            "density", "seed", "patch_size", "weighted", "weight_option",
            "train_accuracy", "val_accuracy", "test_accuracy",
            "precision", "recall", "f1", "epochs_trained", "confusion_matrix_TN", "confusion_matrix_FP", "confusion_matrix_FN", "confusion_matrix_TP"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\\nExperiments completed: {completed_experiments}/{total_experiments}")
    print(f"Results saved to: results_weighted_unweighted.csv and results_weighted_unweighted_detailed.csv")

    return all_results


def create_comparison_plots():
    """Create comparison plots from experiment results."""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Cannot create plots - missing dependencies (pandas, matplotlib, seaborn)")
        return

    # Load results
    df = pd.read_csv("results_weighted_unweighted.csv")

    print("Creating comparison plots...")

    # Create comparison plots for each patch size
    for patch_size in [3, 5]:
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'Weighted vs Unweighted Comparison - {patch_size}×{patch_size}', fontsize=16)

        for i, density in enumerate([0.2, 0.4, 0.6]):
            for j, weighted in enumerate([False, True]):
                data = df[(df['density'] == density) & (df['patch_size'] == patch_size) & (df['weighted'] == weighted)]

                if len(data) == 0:
                    print(f"No data for density={density}, weighted={weighted}, patch_size={patch_size}")
                    continue

                # Get average across seeds
                metrics = data.groupby(['weighted']).get_group(data[['test_accuracy', 'precision', 'recall', 'f1']].mean().to_dict())

                ax = axes[i, j]
                metric_values = metrics[weighted] if weighted else metrics[not weighted]

                # Create bar plot
                x = list(metrics.keys())
                y = list(metric_values.values())

                bars = ax.bar(x, y, color=['lightcoral', 'steelblue'][1 if weighted else 0])
                ax.set_title(f'{"Weighted" if weighted else "Unweighted"} - Density {density}', fontsize=12)
                ax.set_xlabel('Metric')
                ax.set_ylabel('Score')
                ax.set_ylim(0, 1)

                # Add value labels on bars
                for bar, value in zip(bars, y):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{value:.3f}',
                            ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(f"figures_weighted_unweighted/patch{patch_size}_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    print("Comparison plots saved to figures_weighted_unweighted/")


def print_summary():
    """Print experiment summary statistics."""
    try:
        import pandas as pd
    except ImportError:
        print("Cannot print summary - pandas not available")
        return

    df = pd.read_csv("results_weighted_unweighted.csv")

    print("\\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    # Overall statistics
    total_experiments = len(df)
    print(f"Total experiments: {total_experiments}")
    print(f"Densities tested: {sorted(df['density'].unique())}")
    print(f"Patch sizes: {sorted(df['patch_size'].unique())}")
    print(f"Weighting methods: {sorted(df['weighted'].unique())}")

    # Grouped statistics
    grouped = df.groupby(['density', 'patch_size', 'weighted'])

    print("\\nAccuracy by configuration:")
    for (density, patch_size), group in grouped.groups:
        unweighted = group[grouped['weighted'] == False]['test_accuracy'].mean()
        weighted = group[grouped['weighted'] == True]['test_accuracy'].mean()
        improvement = weighted - unweighted

        print(f"  Density {density}, Patch {patch_size}×{patch_size}: "
              f"Unweighted={unweighted:.4f}, Weighted={weighted:.4f}, Improvement={improvement:+.4f}")

    # Recall improvement analysis
    print("\\nRecall improvement analysis:")
    for density in df['density'].unique():
        density_data = df[df['density'] == density]

        unweighted_recall = density_data[density_data['weighted'] == False]['recall'].mean()
        weighted_recall = density_data[density_data['weighted'] == True]['recall'].mean()
        improvement = weighted_recall - unweighted_recall

        print(f"  Density {density}: Unweighted={unweighted_recall:.4f}, "
              f"Weighted={weighted_recall:.4f}, Improvement={improvement:+.4f}")

    # Best configurations
    print("\\nBest performing configurations:")
    best_overall = df.loc[df['test_accuracy'].idxmax()]
    print(f"Best accuracy: {best_overall['density']}, Patch {best_overall['patch_size']}, Weighted {best_overall['weighted']}, Accuracy {best_overall['test_accuracy']:.4f}")

    best_weighted = df.loc[df[df['recall'].idxmax()]
    print(f"Best recall: {best_weighted['density']}, Patch {best_weighted['patch_size']}, Weighted {best_weighted['weighted']}, Recall {best_weighted['recall']:.4f}")


def main():
    """Main function to run the systematic weighted vs unweighted experiments."""
    print("="*80)
    print("WEIGHTED VS UNWEIGHTED COMPARISON EXPERIMENTS")
    print("="*80)
    print("This script will:")
    print("1. Generate datasets for p_alive = 0.2, 0.4, 0.6")
    print("2. Train both weighted and unweighted models for 3×3 and 5×5")
    print("3. Train with 3 seeds for each configuration")
    print("4. Record confusion matrices and metrics")
    print("5. Generate comparison plots and summary")
    print("="*80)

    # Create necessary directories
    Path("data").mkdir(exist_ok=True)
    Path("results_weighted_unweighted").mkdir(exist_ok=True)
    Path("checkpoints_unweighted").mkdir(exist_ok=True)
    Path("checkpoints_weighted").mkdir(exist_ok=True)
    Path("figures_weighted_unweighted").mkdir(exist_ok=True)

    try:
        # Generate datasets if needed
        generate_datasets()

        # Run all experiments
        all_results = run_experiments()

        if all_results:
            # Create comparison plots
            create_comparison_plots()

            # Print summary
            print_summary()
        else:
            print("No experiments completed successfully!")

    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()