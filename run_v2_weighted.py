"""Run V2 weighted experiments using existing data.

This script runs all 18 weighted models using the updated weighting scheme:
- w = min(10, ndead/nalive) instead of w = min(5, sqrt(ndead/nalive))

Uses existing data files and saves results to CSV.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import numpy as np
import json
import csv
from datetime import datetime
from typing import Tuple, Dict, Any

from data import LifePatchDataset
from models import MLP
from confusion_matrix_utils import compute_confusion_entries


def compute_class_weights_v2(dataset: LifePatchDataset) -> Tuple[float, float, torch.Tensor]:
    """Compute positive class weight using V2 weighting scheme: w = min(10, ndead/nalive)"""
    # Count positive and negative samples
    labels = dataset.y.numpy() if hasattr(dataset.y, 'numpy') else dataset.y
    N_pos = np.sum(labels == 1)
    N_neg = np.sum(labels == 0)

    # Compute positive class weight using V2 scheme
    r = N_neg / max(N_pos, 1)  # ndead/nalive ratio
    pos_weight = min(10.0, r)   # New cap at 10 instead of sqrt with cap at 5

    # Create tensor for BCEWithLogitsLoss
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)

    return N_pos, N_neg, pos_weight_tensor


def train_v2_model(experiment_config: Dict[str, Any]) -> Dict[str, Any]:
    """Train a single model with V2 weighting."""

    # Extract config
    npz_path = experiment_config["npz_path"]
    patch_size = experiment_config["patch_size"]
    experiment_id = experiment_config["experiment_id"]

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*80}")
    print(f"🧬 Training: {experiment_id}")
    print(f"📊 Weight scheme: w = min(10, ndead/nalive)")
    print(f"📁 Data: {npz_path}")
    print(f"💻 Device: {device}")
    print(f"{'='*80}")

    # Create datasets
    full_train_dataset = LifePatchDataset(npz_path, split="train", patch_size=patch_size)
    test_dataset = LifePatchDataset(npz_path, split="test", patch_size=patch_size)

    print(f"📈 Data info:")
    print(f"  Training samples: {len(full_train_dataset):,}")
    print(f"  Test samples: {len(test_dataset):,}")

    # Compute class weights using V2 scheme
    N_pos, N_neg, pos_weight_tensor = compute_class_weights_v2(full_train_dataset)

    # Create train/validation split
    val_split = 0.2
    total_size = len(full_train_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    torch.manual_seed(42)  # Fixed seed for reproducible splits
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_dim = 8 if patch_size == 3 else 24
    model = MLP(input_dim=input_dim, hidden_dims=[128, 128], dropout=0.0).to(device)

    # Loss function with V2 class weighting
    pos_weight_tensor = pos_weight_tensor.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Early stopping initialization
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 5
    max_epochs = 30
    best_epoch = 0

    print(f"\n🏋️ Training setup:")
    print(f"  Positive class weight: {pos_weight_tensor.item():.4f}")
    print(f"  Class balance - Pos: {N_pos:,}, Neg: {N_neg:,}")
    print(f"  Max epochs: {max_epochs}, Patience: {patience}")

    # Training loop
    for epoch in range(max_epochs):
        # === TRAINING PHASE ===
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, labels in train_loader:
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
        avg_train_loss = train_loss / train_total
        train_accuracy = train_correct / train_total

        # === VALIDATION PHASE ===
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

        # Calculate validation metrics
        avg_val_loss = val_loss / val_total
        val_accuracy = val_correct / val_total

        # === EARLY STOPPING CHECK ===
        if avg_val_loss <= best_val_loss + 1e-4:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            patience_counter = 0
            improvement_marker = " ✓ NEW BEST"
        else:
            patience_counter += 1
            improvement_marker = f" (patience: {patience_counter}/{patience})"

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{max_epochs:2d} | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}{improvement_marker}")

        # Check early stopping condition
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            print(f"   Best validation loss: {best_val_loss:.4f} (epoch {best_epoch + 1})")
            break

    # Load best model state for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # === FINAL TEST EVALUATION ===
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

    # Calculate additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n✅ Final Results:")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Test Loss: {avg_test_loss:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Return comprehensive results
    return {
        "experiment_id": experiment_id,
        "dataset_path": npz_path,
        "patch_size": patch_size,
        "use_weight": True,
        "pos_weight": pos_weight_tensor.item(),
        "accuracy": test_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "total_samples": test_total,
        "train_samples": len(full_train_dataset),
        "test_samples": len(test_dataset),
        "N_pos": str(N_pos),
        "N_neg": str(N_neg),
        "class_ratio": N_neg / max(N_pos, 1),
        "epochs_trained": best_epoch + 1,
        "best_val_loss": best_val_loss,
        "weight_scheme": "v2_min10_ratio"
    }


def main():
    """Main function to run all V2 weighted experiments."""

    print(f"\n{'='*120}")
    print("🧬 CONWAY'S GAME OF LIFE - V2 WEIGHTED EXPERIMENTS")
    print(f"🎯 Weight Scheme: w = min(10, ndead/nalive) (instead of w = min(5, sqrt(ndead/nalive)))")
    print(f"📊 Total Models: 18 (3 densities × 3 temporal stages × 2 patch sizes × 1 weighting scheme)")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*120}")

    # Define all experiment configurations
    experiment_configs = []

    # Densities, temporal stages, patch sizes, and seeds
    densities = ["p0.2", "p0.4", "p0.6"]
    temporal_stages = ["early", "mid", "late"]
    patch_sizes = [3, 5]
    seeds = [0, 1, 2]

    burn_steps = {
        "early": 10,
        "mid": 60,
        "late": 160
    }

    # Generate experiment configs
    for density in densities:
        for temporal_stage in temporal_stages:
            for patch_size in patch_sizes:
                for seed in seeds:
                    npz_path = f"data/life_patches_{temporal_stage}_{density}_burn{burn_steps[temporal_stage]}_steps40_seed{seed}.npz"
                    experiment_id = f"{density}_{temporal_stage}_patch{patch_size}_seed{seed}_weighted_v2"

                    experiment_configs.append({
                        "npz_path": npz_path,
                        "patch_size": patch_size,
                        "experiment_id": experiment_id,
                        "density": density,
                        "temporal_stage": temporal_stage,
                        "seed": seed
                    })

    print(f"🚀 Total experiments to run: {len(experiment_configs)}")

    # Run all experiments
    all_results = []
    successful_experiments = 0

    for i, config in enumerate(experiment_configs, 1):
        print(f"\n📍 Experiment {i}/{len(experiment_configs)}: {config['experiment_id']}")

        try:
            result = train_v2_model(config)
            all_results.append(result)
            successful_experiments += 1
            print(f"✅ COMPLETED: {config['experiment_id']}")
        except Exception as e:
            print(f"❌ FAILED: {config['experiment_id']} - Error: {str(e)}")
            continue

    print(f"\n{'='*120}")
    print(f"🎉 EXPERIMENTS COMPLETED!")
    print(f"📊 Successful experiments: {successful_experiments}/{len(experiment_configs)}")
    print(f"{'='*120}")

    if successful_experiments == 0:
        print("❌ No experiments completed successfully!")
        return

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"results_18_weighted_v2_{timestamp}.csv"
    json_filename = f"results_18_weighted_v2_{timestamp}.json"

    # Prepare CSV data
    csv_data = []
    csv_data.append([
        "Setting", "Weight", "Accuracy", "Precision", "Recall", "F1",
        "TN", "FP", "FN", "TP", "Num_Seeds"
    ])

    # Group results by setting (ignoring seed)
    settings_dict = {}
    for result in all_results:
        # Extract setting key (without seed)
        parts = result["experiment_id"].split('_')
        density = parts[0]
        temporal_stage = parts[1]
        patch_size = parts[3]  # "patch3" or "patch5"
        setting_key = f"{density}_{temporal_stage}_patch{patch_size}"

        if setting_key not in settings_dict:
            settings_dict[setting_key] = []
        settings_dict[setting_key].append(result)

    # Add averaged results to CSV
    for setting_key, setting_results in sorted(settings_dict.items()):
        # Average results across seeds for this setting
        num_seeds = len(setting_results)

        # Calculate averages
        avg_weight = np.mean([r["pos_weight"] for r in setting_results])
        avg_accuracy = np.mean([r["accuracy"] for r in setting_results])
        avg_precision = np.mean([r["precision"] for r in setting_results])
        avg_recall = np.mean([r["recall"] for r in setting_results])
        avg_f1 = np.mean([r["f1"] for r in setting_results])

        # Sum confusion matrices
        total_tn = sum([r["tn"] for r in setting_results])
        total_fp = sum([r["fp"] for r in setting_results])
        total_fn = sum([r["fn"] for r in setting_results])
        total_tp = sum([r["tp"] for r in setting_results])

        # Add averaged row to CSV
        csv_data.append([
            setting_key,
            f"{avg_weight:.3f}",
            f"{avg_accuracy:.4f}",
            f"{avg_precision:.4f}",
            f"{avg_recall:.4f}",
            f"{avg_f1:.4f}",
            str(total_tn),
            str(total_fp),
            str(total_fn),
            str(total_tp),
            str(num_seeds)
        ])

    # Write to CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)

    print(f"\n💾 Results saved to: {csv_filename}")

    # Also save as JSON for detailed analysis
    with open(json_filename, 'w') as jsonfile:
        json.dump(all_results, jsonfile, indent=2)

    print(f"💾 Detailed results saved to: {json_filename}")

    # Print summary
    print(f"\n{'='*120}")
    print("📊 EXPERIMENT SUMMARY")
    print(f"{'='*120}")

    print(f"\n🎯 Performance Summary by Setting:")

    for setting_key, setting_results in sorted(settings_dict.items()):
        # Calculate metrics
        avg_accuracy = np.mean([r["accuracy"] for r in setting_results])
        avg_f1 = np.mean([r["f1"] for r in setting_results])
        avg_recall = np.mean([r["recall"] for r in setting_results])
        avg_precision = np.mean([r["precision"] for r in setting_results])

        print(f"  {setting_key}:")
        print(f"    Accuracy: {avg_accuracy:.4f}, F1: {avg_f1:.4f}")
        print(f"    Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")

    # Find best performing configurations
    best_accuracy_setting = max(settings_dict.items(),
                             key=lambda x: np.mean([r["accuracy"] for r in x[1]]))
    best_f1_setting = max(settings_dict.items(),
                        key=lambda x: np.mean([r["f1"] for r in x[1]]))
    best_recall_setting = max(settings_dict.items(),
                           key=lambda x: np.mean([r["recall"] for r in x[1]]))

    print(f"\n🏆 Best Performing Configurations:")
    print(f"  Best Accuracy: {best_accuracy_setting[0]} "
          f"({np.mean([r['accuracy'] for r in best_accuracy_setting[1]]):.4f})")
    print(f"  Best F1-Score: {best_f1_setting[0]} "
          f"({np.mean([r['f1'] for r in best_f1_setting[1]]):.4f})")
    print(f"  Best Recall: {best_recall_setting[0]} "
          f"({np.mean([r['recall'] for r in best_recall_setting[1]]):.4f})")

    print(f"\n✅ All experiments completed successfully!")
    print(f"📁 Results saved to: {csv_filename}")
    print(f"📁 Detailed results saved to: {json_filename}")
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()