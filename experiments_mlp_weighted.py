"""Weighted systematic grid experiments for Conway's Game of Life MLP prediction.

This script extends the original experiments_mlp.py to use:
- Class-weighted BCE loss to address class imbalance
- Validation-based early stopping for better model selection
- Separate checkpointing directory for weighted models
- New confusion matrix CSV for weighted results

The experimental design remains identical to the original.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

try:
    from .data import generate_life_patch_dataset
    from .train_weighted import train_one_model_weighted
    from .confusion_matrix_utils import save_confusion_matrix_to_csv, plot_confusion_matrix, aggregate_confusion_matrices
except ImportError:
    from data import generate_life_patch_dataset
    from train_weighted import train_one_model_weighted
    from confusion_matrix_utils import save_confusion_matrix_to_csv, plot_confusion_matrix, aggregate_confusion_matrices


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment (unchanged from original)."""
    name: str
    p_alive: float
    burn_in: int
    num_steps: int
    board_size: int = 128
    num_train_boards: int = 16
    num_test_boards: int = 4
    num_patches_per_step: int = 200


def build_experiment_grid() -> List[tuple]:
    """Build the experiment grid with all combinations of settings (unchanged)."""
    # Define regimes with (name, burn_in, num_steps)
    regimes = [
        ("early", 10, 40),
        ("mid", 60, 40),
        ("late", 160, 40)
    ]

    # Define densities
    densities = [0.2, 0.4, 0.6]

    # Define seeds
    seeds = [0, 1, 2]

    experiments = []

    for regime_name, burn_in, num_steps in regimes:
        for p_alive in densities:
            config_name = f"{regime_name}_p{p_alive:.1f}_burn{burn_in}_steps{num_steps}"
            config = ExperimentConfig(
                name=config_name,
                p_alive=p_alive,
                burn_in=burn_in,
                num_steps=num_steps
            )

            for seed in seeds:
                experiments.append((config, seed))

    return experiments


def run_one_weighted_experiment(config: ExperimentConfig, seed: int) -> List[Dict[str, Any]]:
    """
    Run a single experiment with weighted loss for both patch sizes.

    Args:
        config: Experiment configuration
        seed: Random seed

    Returns:
        List of results for both patch sizes (3 and 5)
    """
    print(f"\n{'='*60}")
    print(f"🔬 WEIGHTED EXPERIMENT: {config.name} (seed={seed})")
    print(f"   p_alive={config.p_alive}, burn_in={config.burn_in}, num_steps={config.num_steps}")
    print(f"{'='*60}")

    # Construct dataset path (same as original)
    out_path = f"data/life_patches_{config.name}_seed{seed}.npz"

    # Generate dataset if needed
    if not Path(out_path).exists():
        print(f"📊 Generating dataset at {out_path}")
        generate_life_patch_dataset(
            out_path=out_path,
            board_size=config.board_size,
            num_train_boards=config.num_train_boards,
            num_test_boards=config.num_test_boards,
            num_steps=config.num_steps,
            burn_in=config.burn_in,
            p_alive=config.p_alive,
            num_patches_per_step=config.num_patches_per_step,
            seed=seed,
        )
    else:
        print(f"✅ Using existing dataset at {out_path}")

    results = []

    # === TRAIN 3×3 MODEL (WEIGHTED) ===
    print(f"\n🎯 Training 3×3 model with class weighting...")
    print(f"   {'─'*50}")

    res3 = train_one_model_weighted(
        patch_size=3,
        npz_path=out_path,
        batch_size=1024,
        max_epochs=30,          # Increased from 10
        learning_rate=1e-3,
        patience=5,            # Early stopping patience
        val_split=0.2,         # Validation split
        epsilon=1e-4,          # Improvement threshold
        checkpoint_dir="checkpoints_weighted",
        seed=seed
    )

    # Extract confusion matrix entries
    tn, fp, fn, tp = res3["confusion_matrix"]

    # Save confusion matrix to NEW CSV file
    cm_data = {
        "patch_size": 3,
        "regime": config.name.split("_")[0],
        "density": config.p_alive,
        "seed": seed,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }
    save_confusion_matrix_to_csv(cm_data, "results_confusion_matrix_weighted.csv")

    # Build comprehensive result entry
    results.append({
        "regime": config.name.split("_")[0],
        "p_alive": config.p_alive,
        "burn_in": config.burn_in,
        "num_steps": config.num_steps,
        "seed": seed,
        "patch_size": 3,
        "train_loss": res3["train_loss"],
        "train_accuracy": res3["train_accuracy"],
        "val_loss": res3["val_loss"],      # NEW: validation loss
        "val_accuracy": res3["val_accuracy"],  # NEW: validation accuracy
        "test_loss": res3["test_loss"],
        "test_accuracy": res3["test_accuracy"],
        "confusion_matrix": res3["confusion_matrix"],
        "epochs_trained": res3["epochs_trained"],
        "best_val_loss": res3["best_val_loss"],  # NEW: best validation loss
        "pos_weight": res3["pos_weight"],        # NEW: positive class weight
        "class_balance": res3["class_balance"],  # NEW: class balance info
        "checkpoint_path": res3["checkpoint_path"]  # NEW: model checkpoint
    })

    # === TRAIN 5×5 MODEL (WEIGHTED) ===
    print(f"\n🎯 Training 5×5 model with class weighting...")
    print(f"   {'─'*50}")

    res5 = train_one_model_weighted(
        patch_size=5,
        npz_path=out_path,
        batch_size=1024,
        max_epochs=30,          # Increased from 10
        learning_rate=1e-3,
        patience=5,            # Early stopping patience
        val_split=0.2,         # Validation split
        epsilon=1e-4,          # Improvement threshold
        checkpoint_dir="checkpoints_weighted",
        seed=seed
    )

    # Extract confusion matrix entries
    tn, fp, fn, tp = res5["confusion_matrix"]

    # Save confusion matrix to NEW CSV file
    cm_data = {
        "patch_size": 5,
        "regime": config.name.split("_")[0],
        "density": config.p_alive,
        "seed": seed,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }
    save_confusion_matrix_to_csv(cm_data, "results_confusion_matrix_weighted.csv")

    # Build comprehensive result entry
    results.append({
        "regime": config.name.split("_")[0],
        "p_alive": config.p_alive,
        "burn_in": config.burn_in,
        "num_steps": config.num_steps,
        "seed": seed,
        "patch_size": 5,
        "train_loss": res5["train_loss"],
        "train_accuracy": res5["train_accuracy"],
        "val_loss": res5["val_loss"],      # NEW: validation loss
        "val_accuracy": res5["val_accuracy"],  # NEW: validation accuracy
        "test_loss": res5["test_loss"],
        "test_accuracy": res5["test_accuracy"],
        "confusion_matrix": res5["confusion_matrix"],
        "epochs_trained": res5["epochs_trained"],
        "best_val_loss": res5["best_val_loss"],  # NEW: best validation loss
        "pos_weight": res5["pos_weight"],        # NEW: positive class weight
        "class_balance": res5["class_balance"],  # NEW: class balance info
        "checkpoint_path": res5["checkpoint_path"]  # NEW: model checkpoint
    })

    # Print weighted results summary for this experiment
    print(f"\n📊 WEIGHTED RESULTS SUMMARY for {config.name} (seed={seed}):")
    print(f"   3×3: Test Acc={res3['test_accuracy']:.4f}, "
          f"Pos Weight={res3['pos_weight']:.4f}, "
          f"Epochs={res3['epochs_trained']}")
    print(f"   5×5: Test Acc={res5['test_accuracy']:.4f}, "
          f"Pos Weight={res5['pos_weight']:.4f}, "
          f"Epochs={res5['epochs_trained']}")
    improvement = res5['test_accuracy'] - res3['test_accuracy']
    print(f"   🔍 Improvement (5×5 vs 3×3): {improvement:+.4f}")

    return results


def print_weighted_summary_table(results: List[Dict[str, Any]]):
    """Print a nicely formatted summary table for weighted experiments."""
    print(f"\n{'='*100}")
    print("📊 WEIGHTED EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*100}")

    # Group results by regime and density
    grouped = {}
    for result in results:
        key = (result["regime"], result["p_alive"])
        if key not in grouped:
            grouped[key] = {}
        grouped[key][result["patch_size"]] = result

    # Extended table with weighted metrics
    print(f"{'Regime':<8} {'p_alive':<8} {'Patch Size':<10} {'Test Acc':<10} {'Train Acc':<10} {'Pos Weight':<11} {'Epochs':<7}")
    print("-" * 80)

    for (regime, p_alive) in sorted(grouped.keys()):
        for patch_size in [3, 5]:
            if patch_size in grouped[(regime, p_alive)]:
                result = grouped[(regime, p_alive)][patch_size]
                print(f"{regime:<8} {p_alive:<8.1f} {patch_size}x{patch_size:<6} "
                      f"{result['test_accuracy']:<10.4f} {result['train_accuracy']:<10.4f} "
                      f"{result['pos_weight']:<11.4f} {result['epochs_trained']:<7}")

        # Show improvement if both patch sizes available
        if 3 in grouped[(regime, p_alive)] and 5 in grouped[(regime, p_alive)]:
            acc3 = grouped[(regime, p_alive)][3]["test_accuracy"]
            acc5 = grouped[(regime, p_alive)][5]["test_accuracy"]
            improvement = acc5 - acc3
            print(f"{'':8} {'':8} {'Improvement':<10} {improvement:<10.4f} {'':<21} {'':<7}")
        print("-" * 80)

    # Class balance statistics
    print(f"\n📈 CLASS BALANCE STATISTICS:")
    all_pos_weights_3 = [r["pos_weight"] for r in results if r["patch_size"] == 3]
    all_pos_weights_5 = [r["pos_weight"] for r in results if r["patch_size"] == 5]

    if all_pos_weights_3:
        print(f"   3×3 Pos Weights: mean={np.mean(all_pos_weights_3):.4f}, "
              f"std={np.std(all_pos_weights_3):.4f}, min={np.min(all_pos_weights_3):.4f}, max={np.max(all_pos_weights_3):.4f}")
    if all_pos_weights_5:
        print(f"   5×5 Pos Weights: mean={np.mean(all_pos_weights_5):.4f}, "
              f"std={np.std(all_pos_weights_5):.4f}, min={np.min(all_pos_weights_5):.4f}, max={np.max(all_pos_weights_5):.4f}")


def save_weighted_results_to_csv(results: List[Dict[str, Any]], output_path: str = "results_mlp_grid_weighted.csv"):
    """Save weighted results to a CSV file with extended metrics."""
    if not results:
        return

    # Extended column order with weighted metrics
    fieldnames = [
        "regime", "p_alive", "burn_in", "num_steps", "seed", "patch_size",
        "train_loss", "train_accuracy", "val_loss", "val_accuracy",  # NEW
        "test_loss", "test_accuracy", "epochs_trained", "best_val_loss",  # NEW
        "pos_weight", "checkpoint_path"  # NEW
    ]

    # Prepare data for CSV (flatten nested structures)
    csv_results = []
    for result in results:
        csv_result = {key: result[key] for key in fieldnames if key in result}

        # Skip complex nested fields for CSV simplicity
        csv_results.append(csv_result)

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_results)

    print(f"\n✅ Weighted results saved to {output_path}")


def run_all_weighted_experiments() -> List[Dict[str, Any]]:
    """Run all experiments in the grid using the weighted approach."""
    experiments = build_experiment_grid()

    print(f"\n{'='*80}")
    print(f"🚀 RUNNING ALL WEIGHTED EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Total models to train: {len(experiments) * 2} (2 patch sizes per experiment)")
    print(f"Using class-weighted loss with validation-based early stopping")
    print(f"Checkpoints saved to: checkpoints_weighted/")
    print(f"Confusion matrices saved to: results_confusion_matrix_weighted.csv")

    all_results = []

    for i, (config, seed) in enumerate(experiments, 1):
        print(f"\n📍 Progress: {i}/{len(experiments)} experiments")

        try:
            results = run_one_weighted_experiment(config, seed)
            all_results.extend(results)
        except Exception as e:
            print(f"❌ Error in weighted experiment {config.name} (seed={seed}): {e}")
            import traceback
            traceback.print_exc()
            continue

    return all_results


def generate_weighted_confusion_matrix_plots(results: List[Dict[str, Any]]) -> None:
    """
    Generate confusion matrix plots for weighted models.
    Aggregates across all seeds (0, 1, 2).

    Args:
        results: List of weighted experiment results
    """
    print(f"\n{'='*80}")
    print("📊 GENERATING WEIGHTED CONFUSION MATRIX PLOTS")
    print(f"{'='*80}")

    # Create figures directory for weighted plots
    figures_dir = Path("figures_weighted")
    figures_dir.mkdir(exist_ok=True)

    # Group results by patch_size, regime, and density
    grouped = {}
    for result in results:
        key = (result["patch_size"], result["regime"], result["p_alive"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)

    print(f"Generating confusion matrix plots for {len(grouped)} unique configurations...")

    # Generate plot for each combination
    for (patch_size, regime, density), group_results in sorted(grouped.items()):
        # Aggregate confusion matrices across all seeds
        cm_list = []
        for result in group_results:
            if "confusion_matrix" in result:
                tn, fp, fn, tp = result["confusion_matrix"]
                cm_list.append({"TN": tn, "FP": fp, "FN": fn, "TP": tp})

        if cm_list:
            # Aggregate across seeds
            aggregated_cm = aggregate_confusion_matrices(cm_list)

            # Generate plot with weighted designation
            filename = f"confmat_patch{patch_size}_regime{regime}_density{density:.1f}_weighted.png"
            save_path = figures_dir / filename

            plot_confusion_matrix(aggregated_cm, patch_size, regime, density, str(save_path))

            # Calculate and print class-specific metrics
            tn, fp, fn, tp = aggregated_cm['TN'], aggregated_cm['FP'], aggregated_cm['FN'], aggregated_cm['TP']
            total = tn + fp + fn + tp
            accuracy = (tn + tp) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"   Generated: {filename}")
            print(f"     Metrics: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")

    print(f"✅ Weighted confusion matrix plots saved to: {figures_dir}")


def main():
    """Main function to run all weighted experiments and save results."""
    import numpy as np  # Import here for statistical calculations

    # Create data and checkpoint directories
    Path("data").mkdir(exist_ok=True)
    Path("checkpoints_weighted").mkdir(exist_ok=True)

    print(f"\n{'='*100}")
    print("🔬 CONWAY'S GAME OF LIFE - WEIGHTED MLP EXPERIMENTS")
    print(f"{'='*100}")
    print("Features:")
    print("  • Class-weighted BCE loss to address imbalance (0 >> 1)")
    print("  • Increased epochs (30) with validation-based early stopping")
    print("  • Positive class weight: min(5.0, sqrt(N_neg/N_pos))")
    print("  • Separate checkpoints for weighted models")
    print("  • New confusion matrix CSV for weighted results")
    print(f"{'='*100}")

    # Run all weighted experiments
    results = run_all_weighted_experiments()

    if not results:
        print("❌ No weighted experiments completed successfully!")
        return

    # Print weighted summary
    print_weighted_summary_table(results)

    # Save weighted results to CSV
    save_weighted_results_to_csv(results)

    # Generate weighted confusion matrix plots
    generate_weighted_confusion_matrix_plots(results)

    print(f"\n{'='*100}")
    print("🎉 ALL WEIGHTED EXPERIMENTS COMPLETED!")
    print(f"{'='*100}")
    print(f"Results Summary:")
    print(f"  • Total weighted models trained: {len(results)}")
    print(f"  • Weighted results CSV: results_mlp_grid_weighted.csv")
    print(f"  • Weighted confusion matrices: results_confusion_matrix_weighted.csv")
    print(f"  • Weighted model checkpoints: checkpoints_weighted/")
    print(f"  • Weighted confusion matrix plots: figures_weighted/")
    print(f"{'='*100}")

    # Calculate and display final statistics
    test_accuracies_3 = [r["test_accuracy"] for r in results if r["patch_size"] == 3]
    test_accuracies_5 = [r["test_accuracy"] for r in results if r["patch_size"] == 5]

    if test_accuracies_3 and test_accuracies_5:
        print(f"\n📈 FINAL WEIGHTED PERFORMANCE COMPARISON:")
        print(f"  3×3 Models: {np.mean(test_accuracies_3):.4f} ± {np.std(test_accuracies_3):.4f}")
        print(f"  5×5 Models: {np.mean(test_accuracies_5):.4f} ± {np.std(test_accuracies_5):.4f}")
        print(f"  Average improvement: {np.mean(test_accuracies_5) - np.mean(test_accuracies_3):+.4f}")


if __name__ == "__main__":
    main()