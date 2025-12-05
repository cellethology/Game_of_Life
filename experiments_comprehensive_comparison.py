"""
Comprehensive Comparison Experiments for Conway's Game of Life MLP Prediction.

This script runs ALL experiment configurations with BOTH weighted and unweighted training
to provide complete comparison of class weighting effects across different settings.

For each configuration (regime × density × seed), it trains:
1. Weighted model (using train_weighted methodology)
2. Unweighted model (using train_fair_comparison methodology)

Results are saved to separate CSV files and summarized for analysis.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

try:
    from .data import generate_life_patch_dataset
    from .train_weighted import train_one_model_weighted
    from .train_fair_comparison import train_one_model_fair_comparison
    from .confusion_matrix_utils import save_confusion_matrix_to_csv, compute_confusion_entries
except ImportError:
    from data import generate_life_patch_dataset
    from train_weighted import train_one_model_weighted
    from train_fair_comparison import train_one_model_fair_comparison
    from confusion_matrix_utils import save_confusion_matrix_to_csv, compute_confusion_entries


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    p_alive: float
    burn_in: int
    num_steps: int
    board_size: int = 128
    num_train_boards: int = 16
    num_test_boards: int = 4
    num_patches_per_step: int = 200


def build_experiment_grid() -> List[tuple]:
    """Build the complete experiment grid with all combinations of settings."""
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


def compute_metrics_from_confusion(tn: int, fp: int, fn: int, tp: int) -> Dict[str, float]:
    """
    Compute comprehensive metrics from confusion matrix entries.

    Args:
        tn, fp, fn, tp: Confusion matrix counts

    Returns:
        Dictionary with accuracy, precision, recall, f1, specificity
    """
    total = tn + fp + fn + tp

    # Basic metrics
    accuracy = (tn + tp) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # False positive and false negative rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "total": total
    }


def run_comprehensive_comparison(config: ExperimentConfig, seed: int) -> List[Dict[str, Any]]:
    """
    Run comprehensive comparison for a single configuration.
    Trains both weighted and unweighted models and records all metrics.

    Args:
        config: Experiment configuration
        seed: Random seed

    Returns:
        List of results for both patch sizes and weighting methods
    """
    print(f"\n{'='*80}")
    print(f"🧪 COMPREHENSIVE COMPARISON: {config.name} (seed={seed})")
    print(f"   p_alive={config.p_alive}, burn_in={config.burn_in}, num_steps={config.num_steps}")
    print(f"   Training WEIGHTED and UNWEIGHTED models for both patch sizes")
    print(f"{'='*80}")

    # Construct dataset path (same for both weighted and unweighted)
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

    # === TRAIN UNWEIGHTED 3×3 MODEL ===
    print(f"\n🎯 Training UNWEIGHTED 3×3 model...")
    print(f"   {'─'*70}")

    res_3_unweighted = train_one_model_fair_comparison(
        patch_size=3,
        npz_path=out_path,
        batch_size=1024,
        max_epochs=30,
        learning_rate=1e-3,
        patience=5,
        val_split=0.2,
        epsilon=1e-4,
        checkpoint_dir="checkpoints_comprehensive_unweighted",
        seed=seed
    )

    # Process unweighted 3×3 results
    tn, fp, fn, tp = res_3_unweighted["confusion_matrix"]
    metrics_3_unweighted = compute_metrics_from_confusion(tn, fp, fn, tp)

    result = {
        "regime": config.name.split("_")[0],
        "density": config.p_alive,
        "burn_in": config.burn_in,
        "num_steps": config.num_steps,
        "seed": seed,
        "patch_size": 3,
        "weighted": False,
        "train_loss": res_3_unweighted["train_loss"],
        "train_accuracy": res_3_unweighted["train_accuracy"],
        "val_loss": res_3_unweighted["val_loss"],
        "val_accuracy": res_3_unweighted["val_accuracy"],
        "test_loss": res_3_unweighted["test_loss"],
        "test_accuracy": res_3_unweighted["test_accuracy"],
        "epochs_trained": res_3_unweighted["epochs_trained"],
        "pos_weight": res_3_unweighted["pos_weight"],
        "class_balance": res_3_unweighted["class_balance"],
        "best_val_loss": res_3_unweighted["best_val_loss"],
        **metrics_3_unweighted
    }

    results.append(result)

    # Save to UNWEIGHTED CSV
    cm_data = {
        "patch_size": 3,
        "regime": config.name.split("_")[0],
        "density": config.p_alive,
        "seed": seed,
        "weighted": False,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }
    save_confusion_matrix_to_csv(cm_data, "results_confusion_matrix_comprehensive.csv")

    print(f"✅ UNWEIGHTED 3×3: Acc={metrics_3_unweighted['accuracy']:.4f}, "
          f"Recall={metrics_3_unweighted['recall']:.4f}")

    # === TRAIN WEIGHTED 3×3 MODEL ===
    print(f"\n⚖️  Training WEIGHTED 3×3 model...")
    print(f"   {'─'*70}")

    res_3_weighted = train_one_model_weighted(
        patch_size=3,
        npz_path=out_path,
        batch_size=1024,
        max_epochs=30,
        learning_rate=1e-3,
        patience=5,
        val_split=0.2,
        epsilon=1e-4,
        checkpoint_dir="checkpoints_comprehensive_weighted",
        seed=seed
    )

    # Process weighted 3×3 results
    tn, fp, fn, tp = res_3_weighted["confusion_matrix"]
    metrics_3_weighted = compute_metrics_from_confusion(tn, fp, fn, tp)

    result = {
        "regime": config.name.split("_")[0],
        "density": config.p_alive,
        "burn_in": config.burn_in,
        "num_steps": config.num_steps,
        "seed": seed,
        "patch_size": 3,
        "weighted": True,
        "train_loss": res_3_weighted["train_loss"],
        "train_accuracy": res_3_weighted["train_accuracy"],
        "val_loss": res_3_weighted["val_loss"],
        "val_accuracy": res_3_weighted["val_accuracy"],
        "test_loss": res_3_weighted["test_loss"],
        "test_accuracy": res_3_weighted["test_accuracy"],
        "epochs_trained": res_3_weighted["epochs_trained"],
        "pos_weight": res_3_weighted["pos_weight"],
        "class_balance": res_3_weighted["class_balance"],
        "best_val_loss": res_3_weighted["best_val_loss"],
        **metrics_3_weighted
    }

    results.append(result)

    # Save to WEIGHTED CSV
    cm_data = {
        "patch_size": 3,
        "regime": config.name.split("_")[0],
        "density": config.p_alive,
        "seed": seed,
        "weighted": True,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }
    save_confusion_matrix_to_csv(cm_data, "results_confusion_matrix_comprehensive.csv")

    print(f"✅ WEIGHTED 3×3: Acc={metrics_3_weighted['accuracy']:.4f}, "
          f"Recall={metrics_3_weighted['recall']:.4f}")

    # Calculate improvement for 3×3
    recall_improvement_3 = metrics_3_weighted['recall'] - metrics_3_unweighted['recall']
    acc_improvement_3 = metrics_3_weighted['accuracy'] - metrics_3_unweighted['accuracy']

    print(f"🔍 3×3 IMPROVEMENT: Acc {acc_improvement_3:+.4f}, Recall {recall_improvement_3:+.4f}")

    # === TRAIN UNWEIGHTED 5×5 MODEL ===
    print(f"\n🎯 Training UNWEIGHTED 5×5 model...")
    print(f"   {'─'*70}")

    res_5_unweighted = train_one_model_fair_comparison(
        patch_size=5,
        npz_path=out_path,
        batch_size=1024,
        max_epochs=30,
        learning_rate=1e-3,
        patience=5,
        val_split=0.2,
        epsilon=1e-4,
        checkpoint_dir="checkpoints_comprehensive_unweighted",
        seed=seed
    )

    # Process unweighted 5×5 results
    tn, fp, fn, tp = res_5_unweighted["confusion_matrix"]
    metrics_5_unweighted = compute_metrics_from_confusion(tn, fp, fn, tp)

    result = {
        "regime": config.name.split("_")[0],
        "density": config.p_alive,
        "burn_in": config.burn_in,
        "num_steps": config.num_steps,
        "seed": seed,
        "patch_size": 5,
        "weighted": False,
        "train_loss": res_5_unweighted["train_loss"],
        "train_accuracy": res_5_unweighted["train_accuracy"],
        "val_loss": res_5_unweighted["val_loss"],
        "val_accuracy": res_5_unweighted["val_accuracy"],
        "test_loss": res_5_unweighted["test_loss"],
        "test_accuracy": res_5_unweighted["test_accuracy"],
        "epochs_trained": res_5_unweighted["epochs_trained"],
        "pos_weight": res_5_unweighted["pos_weight"],
        "class_balance": res_5_unweighted["class_balance"],
        "best_val_loss": res_5_unweighted["best_val_loss"],
        **metrics_5_unweighted
    }

    results.append(result)

    # Save to UNWEIGHTED CSV
    cm_data = {
        "patch_size": 5,
        "regime": config.name.split("_")[0],
        "density": config.p_alive,
        "seed": seed,
        "weighted": False,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }
    save_confusion_matrix_to_csv(cm_data, "results_confusion_matrix_comprehensive.csv")

    print(f"✅ UNWEIGHTED 5×5: Acc={metrics_5_unweighted['accuracy']:.4f}, "
          f"Recall={metrics_5_unweighted['recall']:.4f}")

    # === TRAIN WEIGHTED 5×5 MODEL ===
    print(f"\n⚖️  Training WEIGHTED 5×5 model...")
    print(f"   {'─'*70}")

    res_5_weighted = train_one_model_weighted(
        patch_size=5,
        npz_path=out_path,
        batch_size=1024,
        max_epochs=30,
        learning_rate=1e-3,
        patience=5,
        val_split=0.2,
        epsilon=1e-4,
        checkpoint_dir="checkpoints_comprehensive_weighted",
        seed=seed
    )

    # Process weighted 5×5 results
    tn, fp, fn, tp = res_5_weighted["confusion_matrix"]
    metrics_5_weighted = compute_metrics_from_confusion(tn, fp, fn, tp)

    result = {
        "regime": config.name.split("_")[0],
        "density": config.p_alive,
        "burn_in": config.burn_in,
        "num_steps": config.num_steps,
        "seed": seed,
        "patch_size": 5,
        "weighted": True,
        "train_loss": res_5_weighted["train_loss"],
        "train_accuracy": res_5_weighted["train_accuracy"],
        "val_loss": res_5_weighted["val_loss"],
        "val_accuracy": res_5_weighted["val_accuracy"],
        "test_loss": res_5_weighted["test_loss"],
        "test_accuracy": res_5_weighted["test_accuracy"],
        "epochs_trained": res_5_weighted["epochs_trained"],
        "pos_weight": res_5_weighted["pos_weight"],
        "class_balance": res_5_weighted["class_balance"],
        "best_val_loss": res_5_weighted["best_val_loss"],
        **metrics_5_weighted
    }

    results.append(result)

    # Save to WEIGHTED CSV
    cm_data = {
        "patch_size": 5,
        "regime": config.name.split("_")[0],
        "density": config.p_alive,
        "seed": seed,
        "weighted": True,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }
    save_confusion_matrix_to_csv(cm_data, "results_confusion_matrix_comprehensive.csv")

    print(f"✅ WEIGHTED 5×5: Acc={metrics_5_weighted['accuracy']:.4f}, "
          f"Recall={metrics_5_weighted['recall']:.4f}")

    # Calculate improvement for 5×5
    recall_improvement_5 = metrics_5_weighted['recall'] - metrics_5_unweighted['recall']
    acc_improvement_5 = metrics_5_weighted['accuracy'] - metrics_5_unweighted['accuracy']

    print(f"🔍 5×5 IMPROVEMENT: Acc {acc_improvement_5:+.4f}, Recall {recall_improvement_5:+.4f}")

    # Print comprehensive summary for this configuration
    print(f"\n📊 COMPREHENSIVE SUMMARY for {config.name} (seed={seed}):")
    print(f"   3×3: Unweighted Acc={metrics_3_unweighted['accuracy']:.4f}/Recall={metrics_3_unweighted['recall']:.4f} → "
          f"Weighted Acc={metrics_3_weighted['accuracy']:.4f}/Recall={metrics_3_weighted['recall']:.4f}")
    print(f"        ΔAcc={acc_improvement_3:+.4f}, ΔRecall={recall_improvement_3:+.4f}")
    print(f"   5×5: Unweighted Acc={metrics_5_unweighted['accuracy']:.4f}/Recall={metrics_5_unweighted['recall']:.4f} → "
          f"Weighted Acc={metrics_5_weighted['accuracy']:.4f}/Recall={metrics_5_weighted['recall']:.4f}")
    print(f"        ΔAcc={acc_improvement_5:+.4f}, ΔRecall={recall_improvement_5:+.4f}")

    return results


def print_comprehensive_summary(results: List[Dict[str, Any]]):
    """Print comprehensive summary table for all experiments."""
    print(f"\n{'='*120}")
    print("📊 COMPREHENSIVE EXPERIMENTS SUMMARY")
    print(f"{'='*120}")

    # Group results by configuration
    grouped = {}
    for result in results:
        key = (result["regime"], result["density"], result["seed"])
        if key not in grouped:
            grouped[key] = {}
        weight_key = "weighted" if result["weighted"] else "unweighted"
        grouped[key][weight_key] = result

    # Print detailed table
    print(f"{'Regime':<8} {'Density':<8} {'Seed':<5} {'Model':<15} {'Accuracy':<10} {'Recall':<10} {'Precision':<10} {'F1':<10}")
    print("-" * 90)

    total_configs = 0
    total_weighted_improvements = []
    total_unweighted_improvements = []

    for (regime, density, seed) in sorted(grouped.keys()):
        config_group = grouped[(regime, density, seed)]

        if "unweighted" in config_group and "weighted" in config_group:
            unweighted = config_group["unweighted"]
            weighted = config_group["weighted"]

            total_configs += 1

            # 3×3 comparison
            if unweighted["patch_size"] == 3 and weighted["patch_size"] == 3:
                acc_change = weighted["accuracy"] - unweighted["accuracy"]
                recall_change = weighted["recall"] - unweighted["recall"]

                print(f"{regime:<8} {density:<8.1f} {seed:<5} "
                      f"{'3×3 Unweighted':<15} {unweighted['accuracy']:<10.4f} {unweighted['recall']:<10.4f} {unweighted['precision']:<10.4f} {unweighted['f1']:<10.4f}")
                print(f"{'':8} {'':8.1f} {'':5} {'3×3 Weighted  ':<15} {weighted['accuracy']:<10.4f} {weighted['recall']:<10.4f} {weighted['precision']:<10.4f} {weighted['f1']:<10.4f}")
                print(f"{'':8} {'':8.1f} {'':5} {'3×3 Δ         ':<15} {acc_change:+10.4f} {recall_change:+10.4f} {weighted['precision']-unweighted['precision']:+10.4f} {weighted['f1']-unweighted['f1']:+10.4f}")

                total_weighted_improvements.append(recall_change)

            # 5×5 comparison
            elif unweighted["patch_size"] == 5 and weighted["patch_size"] == 5:
                acc_change = weighted["accuracy"] - unweighted["accuracy"]
                recall_change = weighted["recall"] - unweighted["recall"]

                print(f"{regime:<8} {density:<8.1f} {seed:<5} "
                      f"{'5×5 Unweighted':<15} {unweighted['accuracy']:<10.4f} {unweighted['recall']:<10.4f} {unweighted['precision']:<10.4f} {unweighted['f1']:<10.4f}")
                print(f"{'':8} {'':8.1f} {'':5} {'5×5 Weighted  ':<15} {weighted['accuracy']:<10.4f} {weighted['recall']:<10.4f} {weighted['precision']:<10.4f} {weighted['f1']:<10.4f}")
                print(f"{'':8} {'':8.1f} {'':5} {'5×5 Δ         ':<15} {acc_change:+10.4f} {recall_change:+10.4f} {weighted['precision']-unweighted['precision']:+10.4f} {weighted['f1']-unweighted['f1']:+10.4f}")

                total_weighted_improvements.append(recall_change)

        print("-" * 90)

    # Print overall statistics
    if total_weighted_improvements:
        avg_recall_improvement = np.mean(total_weighted_improvements)
        std_recall_improvement = np.std(total_weighted_improvements)

        print(f"\n📈 OVERALL WEIGHTING EFFECTS:")
        print(f"   Total configurations compared: {total_configs}")
        print(f"   Average recall improvement: {avg_recall_improvement:+.4f} ± {std_recall_improvement:.4f}")
        print(f"   Positive improvements: {sum(1 for x in total_weighted_improvements if x > 0)}/{len(total_weighted_improvements)}")
        print(f"   Negative improvements: {sum(1 for x in total_weighted_improvements if x < 0)}/{len(total_weighted_improvements)}")


def save_comprehensive_results_to_csv(results: List[Dict[str, Any]], output_path: str = "results_comprehensive_comparison.csv"):
    """Save comprehensive results to CSV file."""
    if not results:
        return

    # Define comprehensive column order
    fieldnames = [
        "regime", "density", "burn_in", "num_steps", "seed", "patch_size", "weighted",
        "train_loss", "train_accuracy", "val_loss", "val_accuracy",
        "test_loss", "test_accuracy", "epochs_trained", "best_val_loss",
        "pos_weight", "accuracy", "precision", "recall", "specificity", "f1",
        "fpr", "fnr", "tn", "fp", "fn", "tp", "total"
    ]

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Comprehensive results saved to {output_path}")


def run_all_comprehensive_experiments() -> List[Dict[str, Any]]:
    """Run all comprehensive experiments with both weighted and unweighted training."""
    experiments = build_experiment_grid()

    print(f"\n{'='*120}")
    print(f"🚀 RUNNING COMPREHENSIVE COMPARISON EXPERIMENTS")
    print(f"{'='*120}")
    print(f"Total configurations: {len(experiments)}")
    print(f"Models per configuration: 4 (2 patch sizes × 2 weighting methods)")
    print(f"Total models to train: {len(experiments) * 4}")
    print(f"Each configuration trains: Unweighted 3×3, Weighted 3×3, Unweighted 5×5, Weighted 5×5")
    print(f"{'='*120}")

    all_results = []

    for i, (config, seed) in enumerate(experiments, 1):
        print(f"\n📍 Progress: {i}/{len(experiments)} experiments ({i/len(experiments)*100:.1f}%)")

        try:
            results = run_comprehensive_comparison(config, seed)
            all_results.extend(results)

            print(f"✅ Configuration {config.name} (seed={seed}) completed successfully!")

        except Exception as e:
            print(f"❌ Error in comprehensive experiment {config.name} (seed={seed}): {e}")
            import traceback
            traceback.print_exc()
            continue

    return all_results


def main():
    """Main function to run comprehensive comparison experiments."""
    # Create necessary directories
    Path("data").mkdir(exist_ok=True)
    Path("checkpoints_comprehensive_unweighted").mkdir(exist_ok=True)
    Path("checkpoints_comprehensive_weighted").mkdir(exist_ok=True)

    print("="*120)
    print("🔬 CONWAY'S GAME OF LIFE - COMPREHENSIVE WEIGHTED vs UNWEIGHTED COMPARISON")
    print("="*120)
    print("Features:")
    print("  • ALL experiment configurations (regimes × densities × seeds)")
    print("  • BOTH weighting methods (weighted vs unweighted)")
    print("  • BOTH patch sizes (3×3 vs 5×5)")
    print("  • IDENTICAL training procedures except for class weighting")
    print("  • Comprehensive metrics tracking (accuracy, recall, precision, F1, etc.)")
    print("  • Detailed confusion matrix recording for each configuration")
    print("="*120)

    # Run all comprehensive experiments
    results = run_all_comprehensive_experiments()

    if not results:
        print("❌ No comprehensive experiments completed successfully!")
        return

    # Print comprehensive summary
    print_comprehensive_summary(results)

    # Save comprehensive results
    save_comprehensive_results_to_csv(results)

    print(f"\n{'='*120}")
    print("🎉 ALL COMPREHENSIVE EXPERIMENTS COMPLETED!")
    print(f"{'='*120}")
    print(f"Results Summary:")
    print(f"  • Total models trained: {len(results)}")
    print(f"  • Comprehensive confusion matrices: results_confusion_matrix_comprehensive.csv")
    print(f"  • Detailed metrics: results_comprehensive_comparison.csv")
    print(f"  • Checkpoints:")
    print(f"    - Unweighted models: checkpoints_comprehensive_unweighted/")
    print(f"    - Weighted models: checkpoints_comprehensive_weighted/")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()