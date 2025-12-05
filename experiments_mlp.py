"""Systematic grid experiments for Conway's Game of Life MLP prediction."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

try:
    from .data import generate_life_patch_dataset
    from .train import train_one_model
    from .confusion_matrix_utils import save_confusion_matrix_to_csv, plot_confusion_matrix, aggregate_confusion_matrices
except ImportError:
    from data import generate_life_patch_dataset
    from train import train_one_model
    from confusion_matrix_utils import save_confusion_matrix_to_csv, plot_confusion_matrix, aggregate_confusion_matrices


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
    """Build the experiment grid with all combinations of settings."""

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


def run_one_experiment(config: ExperimentConfig, seed: int) -> List[Dict[str, Any]]:
    """Run a single experiment for both patch sizes."""
    print(f"\n{'='*60}")
    print(f"Running experiment: {config.name} (seed={seed})")
    print(f"p_alive={config.p_alive}, burn_in={config.burn_in}, num_steps={config.num_steps}")

    # Construct dataset path
    out_path = f"data/life_patches_{config.name}_seed{seed}.npz"

    # Generate dataset if needed
    if not Path(out_path).exists():
        print(f"Generating dataset at {out_path}")
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
        print(f"Using existing dataset at {out_path}")

    results = []

    # Train 3x3 model
    print("\nTraining 3x3 model...")
    res3 = train_one_model(patch_size=3, npz_path=out_path)

    # Extract confusion matrix entries
    tn, fp, fn, tp = res3["confusion_matrix"]

    # Save confusion matrix to CSV
    cm_data = {
        "patch_size": 3,
        "regime": config.name.split("_")[0],
        "density": config.p_alive,
        "seed": seed,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }
    save_confusion_matrix_to_csv(cm_data, "results_confusion_matrix.csv")

    results.append({
        "regime": config.name.split("_")[0],
        "p_alive": config.p_alive,
        "burn_in": config.burn_in,
        "num_steps": config.num_steps,
        "seed": seed,
        "patch_size": 3,
        "train_loss": res3["train_loss"],
        "train_accuracy": res3["train_accuracy"],
        "test_loss": res3["test_loss"],
        "test_accuracy": res3["test_accuracy"],
        "confusion_matrix": res3["confusion_matrix"],  # Add confusion matrix
        "epochs_trained": res3["epochs_trained"],    # Add actual epochs trained
    })

    # Train 5x5 model
    print("\nTraining 5x5 model...")
    res5 = train_one_model(patch_size=5, npz_path=out_path)

    # Extract confusion matrix entries
    tn, fp, fn, tp = res5["confusion_matrix"]

    # Save confusion matrix to CSV
    cm_data = {
        "patch_size": 5,
        "regime": config.name.split("_")[0],
        "density": config.p_alive,
        "seed": seed,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }
    save_confusion_matrix_to_csv(cm_data, "results_confusion_matrix.csv")

    results.append({
        "regime": config.name.split("_")[0],
        "p_alive": config.p_alive,
        "burn_in": config.burn_in,
        "num_steps": config.num_steps,
        "seed": seed,
        "patch_size": 5,
        "train_loss": res5["train_loss"],
        "train_accuracy": res5["train_accuracy"],
        "test_loss": res5["test_loss"],
        "test_accuracy": res5["test_accuracy"],
        "confusion_matrix": res5["confusion_matrix"],  # Add confusion matrix
        "epochs_trained": res5["epochs_trained"],    # Add actual epochs trained
    })

    return results


def print_summary_table(results: List[Dict[str, Any]]):
    """Print a nicely formatted summary table."""
    print(f"\n{'='*80}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*80}")

    # Group results by regime and density
    grouped = {}
    for result in results:
        key = (result["regime"], result["p_alive"])
        if key not in grouped:
            grouped[key] = {}
        grouped[key][result["patch_size"]] = result

    # Print table
    print(f"{'Regime':<8} {'p_alive':<8} {'Patch Size':<10} {'Test Acc':<10} {'Train Acc':<10}")
    print("-" * 60)

    for (regime, p_alive) in sorted(grouped.keys()):
        for patch_size in [3, 5]:
            if patch_size in grouped[(regime, p_alive)]:
                result = grouped[(regime, p_alive)][patch_size]
                print(f"{regime:<8} {p_alive:<8.1f} {patch_size}x{patch_size:<6} "
                      f"{result['test_accuracy']:<10.4f} {result['train_accuracy']:<10.4f}")

        # Show improvement
        if 3 in grouped[(regime, p_alive)] and 5 in grouped[(regime, p_alive)]:
            acc3 = grouped[(regime, p_alive)][3]["test_accuracy"]
            acc5 = grouped[(regime, p_alive)][5]["test_accuracy"]
            improvement = acc5 - acc3
            print(f"{'':8} {'':8} {'Improvement':<10} {improvement:<10.4f} {'':<10}")
        print("-" * 60)


def save_results_to_csv(results: List[Dict[str, Any]], output_path: str = "results_mlp_grid.csv"):
    """Save results to a CSV file."""
    if not results:
        return

    # Define column order
    fieldnames = [
        "regime", "p_alive", "burn_in", "num_steps", "seed", "patch_size",
        "train_loss", "train_accuracy", "test_loss", "test_accuracy"
    ]

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")


def run_all_experiments() -> List[Dict[str, Any]]:
    """Run all experiments in the grid."""
    experiments = build_experiment_grid()

    print(f"Running {len(experiments)} experiments...")
    print(f"Total models to train: {len(experiments) * 2} (2 patch sizes per experiment)")

    all_results = []

    for i, (config, seed) in enumerate(experiments, 1):
        print(f"\nProgress: {i}/{len(experiments)} experiments")

        try:
            results = run_one_experiment(config, seed)
            all_results.extend(results)
        except Exception as e:
            print(f"Error in experiment {config.name} (seed={seed}): {e}")
            continue

    return all_results


def generate_confusion_matrix_plots(results: List[Dict[str, Any]]) -> None:
    """
    Generate confusion matrix plots for each (patch_size, regime, density) combination.
    Aggregates across all seeds (0, 1, 2).
    """
    print(f"\n{'='*60}")
    print("GENERATING CONFUSION MATRIX PLOTS")
    print(f"{'='*60}")

    # Create figures directory
    Path("figures").mkdir(exist_ok=True)

    # Group results by patch_size, regime, and density
    grouped = {}
    for result in results:
        key = (result["patch_size"], result["regime"], result["p_alive"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)

    # Generate plot for each combination
    for (patch_size, regime, density), group_results in grouped.items():
        # Aggregate confusion matrices across all seeds
        cm_list = []
        for result in group_results:
            if "confusion_matrix" in result:
                tn, fp, fn, tp = result["confusion_matrix"]
                cm_list.append({"TN": tn, "FP": fp, "FN": fn, "TP": tp})

        if cm_list:
            # Aggregate across seeds
            aggregated_cm = aggregate_confusion_matrices(cm_list)

            # Generate plot
            filename = f"confmat_patch{patch_size}_regime{regime}_density{density:.1f}.png"
            save_path = f"figures/{filename}"

            plot_confusion_matrix(aggregated_cm, patch_size, regime, density, save_path)

    print(f"Confusion matrix plots generated for {len(grouped)} combinations")


def main():
    """Main function to run all experiments and save results."""
    # Create data directory
    Path("data").mkdir(exist_ok=True)

    # Run all experiments
    results = run_all_experiments()

    # Print summary
    print_summary_table(results)

    # Save to CSV
    save_results_to_csv(results)

    # Generate confusion matrix plots
    generate_confusion_matrix_plots(results)

    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"Total results: {len(results)}")
    print(f"Confusion matrix CSV: results_confusion_matrix.csv")
    print(f"Confusion matrix plots: figures/ directory")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()