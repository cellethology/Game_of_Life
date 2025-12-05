"""Analyze the experimental results for Conway's Game of Life MLP experiments."""

import csv
import numpy as np
from pathlib import Path

def analyze_results(csv_path="results_mlp_grid.csv"):
    """Analyze experimental results and create summary statistics."""

    # Load results
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in ['p_alive', 'burn_in', 'num_steps', 'seed', 'patch_size',
                       'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy']:
                row[key] = float(row[key])
            results.append(row)

    print("=" * 80)
    print("CONWAY'S GAME OF LIFE MLP PREDICTION - COMPREHENSIVE RESULTS")
    print("=" * 80)

    # Group results by configuration
    grouped = {}
    for result in results:
        key = (result["regime"], result["p_alive"])
        if key not in grouped:
            grouped[key] = {"3x3": [], "5x5": []}

        if result["patch_size"] == 3:
            grouped[key]["3x3"].append(result)
        else:
            grouped[key]["5x5"].append(result)

    # Calculate summary statistics
    print("\n1. DETAILED RESULTS BY CONFIGURATION")
    print("-" * 80)

    for (regime, p_alive) in sorted(grouped.keys()):
        print(f"\n{regime.upper()} REGIME (p_alive={p_alive:.1f}):")

        # Calculate means and stds for each patch size
        for patch_size in ["3x3", "5x5"]:
            if grouped[(regime, p_alive)][patch_size]:
                test_accs = [r["test_accuracy"] for r in grouped[(regime, p_alive)][patch_size]]
                mean_acc = np.mean(test_accs)
                std_acc = np.std(test_accs)

                print(f"  {patch_size}: {mean_acc:.4f} ± {std_acc:.4f} (mean ± std)")

        # Calculate improvement
        if grouped[(regime, p_alive)]["3x3"] and grouped[(regime, p_alive)]["5x5"]:
            acc3 = [r["test_accuracy"] for r in grouped[(regime, p_alive)]["3x3"]]
            acc5 = [r["test_accuracy"] for r in grouped[(regime, p_alive)]["5x5"]]
            improvements = [a5 - a3 for a3, a5 in zip(acc3, acc5)]
            mean_improvement = np.mean(improvements)
            std_improvement = np.std(improvements)
            print(f"  IMPROVEMENT: {mean_improvement:.4f} ± {std_improvement:.4f}")

    print("\n\n2. OVERALL TRENDS ANALYSIS")
    print("-" * 80)

    # Overall comparison by regime
    regimes = ["early", "mid", "late"]
    print("\nPerformance by Time Regime:")
    for regime in regimes:
        regime_results_3x3 = []
        regime_results_5x5 = []

        for (r, p_alive) in grouped.keys():
            if r == regime:
                regime_results_3x3.extend([res["test_accuracy"] for res in grouped[(r, p_alive)]["3x3"]])
                regime_results_5x5.extend([res["test_accuracy"] for res in grouped[(r, p_alive)]["5x5"]])

        if regime_results_3x3 and regime_results_5x5:
            print(f"  {regime.upper()}:")
            print(f"    3x3: {np.mean(regime_results_3x3):.4f} ± {np.std(regime_results_3x3):.4f}")
            print(f"    5x5: {np.mean(regime_results_5x5):.4f} ± {np.std(regime_results_5x5):.4f}")
            print(f"    Improvement: {np.mean(regime_results_5x5) - np.mean(regime_results_3x3):.4f}")

    # Overall comparison by density
    densities = [0.2, 0.4, 0.6]
    print("\nPerformance by Initial Density:")
    for density in densities:
        density_results_3x3 = []
        density_results_5x5 = []

        for (regime, p_alive) in grouped.keys():
            if p_alive == density:
                density_results_3x3.extend([res["test_accuracy"] for res in grouped[(regime, p_alive)]["3x3"]])
                density_results_5x5.extend([res["test_accuracy"] for res in grouped[(regime, p_alive)]["5x5"]])

        if density_results_3x3 and density_results_5x5:
            print(f"  p_alive={density:.1f}:")
            print(f"    3x3: {np.mean(density_results_3x3):.4f} ± {np.std(density_results_3x3):.4f}")
            print(f"    5x5: {np.mean(density_results_5x5):.4f} ± {np.std(density_results_5x5):.4f}")
            print(f"    Improvement: {np.mean(density_results_5x5) - np.mean(density_results_3x3):.4f}")

    print("\n\n3. KEY INSIGHTS")
    print("-" * 80)

    # Find best and worst performing configurations
    all_improvements = []
    for (regime, p_alive) in grouped.keys():
        if grouped[(regime, p_alive)]["3x3"] and grouped[(regime, p_alive)]["5x5"]:
            acc3 = np.mean([r["test_accuracy"] for r in grouped[(regime, p_alive)]["3x3"]])
            acc5 = np.mean([r["test_accuracy"] for r in grouped[(regime, p_alive)]["5x5"]])
            improvement = acc5 - acc3
            all_improvements.append((regime, p_alive, improvement, acc3, acc5))

    # Sort by improvement
    all_improvements.sort(key=lambda x: x[2], reverse=True)

    print("\nTop 5 Configurations by 5x5-3x3 Improvement:")
    for i, (regime, p_alive, improvement, acc3, acc5) in enumerate(all_improvements[:5]):
        print(f"  {i+1}. {regime} (p_alive={p_alive:.1f}): +{improvement:.4f} "
              f"(3x3={acc3:.4f} → 5x5={acc5:.4f})")

    # Overall statistics
    all_3x3 = [r["test_accuracy"] for r in results if r["patch_size"] == 3]
    all_5x5 = [r["test_accuracy"] for r in results if r["patch_size"] == 5]

    print(f"\n\n4. OVERALL SUMMARY")
    print("-" * 80)
    print(f"Total experiments: {len(all_3x3)} configurations × 2 patch sizes × 3 seeds = {len(results)} models")
    print(f"\nOverall Performance:")
    print(f"  3×3 Neighborhood: {np.mean(all_3x3):.4f} ± {np.std(all_3x3):.4f}")
    print(f"  5×5 Neighborhood: {np.mean(all_5x5):.4f} ± {np.std(all_5x5):.4f}")
    print(f"  Average Improvement: {np.mean(all_5x5) - np.mean(all_3x3):.4f}")
    print(f"  Consistency: {np.std([a5-a3 for a3, a5 in zip(all_3x3, all_5x5)]):.4f} std of improvements")

    return grouped, all_improvements

if __name__ == "__main__":
    analyze_results()