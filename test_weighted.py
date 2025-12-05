"""
Test script for weighted training - run a few representative configurations
to verify the weighted training pipeline works correctly before running full experiments.

This script tests 3 representative configurations:
1. Early regime with low density (class imbalance likely severe)
2. Mid regime with medium density (balanced case)
3. Late regime with high density (complex dynamics)

This gives us confidence that the weighted training works across different scenarios.
"""

import sys
from pathlib import Path
from dataclasses import dataclass

try:
    from .data import generate_life_patch_dataset
    from .train_weighted import train_one_model_weighted
    from .confusion_matrix_utils import save_confusion_matrix_to_csv
except ImportError:
    from data import generate_life_patch_dataset
    from train_weighted import train_one_model_weighted
    from confusion_matrix_utils import save_confusion_matrix_to_csv


@dataclass
class TestConfig:
    """Configuration for test experiments."""
    name: str
    p_alive: float
    burn_in: int
    num_steps: int


def run_test_experiment(config: TestConfig, seed: int = 0):
    """Run a single test experiment for both patch sizes."""
    print(f"\n{'='*80}")
    print(f"🧪 TEST EXPERIMENT: {config.name}")
    print(f"   p_alive={config.p_alive}, burn_in={config.burn_in}, num_steps={config.num_steps}")
    print(f"{'='*80}")

    # Construct dataset path
    out_path = f"data/life_patches_test_{config.name}_seed{seed}.npz"

    # Generate dataset if needed
    if not Path(out_path).exists():
        print(f"📊 Generating test dataset at {out_path}")
        generate_life_patch_dataset(
            out_path=out_path,
            board_size=128,
            num_train_boards=8,   # Reduced for testing
            num_test_boards=2,    # Reduced for testing
            num_steps=config.num_steps,
            burn_in=config.burn_in,
            p_alive=config.p_alive,
            num_patches_per_step=100,  # Reduced for testing
            seed=seed,
        )
    else:
        print(f"✅ Using existing test dataset at {out_path}")

    print(f"\n🎯 Training 3×3 model (weighted)...")
    print(f"   {'─'*60}")

    # Train 3×3 model
    res3 = train_one_model_weighted(
        patch_size=3,
        npz_path=out_path,
        batch_size=512,          # Smaller batch for testing
        max_epochs=15,           # Reduced epochs for testing
        learning_rate=1e-3,
        patience=3,              # Reduced patience for testing
        val_split=0.2,
        epsilon=1e-4,
        checkpoint_dir="checkpoints_test",
        seed=seed
    )

    print(f"\n🎯 Training 5×5 model (weighted)...")
    print(f"   {'─'*60}")

    # Train 5×5 model
    res5 = train_one_model_weighted(
        patch_size=5,
        npz_path=out_path,
        batch_size=512,          # Smaller batch for testing
        max_epochs=15,           # Reduced epochs for testing
        learning_rate=1e-3,
        patience=3,              # Reduced patience for testing
        val_split=0.2,
        epsilon=1e-4,
        checkpoint_dir="checkpoints_test",
        seed=seed
    )

    # Save confusion matrices to test CSV
    tn, fp, fn, tp = res3["confusion_matrix"]
    cm_data = {
        "patch_size": 3,
        "regime": config.name.split("_")[0],
        "density": config.p_alive,
        "seed": seed,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }
    save_confusion_matrix_to_csv(cm_data, "results_confusion_matrix_test.csv")

    tn, fp, fn, tp = res5["confusion_matrix"]
    cm_data = {
        "patch_size": 5,
        "regime": config.name.split("_")[0],
        "density": config.p_alive,
        "seed": seed,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }
    save_confusion_matrix_to_csv(cm_data, "results_confusion_matrix_test.csv")

    # Print results summary
    print(f"\n📊 TEST RESULTS SUMMARY for {config.name}:")
    print(f"   3×3: Acc={res3['test_accuracy']:.4f}, PosWeight={res3['pos_weight']:.4f}, "
          f"Epochs={res3['epochs_trained']}, ValLoss={res3['best_val_loss']:.4f}")
    print(f"   5×5: Acc={res5['test_accuracy']:.4f}, PosWeight={res5['pos_weight']:.4f}, "
          f"Epochs={res5['epochs_trained']}, ValLoss={res5['best_val_loss']:.4f}")

    improvement = res5['test_accuracy'] - res3['test_accuracy']
    print(f"   🔍 Improvement: {improvement:+.4f}")

    # Class-specific metrics
    def compute_metrics(cm):
        tn, fp, fn, tp = cm
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    prec3, rec3, f13 = compute_metrics(res3['confusion_matrix'])
    prec5, rec5, f15 = compute_metrics(res5['confusion_matrix'])

    print(f"   🎯 Class 1 (Alive) Metrics:")
    print(f"      3×3: Precision={prec3:.4f}, Recall={rec3:.4f}, F1={f13:.4f}")
    print(f"      5×5: Precision={prec5:.4f}, Recall={rec5:.4f}, F1={f15:.4f}")
    print(f"      Recall Improvement: {rec5 - rec3:+.4f}")

    return {
        "config": config.name,
        "patch3": res3,
        "patch5": res5,
        "improvement": improvement,
        "recall_improvement": rec5 - rec3
    }


def main():
    """Run test experiments on representative configurations."""
    print("="*80)
    print("🧪 WEIGHTED TRAINING PIPELINE TEST")
    print("="*80)
    print("Testing weighted training on 3 representative configurations:")
    print("1. Early regime with low density (severe class imbalance)")
    print("2. Mid regime with medium density (balanced case)")
    print("3. Late regime with high density (complex dynamics)")
    print("="*80)

    # Create test directories
    Path("data").mkdir(exist_ok=True)
    Path("checkpoints_test").mkdir(exist_ok=True)

    # Representative test configurations
    test_configs = [
        TestConfig("early_p0.2_burn10_steps40", 0.2, 10, 40),  # Low density - severe imbalance
        TestConfig("mid_p0.4_burn60_steps40", 0.4, 60, 40),   # Medium density - balanced
        TestConfig("late_p0.6_burn160_steps40", 0.6, 160, 40), # High density - complex
    ]

    all_test_results = []

    for i, config in enumerate(test_configs, 1):
        print(f"\n📍 Test {i}/{len(test_configs)}: {config.name}")

        try:
            result = run_test_experiment(config, seed=42)  # Fixed seed for reproducibility
            all_test_results.append(result)
            print(f"✅ Test {i} completed successfully!")

        except Exception as e:
            print(f"❌ Test {i} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print overall test summary
    print(f"\n{'='*80}")
    print("🎉 ALL TESTS COMPLETED!")
    print(f"{'='*80}")

    if all_test_results:
        print(f"\n📊 OVERALL TEST SUMMARY:")
        print(f"{'Config':<25} {'3×3 Acc':<10} {'5×5 Acc':<10} {'Improve':<10} {'Recall Imp':<12}")
        print("-" * 75)

        total_improvement = 0
        total_recall_improvement = 0

        for result in all_test_results:
            config_name = result["config"]
            acc3 = result["patch3"]["test_accuracy"]
            acc5 = result["patch5"]["test_accuracy"]
            improvement = result["improvement"]
            recall_imp = result["recall_improvement"]

            print(f"{config_name:<25} {acc3:<10.4f} {acc5:<10.4f} {improvement:<10.4f} {recall_imp:<12.4f}")

            total_improvement += improvement
            total_recall_improvement += recall_imp

        avg_improvement = total_improvement / len(all_test_results)
        avg_recall_improvement = total_recall_improvement / len(all_test_results)

        print("-" * 75)
        print(f"{'Average':<25} {'':<10} {'':<10} {avg_improvement:<10.4f} {avg_recall_improvement:<12.4f}")

        print(f"\n🔍 Key Observations:")
        print(f"   • Average accuracy improvement: {avg_improvement:+.4f}")
        print(f"   • Average class 1 recall improvement: {avg_recall_improvement:+.4f}")

        if avg_recall_improvement > 0:
            print(f"   ✅ Weighted training successfully improved class 1 recall!")
        else:
            print(f"   ⚠️  Weighted training may need parameter adjustment")

        print(f"\n📝 Files created:")
        print(f"   • Test datasets: data/life_patches_test_*.npz")
        print(f"   • Test checkpoints: checkpoints_test/best_model_*.pth")
        print(f"   • Test confusion matrices: results_confusion_matrix_test.csv")

        print(f"\n🚀 Next Steps:")
        if avg_recall_improvement > 0:
            print(f"   ✅ Weighted training looks good - ready for full experiments!")
            print(f"   Run: python -m experiments_mlp_weighted")
        else:
            print(f"   🔧 Consider adjusting class weighting formula or early stopping")
            print(f"   Review individual test results for optimization opportunities")

    else:
        print(f"❌ All tests failed - please debug before running full experiments")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()