import pandas as pd
import numpy as np

# Load the exact data
df = pd.read_csv('results_mlp_grid.csv')

print("=== ACCURATE DATA ANALYSIS ===")
print(f"Total experiments: {len(df)}")
print(f"Columns: {list(df.columns)}")
print()

# Overall statistics
print("1. OVERALL ACCURACY (exact values):")
overall_stats = df.groupby('patch_size')['test_accuracy'].agg(['mean', 'std', 'count']).reset_index()
print(overall_stats)
print()

print(f"3×3 average accuracy: {overall_stats.loc[overall_stats['patch_size'] == 3, 'mean'].iloc[0]:.6f}")
print(f"5×5 average accuracy: {overall_stats.loc[overall_stats['patch_size'] == 5, 'mean'].iloc[0]:.6f}")
improvement = overall_stats.loc[overall_stats['patch_size'] == 5, 'mean'].iloc[0] - overall_stats.loc[overall_stats['patch_size'] == 3, 'mean'].iloc[0]
print(f"Average improvement: {improvement:.6f} ({improvement*100:.4f}%)")
print()

# By regime
print("2. BY REGIME (exact values):")
regime_stats = df.groupby(['regime', 'patch_size'])['test_accuracy'].mean().unstack()
print(regime_stats)
print()

for regime in ['early', 'mid', 'late']:
    acc_3x3 = regime_stats.loc[regime, 3]
    acc_5x5 = regime_stats.loc[regime, 5]
    improvement = acc_5x5 - acc_3x3
    print(f"{regime}: 3×3={acc_3x3:.6f}, 5×5={acc_5x5:.6f}, improvement={improvement:.6f} ({improvement*100:.4f}%)")
print()

# By density
print("3. BY DENSITY (exact values):")
density_stats = df.groupby(['p_alive', 'patch_size'])['test_accuracy'].mean().unstack()
print(density_stats)
print()

for density in sorted(df['p_alive'].unique()):
    acc_3x3 = density_stats.loc[density, 3]
    acc_5x5 = density_stats.loc[density, 5]
    improvement = acc_5x5 - acc_3x3
    print(f"Density {density}: 3×3={acc_3x3:.6f}, 5×5={acc_5x5:.6f}, improvement={improvement:.6f} ({improvement*100:.4f}%)")
print()

# Best configurations
print("4. BEST CONFIGURATIONS (exact values):")
best_3x3 = df[df['patch_size'] == 3].loc[df[df['patch_size'] == 3]['test_accuracy'].idxmax()]
best_5x5 = df[df['patch_size'] == 5].loc[df[df['patch_size'] == 5]['test_accuracy'].idxmax()]

print("Best 3×3 configuration:")
print(f"  {best_3x3['regime']}, density {best_3x3['p_alive']}, seed {best_3x3['seed']}: {best_3x3['test_accuracy']:.6f}")
print("Best 5×5 configuration:")
print(f"  {best_5x5['regime']}, density {best_5x5['p_alive']}, seed {best_5x5['seed']}: {best_5x5['test_accuracy']:.6f}")
print()

# Verify calculations with raw data
print("5. VERIFICATION - Raw data sample:")
print("Top 5 best performing 5×5 configurations:")
top_5x5 = df[df['patch_size'] == 5].sort_values('test_accuracy', ascending=False).head()
for _, row in top_5x5.iterrows():
    print(f"  {row['regime']}, density {row['p_alive']}, seed {row['seed']}: {row['test_accuracy']:.6f}")
print()

print("Top 5 best performing 3×3 configurations:")
top_3x3 = df[df['patch_size'] == 3].sort_values('test_accuracy', ascending=False).head()
for _, row in top_3x3.iterrows():
    print(f"  {row['regime']}, density {row['p_alive']}, seed {row['seed']}: {row['test_accuracy']:.6f}")