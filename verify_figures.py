import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('results_mlp_grid.csv')

print("=== VERIFICATION OF FIGURE DATA ===")
print()

# Verify Figure 1: Overall Performance
print("FIGURE 1 - Overall Performance:")
stats = df.groupby('patch_size')['test_accuracy'].agg(['mean', 'std']).reset_index()
print("3×3 mean:", stats.loc[stats['patch_size'] == 3, 'mean'].iloc[0])
print("5×5 mean:", stats.loc[stats['patch_size'] == 5, 'mean'].iloc[0])
print("3×3 std:", stats.loc[stats['patch_size'] == 3, 'std'].iloc[0])
print("5×5 std:", stats.loc[stats['patch_size'] == 5, 'std'].iloc[0])
improvement = stats.loc[stats['patch_size'] == 5, 'mean'].iloc[0] - stats.loc[stats['patch_size'] == 3, 'mean'].iloc[0]
print("Improvement:", improvement)
print()

# Verify Figure 2: Regime Comparison
print("FIGURE 2 - Regime Comparison:")
regime_performance = df.groupby(['regime', 'patch_size'])['test_accuracy'].mean().unstack()
regimes = ['early', 'mid', 'late']

for regime in regimes:
    acc_3x3 = regime_performance.loc[regime, 3]
    acc_5x5 = regime_performance.loc[regime, 5]
    print(f"{regime}: 3×3={acc_3x3:.6f}, 5×5={acc_5x5:.6f}, diff={acc_5x5-acc_3x3:.6f}")
print()

# Verify Figure 3: Density Comparison
print("FIGURE 3 - Density Comparison:")
density_performance = df.groupby(['p_alive', 'patch_size'])['test_accuracy'].mean().unstack()
densities = sorted(df['p_alive'].unique())

for density in densities:
    acc_3x3 = density_performance.loc[density, 3]
    acc_5x5 = density_performance.loc[density, 5]
    print(f"Density {density}: 3×3={acc_3x3:.6f}, 5×5={acc_5x5:.6f}, diff={acc_5x5-acc_3x3:.6f}")
print()

# Verify Figure 4: Improvement Heatmap
print("FIGURE 4 - Improvement Heatmap:")
improvement_data = []
for regime in ['early', 'mid', 'late']:
    for density in [0.2, 0.4, 0.6]:
        acc_3x3 = df[(df['regime'] == regime) & (df['patch_size'] == 3) & (df['p_alive'] == density)]['test_accuracy'].mean()
        acc_5x5 = df[(df['regime'] == regime) & (df['patch_size'] == 5) & (df['p_alive'] == density)]['test_accuracy'].mean()
        improvement = acc_5x5 - acc_3x3
        improvement_data.append(improvement)
        print(f"{regime}, density {density}: improvement = {improvement:.6f}")

print()
print("Heatmap matrix:")
improvement_matrix = np.array(improvement_data).reshape(3, 3)
print(improvement_matrix)
print()

# Verify Figure 5: Raw scatter data
print("FIGURE 5 - Scatter Plot Verification:")
print("Sample of actual test_loss and test_accuracy values:")
sample_data = df[['patch_size', 'test_loss', 'test_accuracy']].head(10)
print(sample_data)
print()

# Verify Figure 6: Box plot data
print("FIGURE 6 - Box Plot Verification:")
for regime in ['early', 'mid', 'late']:
    for patch_size in [3, 5]:
        subset = df[(df['regime'] == regime) & (df['patch_size'] == patch_size)]['test_accuracy']
        print(f"{regime}, {patch_size}×{patch_size}: min={subset.min():.6f}, max={subset.max():.6f}, mean={subset.mean():.6f}")
        print(f"  Values: {sorted(subset.values)}")