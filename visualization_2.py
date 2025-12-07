import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Set style
plt.rcParams['axes.unicode_minus'] = False  # Display minus sign correctly
sns.set_style("whitegrid")

print("="*80)
print("--- Starting Static Scores Heatmap Visualization ---")
print("="*80)


# --- Step 1: Read saved static score data ---
try:
    U1_df = pd.read_csv('datasets/Static_Scores_U1_Disaster_Hazard.csv', index_col=0)
    U2_df = pd.read_csv('datasets/Static_Scores_U2_Urban_Vulnerability.csv', index_col=0)
    print(f"\n  Successfully loaded data:")
    print(f"  - U1 (Disaster Hazard): {U1_df.shape[0]} cities × {U1_df.shape[1]} years")
    print(f"  - U2 (Urban Vulnerability): {U2_df.shape[0]} cities × {U2_df.shape[1]} years")
except FileNotFoundError as e:
    print(f"\nError: Data files not found!")
    print("Please run ccd_model_with_t.py first to generate data files.")
    exit()

# Convert column names to integer years if needed
U1_df.columns = [int(col) for col in U1_df.columns]
U2_df.columns = [int(col) for col in U2_df.columns]


# --- Step 2: Transpose data (years in rows, cities in columns) ---
U1_df_T = U1_df.T  # Transpose: rows=years, columns=cities
U2_df_T = U2_df.T

print(f"\nTransposed data shape (years × cities): {U1_df_T.shape}")


# --- Step 3: Plot dual-panel heatmap ---
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# (a) Disaster Hazard
sns.heatmap(U1_df_T,
            ax=axes[0],
            cmap='RdYlBu_r',  # Red-Yellow-Blue reversed colormap, higher values are redder
            cbar_kws={'label': 'Score'},
            linewidths=0.5,
            linecolor='white',
            vmin=0,  # Standardized data range is 0-1
            vmax=1,
            fmt='.2f',
            annot=False)  # Don't annotate values if too many cities

axes[0].set_title('(a) Disaster Hazard Level', fontsize=14, fontweight='bold', pad=15)
axes[0].set_xlabel('City', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Year', fontsize=12, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)

# (b) Urban Vulnerability
sns.heatmap(U2_df_T,
            ax=axes[1],
            cmap='RdYlBu_r',
            cbar_kws={'label': 'Score'},
            linewidths=0.5,
            linecolor='white',
            vmin=0,
            vmax=1,
            fmt='.2f',
            annot=False)
axes[1].set_title('(b) Urban Vulnerability Level', fontsize=14, fontweight='bold', pad=15)
axes[1].set_xlabel('City', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Year', fontsize=12, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)

# Adjust layout
plt.tight_layout()


# --- Step 4: Save figure ---
output_filename = 'Visualization_outputs/Fig_Static_Scores_Heatmap.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nHeatmap saved to: {output_filename}")

# Display figure
plt.show()

print("\n" + "="*80)
print("--- Visualization Complete! ---")
print("="*80)
