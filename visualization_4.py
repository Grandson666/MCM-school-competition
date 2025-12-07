import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random


# Set style
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("white")

print("="*80)
print("--- Starting C-T-D Dual-Axis Visualization ---")
print("="*80)


# --- Step 1: Read saved C, T, D data from Excel ---
try:
    C_df = pd.read_excel('datasets/CTD_values_all_years.xlsx', sheet_name='C_improved', index_col=0)
    T_df = pd.read_excel('datasets/CTD_values_all_years.xlsx', sheet_name='T_coordination', index_col=0)
    D_df = pd.read_excel('datasets/CTD_values_all_years.xlsx', sheet_name='D_dynamic', index_col=0)
    print(f"\n  Successfully loaded data:")
    print(f"  - C (Coupling degree): {C_df.shape[0]} cities × {C_df.shape[1]} years")
    print(f"  - T (Coordination index): {T_df.shape[0]} cities × {T_df.shape[1]} years")
    print(f"  - D (CCD): {D_df.shape[0]} cities × {D_df.shape[1]} years")
except FileNotFoundError as e:
    print(f"\nError: Data file not found!")
    print("Please run ccd_model_with_t.py first to generate CTD_values_all_years.xlsx")
    exit()

# Convert column names to integer years if needed
C_df.columns = [int(col) for col in C_df.columns]
T_df.columns = [int(col) for col in T_df.columns]
D_df.columns = [int(col) for col in D_df.columns]


# --- Step 2: Select 9 random cities ---
all_cities = C_df.index.tolist()
print(f"\nAvailable cities: {all_cities}")

# Randomly select 9 cities
selected_cities = random.sample(all_cities, 9)
print(f"Selected cities for visualization: {selected_cities}")


# --- Step 3: Select specific years for X-axis ---
selected_years = [2016, 2019, 2021, 2024]
print(f"Selected years for X-axis: {selected_years}")


# --- Step 4: Create 3x3 subplot layout ---
fig, axes = plt.subplots(3, 3, figsize=(16, 16))
axes = axes.flatten()

# Color scheme
bar_color_c = '#FF8C42'  # Orange for C
bar_color_t = '#F4D35E'  # Yellow for T
line_color = '#8B4513'  # Brown for CCD line


# --- Step 5: Plot each city ---
for idx, city in enumerate(selected_cities):
    ax1 = axes[idx]

    # Extract data for this city for selected years
    C_values = C_df.loc[city, selected_years].values
    T_values = T_df.loc[city, selected_years].values
    D_values = D_df.loc[city, selected_years].values

    # Set up X-axis positions
    x_pos = np.arange(len(selected_years))
    bar_width = 0.35

    # Plot bars for C and T on primary Y-axis
    bars_c = ax1.bar(x_pos - bar_width/2, C_values, bar_width, label='C (Coupling degree)', color=bar_color_c, alpha=0.8)
    bars_t = ax1.bar(x_pos + bar_width/2, T_values, bar_width, label='T (Coordination index)', color=bar_color_t, alpha=0.8)

    # Configure primary Y-axis (for C and T)
    ax1.set_ylabel('Coordinated Influence (C & T)', fontsize=10, fontweight='bold')
    ax1.set_ylim(-0.2, max(max(C_values), max(T_values)) * 1.2)
    ax1.tick_params(axis='y', labelsize=9)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8)

    # Create secondary Y-axis for D (CCD)
    ax2 = ax1.twinx()

    # Plot line for D on secondary Y-axis
    line_d = ax2.plot(x_pos, D_values, color=line_color, linewidth=3, marker='o', markersize=8, label='CCD', zorder=10)

    # Configure secondary Y-axis (for D)
    ax2.set_ylabel('CCD', fontsize=10, fontweight='bold')
    ax2.set_ylim(0, max(D_values) * 1.3)
    ax2.tick_params(axis='y', labelsize=9)

    # Set X-axis
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(selected_years, fontsize=9)
    ax1.set_xlabel('Year', fontsize=10, fontweight='bold')

    # Set title
    ax1.set_title(f'({chr(97+idx)}) {city}', fontsize=11, fontweight='bold', pad=10)

    # Remove all grid lines
    ax1.grid(False)
    ax2.grid(False)

# Add legend to the last subplot
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=bar_color_c, alpha=0.8, label='C (Coupling degree)'),
    Patch(facecolor=bar_color_t, alpha=0.8, label='T (Coordination index)'),
    Line2D([0], [0], color=line_color, linewidth=3, marker='o', markersize=8, label='CCD')
]
axes[-1].legend(handles=legend_elements, loc='upper left', fontsize=6, framealpha=0.9)

# Adjust layout
fig.subplots_adjust(hspace=0.4, wspace=0.5)


# --- Step 6: Save figure ---
output_filename = 'visualization_outputs/Fig_CTD_DualAxis_9_Random_Cities.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nC-T-D dual-axis plot saved to: {output_filename}")

# Display figure
plt.show()

print("\n" + "="*80)
print("--- Visualization Complete! ---")
print("="*80)
