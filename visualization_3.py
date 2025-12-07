import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random


# Set style
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

print("="*80)
print("--- Starting Fitting Curve Visualization ---")
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


# --- Step 2: Select 9 random cities ---
# Choose cities with different characteristics based on the data
all_cities = U1_df.index.tolist()
print(f"\nAvailable cities: {all_cities}")

# Select 9 cities
selected_cities = random.sample(all_cities, 9)
print(f"Selected cities for visualization: {selected_cities}")


# --- Step 3: Create 3x3 subplot layout ---
fig, axes = plt.subplots(3, 3, figsize=(16, 16))
axes = axes.flatten()  # Flatten to 1D array for easier iteration

# Color scheme
scatter_color = '#7CB342'  # Green
line_color = '#FF6F00'  # Orange


# --- Step 4: Plot each city ---
for idx, city in enumerate(selected_cities):
    ax = axes[idx]

    # Extract data for this city across all years
    U1_values = U1_df.loc[city].values  # Disaster Hazard (X-axis)
    U2_values = U2_df.loc[city].values  # Urban Vulnerability (Y-axis)

    # Plot scatter points
    ax.scatter(U1_values, U2_values,
               s=80, alpha=0.7, color=scatter_color,
               edgecolors='darkgreen', linewidth=1.5,
               label='Data points', zorder=3)
    
    # Fit polynomial curve
    # Filter out any NaN values
    valid_mask = ~(np.isnan(U1_values) | np.isnan(U2_values))
    U1_valid = U1_values[valid_mask]
    U2_valid = U2_values[valid_mask]

    if len(U1_valid) > 2:
        # Polynomial fitting (degree=2)
        coefficients = np.polyfit(U1_valid, U2_valid, deg=2)
        polynomial = np.poly1d(coefficients)

        # Generate smooth curve
        U1_range = np.linspace(U1_valid.min(), U1_valid.max(), 100)
        U2_fitted = polynomial(U1_range)

        # Plot fitted curve
        ax.plot(U1_range, U2_fitted,
                color=line_color, linewidth=3,
                label='Fitted curve', zorder=2)
        
    # Set labels and title
    ax.set_xlabel('Disaster Hazard', fontsize=11, fontweight='bold')
    ax.set_ylabel('Urban Vulnerability', fontsize=11, fontweight='bold')
    ax.set_title(f'({chr(97+idx)}) {city}', fontsize=12, fontweight='bold', pad=10)

    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Set background color
    ax.set_facecolor('#FAFAFA')

# Add legend to the last subplot
axes[-1].legend(loc='lower right', fontsize=9, framealpha=0.9)

# Adjust layout
fig.subplots_adjust(hspace=0.4, wspace=0.3)


# --- Step 5: Save figure ---
output_filename = 'visualization_outputs/Fig_Fitting_Curves_9_Random_Cities.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nFitting curve plot saved to: {output_filename}")

# Display figure
plt.show()

print("\n" + "="*80)
print("--- Visualization Complete! ---")
print("="*80)
