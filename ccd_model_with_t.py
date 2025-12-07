import pandas as pd
import numpy as np
from IPython.display import display


# --- Step 0: Define Core Functions ---
def standardize_data(df, indicator_types):
    """Standardize the given DataFrame using range normalization"""
    df_standardized = df.copy()
    for col in df_standardized.columns:
        indicator_type = indicator_types[col]
        if indicator_type == 'P':
            min_val = df_standardized[col].min()
            max_val = df_standardized[col].max()
            if max_val != min_val:
                df_standardized[col] = (df_standardized[col] - min_val) / (max_val - min_val)
            else:
                df_standardized[col] = 0
        elif indicator_type == 'N':
            min_val = df_standardized[col].min()
            max_val = df_standardized[col].max()
            if max_val != min_val:
                df_standardized[col] = (max_val - df_standardized[col]) / (max_val - min_val)
            else:
                df_standardized[col] = 0
    return df_standardized


def calculate_entropy_weights(df_subsystem):
    """Calculate entropy weights for the given subsystem dataframe"""
    m, n = df_subsystem.shape
    data = df_subsystem + 1e-9
    P = data.apply(lambda x: x / x.sum(), axis=0)
    k = 1 / np.log(m)
    entropy_sum = P.apply(lambda x: (x * np.log(x)).sum(), axis=0)
    E = -k * entropy_sum
    d = 1 - E
    W = d / d.sum()
    return pd.Series(W.values, index=df_subsystem.columns)


# --- Step 1: Load Multi-Year Raw Data ---
time_series_file = 'datasets/Synthetic_TimeSeries_Data_2016-2024_With_City_Variance.xlsx'
try:
    # Read all years' data into a dictionary, key as year
    xls = pd.ExcelFile(time_series_file)
    years = [int(sheet_name) for sheet_name in xls.sheet_names]
    t_min, t_max = min(years), max(years)

    raw_data_dict = {}
    for year in years:
        df_year = pd.read_excel(xls, sheet_name=str(year), header=[0, 1, 2, 3, 4], index_col=0)
        # Remove blank characters from column names
        df_year.columns = pd.MultiIndex.from_tuples(
            [tuple(str(level).strip() for level in col) for col in df_year.columns],
            names=df_year.columns.names
        )
        raw_data_dict[year] = df_year

    print(f"--- Successfully loaded {len(years)} years of data ({t_min}-{t_max}) ---")
except FileNotFoundError:
    print(f"Error: Time series data file '{time_series_file}' not found")
    exit()

# Extract indicator types (from the last year's data, as they are fixed)
indicator_types_multi = raw_data_dict[t_max].columns.get_level_values(3)
indicator_names_full = raw_data_dict[t_max].columns


# --- Step 2: Standardize Data for Each Year ---
standardized_data_dict = {}
for year, df_raw in raw_data_dict.items():
    # Extract pure numerical data and single-level indicator types
    df_values = df_raw.copy()
    df_values.columns = indicator_names_full
    indicator_types_flat = pd.Series(indicator_types_multi, index=indicator_names_full)

    standardized_data_dict[year] = standardize_data(df_values, indicator_types_flat)
print(f"--- Completed standardization for {len(years)} years of data ---\n")


# --- Step 3: Loop to Calculate Dynamic Weights and Static Scores for Each Year ---
print("--- Starting loop calculation of Dynamic Weights and Static Comprehensive Scores for each year ---")
static_scores_dict = {}
for year in years:
    df_std_year = standardized_data_dict[year]

    # 1. Divide current year's data by subsystem
    df_hazard_year = df_std_year.xs('Disaster Hazard', level=0, axis=1)
    df_vulnerability_year = df_std_year.xs('Urban Vulnerability', level=0, axis=1)

    # 2. Calculate weights independently for current year (Dynamic Weight Method)
    weights_hazard_t = calculate_entropy_weights(df_hazard_year)
    weights_vulnerability_t = calculate_entropy_weights(df_vulnerability_year)

    # 3. Calculate static scores for current year using current year's weights
    U1_t = df_hazard_year.dot(weights_hazard_t)
    U2_t = df_vulnerability_year.dot(weights_vulnerability_t)

    static_scores_dict[year] = pd.DataFrame({'U1_static': U1_t, 'U2_static': U2_t})
    print(f"  -> Weights and static scores calculation completed for year {year}.")
print("--- Static comprehensive scores U1_t and U2_t calculation completed for all years ---\n")


# --- Step 4: Calculate Time Coefficient F ---
print("--- Starting calculation of time coefficient F for dynamic model ---")

# 1. Calculate annual mean of each subsystem's indicators
mean_hazard_yearly = {}
mean_vulnerability_yearly = {}
for year, df_std in standardized_data_dict.items():
    df_hazard = df_std.xs('Disaster Hazard', level=0, axis=1)
    df_vulnerability = df_std.xs('Urban Vulnerability', level=0, axis=1)
    mean_hazard_yearly[year] = df_hazard.mean()  # Mean of each indicator for that year
    mean_vulnerability_yearly[year] = df_vulnerability.mean()

# 2. Define time sequence mapping
year_indices = {year: i + 1 for i, year in enumerate(sorted(years))}

# 3. Calculate time coefficient F_UL for UL system
f_ul_sum = 0
for t_year in sorted(years)[1:]:  # Start from second year to calculate growth rate
    t_prev = t_year - 1
    t_index = year_indices[t_year]

    mean_t = mean_hazard_yearly[t_year]
    mean_t_prev = mean_hazard_yearly[t_prev]

    # Calculate growth rate f_j(t) for each indicator
    growth_rates = (mean_t - mean_t_prev) / (mean_t_prev + 1e-9)

    # Accumulate Σ(f_j(t) / t)
    f_ul_sum += (growth_rates / t_index).sum()

F_UL = f_ul_sum

# 4. Calculate time coefficient F_FR for FR system
f_fr_sum = 0
for t_year in sorted(years)[1:]:
    t_prev = t_year - 1
    t_index = year_indices[t_year]

    mean_t = mean_vulnerability_yearly[t_year]
    mean_t_prev = mean_vulnerability_yearly[t_prev]

    growth_rates = (mean_t - mean_t_prev) / (mean_t_prev + 1e-9)
    f_fr_sum += (growth_rates / t_index).sum()

F_FR = f_fr_sum

print(f"Step 1/2: Time coefficient calculation completed")
print(f"  - UL system (Disaster Hazard) time coefficient F_UL = {F_UL:.6f}")
print(f"  - FR system (Urban Vulnerability) time coefficient F_FR = {F_FR:.6f}")


# --- Step 5: Calculate Dynamic Comprehensive Scores for Each Year (UL_t, FR_t) ---
dynamic_scores_dict = {}
for year in years:
    ul_t = static_scores_dict[year]['U1_static']
    fr_t = static_scores_dict[year]['U2_static']

    # Calculate time decay factor using respective time coefficients
    time_decay_ul = (1 - F_UL) ** (t_max - year)
    time_decay_fr = (1 - F_FR) ** (t_max - year)

    UL_t = ul_t * time_decay_ul
    FR_t = fr_t * time_decay_fr

    dynamic_scores_dict[year] = pd.DataFrame({'U1_dynamic': UL_t, 'U2_dynamic': FR_t})

print("Step 2/2: Dynamic comprehensive scores UL_t and FR_t calculated for all years")
print(f"  - UL system uses decay factor: (1-F_UL)^(t_max-t)")
print(f"  - FR system uses decay factor: (1-F_FR)^(t_max-t)\n")


# --- Step 6: Calculate Dynamic Coupling Coordination Degree D for Each Year ---
print("--- Starting calculation of dynamic coupling coordination degree D for each year ---")
final_results_list = []
alpha, beta = 0.5, 0.5

for year in sorted(years):
    df_dynamic = dynamic_scores_dict[year]
    U1 = df_dynamic['U1_dynamic']
    U2 = df_dynamic['U2_dynamic']

    # Calculate improved coupling degree C
    C_numerator = np.where(U1 >= U2, (U2 - U1) * (U1 / (U2 + 1e-9)), (U1 - U2) * (U2 / (U1 + 1e-9)))
    C_t = np.sqrt(np.abs(1 - C_numerator))

    # Calculate comprehensive coordination index T
    T_t = alpha * U1 + beta * U2

    # Calculate final coupling coordination degree D
    D_t = np.sqrt(C_t * T_t)

    # Store current year's results
    year_results = pd.DataFrame({
        'Year': year,
        'City': df_dynamic.index,
        'U1_dynamic': U1,
        'U2_dynamic': U2,
        'C_improved': C_t,
        'T_coordination': T_t,
        'D_dynamic': D_t
    })
    final_results_list.append(year_results)

# Merge results for all years
final_df = pd.concat(final_results_list).reset_index(drop=True)
print("--- Dynamic coupling coordination degree calculation completed for all years! ---\n")


# --- Step 7: Display Results ---
print("="*80 + "\n")
print("--- Dynamic Coupling Coordination Degree Model Final Results (2016-2024) ---")

# For clearer display, pivot data so each row is a city, columns are D values for each year
pivot_df = final_df.pivot(index='City', columns='Year', values='D_dynamic')

# Calculate mean and latest year value for each city, and sort
pivot_df['Mean_D'] = pivot_df.mean(axis=1)
pivot_df_sorted = pivot_df.sort_values(by=t_max, ascending=False)

print("\nDynamic Coupling Coordination Degree (D) Trend by City Over Years")
display(pivot_df_sorted)

print("\nDetailed Results for {} by City (Sorted by D value in descending order)".format(t_max))
display(final_df[final_df['Year'] == t_max].sort_values(by='D_dynamic', ascending=False).set_index('City'))


# --- Step 8: Save Static Comprehensive Scores Data for Visualization ---
print("\n" + "="*80)
print("--- Saving static comprehensive scores data to file ---")

# Convert static_scores_dict to a format convenient for saving and reading
# Create two DataFrames: one for U1_static, one for U2_static
U1_data = {}
U2_data = {}

for year in sorted(years):
    U1_data[year] = static_scores_dict[year]['U1_static']
    U2_data[year] = static_scores_dict[year]['U2_static']

# Convert to DataFrame (rows=cities, columns=years)
U1_df = pd.DataFrame(U1_data)
U2_df = pd.DataFrame(U2_data)

# Save as CSV files (more convenient to read and view)
U1_df.to_csv('datasets/Static_Scores_U1_Disaster_Hazard.csv')
U2_df.to_csv('datasets/Static_Scores_U2_Urban_Vulnerability.csv')

print(f"  U1 static scores (Disaster Hazard) saved to: Static_Scores_U1_Disaster_Hazard.csv")
print(f"  U2 static scores (Urban Vulnerability) saved to: Static_Scores_U2_Urban_Vulnerability.csv")
print(f"  Data dimensions: {U1_df.shape[0]} cities × {U1_df.shape[1]} years")
print("="*80)


# --- Step 9: Save C, T, D Values to Different Sheets in Excel File ---
print("\n" + "="*80)
print("--- Saving C, T, D values to Excel file ---")

# Extract C, T, D values from final_df and convert to pivot table format (rows=cities, columns=years)
C_data = {}
T_data = {}
D_data = {}

for year in sorted(years):
    df_year = final_df[final_df['Year'] == year].set_index('City')
    C_data[year] = df_year['C_improved']
    T_data[year] = df_year['T_coordination']
    D_data[year] = df_year['D_dynamic']

# Convert to DataFrame (rows=cities, columns=years)
C_df = pd.DataFrame(C_data)
T_df = pd.DataFrame(T_data)
D_df = pd.DataFrame(D_data)

# Save to different sheets in the same Excel file
excel_filename = 'datasets/CTD_Values_All_Years.xlsx'
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    C_df.to_excel(writer, sheet_name='C_improved')
    T_df.to_excel(writer, sheet_name='T_coordination')
    D_df.to_excel(writer, sheet_name='D_dynamic')

print(f"  C, T, D values saved to: {excel_filename}")
print(f"  - Sheet 'C_improved': Coupling degree")
print(f"  - Sheet 'T_coordination': Comprehensive coordination index")
print(f"  - Sheet 'D_dynamic': Coupling coordination degree")
print(f"  Data dimensions: {C_df.shape[0]} cities × {C_df.shape[1]} years")
print("="*80)
