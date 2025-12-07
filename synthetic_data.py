import pandas as pd
import numpy as np
from IPython.display import display


# --- Step 1: Load Baseline Data (Assumed to be 2023) ---
print("--- Starting generation of City-Differentiated 2016-2024 synthetic time series dataset ---")

baseline_file = 'datasets/Original_Data.xlsx'
try:
    df_base = pd.read_excel(baseline_file, header=[0, 1, 2, 3, 4], index_col=0)
    print(f"Successfully loaded baseline year data: '{baseline_file}' (considered as 2023 data).\n")
except FileNotFoundError:
    print(f"Error: Baseline file '{baseline_file}' not found. Please ensure the file exists.")
    exit()

# Dictionary to store all years' data
all_data_by_year = {2023: df_base}

# Define year range
years_to_generate = range(2022, 2015, -1)  # From 2022 back to 2016
years_future = [2024]  # Forecast 2024


# --- Step 2: Define Annual Change Rules for Each Indicator ---
growth_rates = {
    # Disaster Hazard
    'Proportion of Population Aged 65 and Above': 0.032,
    'Population in Poverty': -0.100,
    'Population Density': 0.004,
    'Employment Density': 0.018,
    'Number of Domestic Tourist Arrivals': 0.100,
    'Number of Public Buses (and Trolleybuses) per 10,000 Population': 0.038,
    'Highway Passenger Traffic Volume': 0.014,
    'PM2.5 Concentration': -0.065,
    'Relative Humidity': 0.000,
    'Average Temperature': 0.002,
    # Urban Vulnerability
    'Residential Density': 0.016,
    'Density of Daily Life Service Facilities': 0.065,
    'Density of Transportation Facilities': 0.045,
    'Green Coverage Rate of Built-up Area': 0.009,
    'Open Space Density': 0.023,
    'Unemployment Rate': -0.005,
    'Income Level': 0.082,
    'Health Expenditure as Percentage of GDP': 0.030,
    'Annual Average Savings Deposit Balance per Resident': 0.125,
    'Proportion of Emergency Supplies Reserve Expenditure to GDP (%) (per 10,000 yuan)': 0.100,
    'Number of Physicians (per 10,000 people)': 0.045,
    'Hospital Beds (per 10,000 people)': 0.040,
    'Basic Social Insurance Coverage Rate': 0.008
}


# --- Step 2.5: Define Noise Ranges Based on Growth Rate Intervals ---
def get_noise_ranges(rate):
    """
    Return corresponding noise ranges based on the absolute value of the growth rate
    Returns: (indicator_noise_range, city_noise_range)
    """
    abs_rate = abs(rate)
    if abs_rate < 0.02:
        # Small interval: |rate| < 0.02 (e.g., Population Density 0.004, Average Temperature 0.002, etc.)
        # Set smaller noise range to avoid noise overwhelming true growth rate
        return (-0.005, 0.005), (-0.003, 0.003)
    elif abs_rate < 0.08:
        # Medium interval: 0.02 <= |rate| < 0.08 (e.g., Proportion of Aged 65+ 0.032, PM2.5 -0.065, etc.)
        # Moderate noise range
        return (-0.015, 0.015), (-0.008, 0.008)
    else:
        # Large interval: |rate| >= 0.08 (e.g., Tourist Arrivals 0.100, Population in Poverty -0.200, etc.)
        # Larger noise range allowed
        return (-0.025, 0.025), (-0.015, 0.015)


# --- Step 3: Loop to Generate Historical Years Data (2016-2022) ---
for year in years_to_generate:
    print(f"Generating simulated data for {year}...")
    last_year_data = all_data_by_year[year + 1].copy()

    current_year_data = pd.DataFrame(index=last_year_data.index, columns=last_year_data.columns)
    num_cities = len(current_year_data)

    for indicator_tuple in current_year_data.columns:
        indicator_name = indicator_tuple[2]
        rate = growth_rates.get(indicator_name, 0.01)

        # Get corresponding noise ranges based on growth rate magnitude
        indicator_noise_range, city_noise_range = get_noise_ranges(rate)
        indicator_noise = np.random.uniform(indicator_noise_range[0], indicator_noise_range[1])
        city_specific_noise = np.random.uniform(city_noise_range[0], city_noise_range[1], size=num_cities)

        # Default backward-looking divisor
        effective_rate_vector = 1 + rate + indicator_noise + city_specific_noise
        rate_series = pd.Series(effective_rate_vector, index=current_year_data.index)

        # Default backward calculation
        new_values = last_year_data[indicator_tuple] / rate_series

        # Special event handling (COVID-19)
        if indicator_name in ['Number of Domestic Tourist Arrivals', 'Highway Passenger Traffic Volume']:
            if year == 2022:
                shock_factor = np.random.uniform(0.75, 0.85, size=num_cities)
                new_values = last_year_data[indicator_tuple] * shock_factor
            elif year in [2021, 2020]:
                shock_factor = np.random.uniform(0.85, 1.15, size=num_cities)
                new_values = last_year_data[indicator_tuple] * shock_factor
            elif year == 2019:
                rebound_shock_factor = np.random.uniform(0.5, 0.6, size=num_cities)
                new_values = last_year_data[indicator_tuple] / rebound_shock_factor

        # Special handling for poverty alleviation campaign
        if indicator_name == 'Population in Poverty' and year < 2020:
             # Construct a smaller divisor to achieve faster reverse growth (i.e., accelerated forward decline)
             faster_decline_divisor = pd.Series(1 + rate * 2 + indicator_noise + city_specific_noise, index=current_year_data.index)
             new_values = last_year_data[indicator_tuple] / faster_decline_divisor.clip(lower=0.01)

        current_year_data[indicator_tuple] = np.clip(new_values, 0, None)

    all_data_by_year[year] = current_year_data


# --- Step 4: Generate Future Years Data (2024) ---
print(f"Generating simulated data for {years_future[0]}...")
last_year_data = all_data_by_year[2023].copy()
future_year_data = last_year_data.copy()
num_cities = len(future_year_data)

for indicator_tuple in future_year_data.columns:
    indicator_name = indicator_tuple[2]
    rate = growth_rates.get(indicator_name, 0.01)

    # Get corresponding noise ranges based on growth rate magnitude
    indicator_noise_range, city_noise_range = get_noise_ranges(rate)
    indicator_noise = np.random.uniform(indicator_noise_range[0], indicator_noise_range[1])
    city_specific_noise = np.random.uniform(city_noise_range[0], city_noise_range[1], size=num_cities)
    effective_rate_vector = 1 + rate + indicator_noise + city_specific_noise
    rate_series = pd.Series(effective_rate_vector, index=future_year_data.index)

    new_values = last_year_data[indicator_tuple] * rate_series
    future_year_data[indicator_tuple] = np.clip(new_values, 0, None)

all_data_by_year[years_future[0]] = future_year_data

print("\n--- City-Differentiated synthetic data for all years (2016-2024) has been generated ---")
print("Displaying simulated data for 2016 as example:")
display(all_data_by_year[2016].head())


# --- Step 5: Export All Data to an Excel File ---
output_filename = "datasets/Synthetic_TimeSeries_Data_2016-2024_With_City_Variance.xlsx"
try:
    with pd.ExcelWriter(output_filename) as writer:
        for year, df_year in sorted(all_data_by_year.items()):
            df_year.to_excel(writer, sheet_name=str(year))
    print(f"\nData export successful!")
    print(f"All 9 years of synthetic data saved to file: '{output_filename}'")
    print("Each year's data is stored in corresponding sheet.")
except Exception as e:
    print(f"\nData export failed: {e}")
