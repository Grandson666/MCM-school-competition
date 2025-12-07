# Urban Disaster Resilience Assessment Project

## Overview

This project implements a **Dynamic Coupling Coordination Degree (CCD) Model** to assess urban disaster resilience across multiple cities from 2016 to 2024. The model evaluates the interaction between **Disaster Hazard** (U1) and **Urban Vulnerability** (U2) subsystems using dynamic entropy weighting and time-decay mechanisms.

## Project Structure

```
MCM-school-competition/
├── README.md                        # Project documentation
├── synthetic_data.py                # Generates time-series data (2016-2024)
├── ccd_model_with_t.py              # Core CCD model implementation
├── visualization_1.py               # Indicator trend analysis
├── visualization_2.py               # Static scores heatmap
├── visualization_3.py               # Additional visualization
├── visualization_4.py               # U1-U2 fitting curves
├── datasets/
│   ├── Original_Data.xlsx                               # Baseline data (2023)
│   ├── Synthetic_TimeSeries_Data_2016-2024_*.xlsx      # Generated time-series
│   ├── Static_Scores_U1_Disaster_Hazard.csv            # U1 scores output
│   ├── Static_Scores_U2_Urban_Vulnerability.csv        # U2 scores output
│   └── CTD_Values_All_Years.xlsx                       # C, T, D values
├── references/                      # Reference materials
└── visualization_outputs/           # Generated charts
```

## Core Methodology

### 1. Data Standardization
Range normalization applied to all indicators:
- **Positive indicators** (higher is better): `(x - min) / (max - min)`
- **Negative indicators** (lower is better): `(max - x) / (max - min)`

### 2. Dynamic Entropy Weighting
Weights calculated independently for each year:
1. Normalize: `P_ij = X_ij / Σ(X_ij)`
2. Entropy: `E_j = -k × Σ(P_ij × ln(P_ij))`
3. Weight: `W_j = (1 - E_j) / Σ(1 - E_j)`

### 3. Time Coefficient Calculation
- **Annual change rate**: `f_j(t) = (x̄_tj - x̄_(t-1)j) / x̄_(t-1)j`
- **Time coefficient**: `F = Σ Σ (f_j(t) / t)`

### 4. Dynamic Comprehensive Scores
- **U1 (Disaster Hazard)**: `U1_t = u1_t × (1-F_U1)^(t_max - t)`
- **U2 (Urban Vulnerability)**: `U2_t = u2_t × (1-F_U2)^(t_max - t)`

### 5. Coupling Coordination Degree (CCD)
- **Coupling degree (C)**: Improved piecewise function handling U1-U2 asymmetry
- **Coordination index (T)**: `T = α × U1 + β × U2` (α=β=0.5)
- **CCD (D)**: `D = √(C × T)`

## Quick Start

### Installation
```bash
pip install pandas numpy matplotlib seaborn openpyxl
```

### Usage
1. **Generate synthetic data**: `python synthetic_data.py`
2. **Run CCD model**: `python ccd_model_with_t.py`
3. **Create visualizations**: `python visualization_*.py`

## Outputs

- **CSV files**: U1 and U2 static scores by city and year
- **Excel files**: C, T, D values across all years
- **PNG files**: Heatmaps, trend charts, fitting curves

## Key Features

- **Dynamic weighting**: Adapts to annual data distribution changes
- **Time decay**: Emphasizes recent years in longitudinal analysis
- **City differentiation**: Synthetic data with realistic inter-city variance
- **Multi-dimensional visualization**: Heatmaps, trends, and correlation charts

---

**Data Sources**: 23 indicators across 2 subsystems (10 for U1, 13 for U2)  
**Time Range**: 2016-2024 (historical + forecast)  
**Analysis Method**: Dynamic CCD with entropy weighting
