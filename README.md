# CS730 — Advanced Deep Learning | Spring 2026
## Project B: Multi-City Climate Prediction System

---

## Overview

This project implements city-specific LSTM (Long Short-Term Memory) models to predict daily temperature for **15 major cities in Sindh, Pakistan** using climate data from 2000–2024. Each city has a tailored model architecture accounting for its unique climate complexity and missing data patterns (indicated by -999 sentinel values).

---

## Repository Structure

```
Sindh Multi-City Climate Prediction System/
├── Report.pdf              # Technical report (IEEE format)
├── Notebook.ipynb          # Main Jupyter Notebook (end-to-end runnable)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── models/
│       ├── Karachi_model.h5
│       ├── Hyderabad_model.h5
│       ├── Sukkur_model.h5
│       ├── Larkana_model.h5
│       ├── Jacobabad_model.h5
│       ├── Dadu_model.h5
│       ├── Thatta_model.h5
│       ├── Badin_model.h5
│       ├── Mirpurkhas_model.h5
│       ├── Tharparkar_model.h5
│       ├── Shikarpur_model.h5
│       ├── Ghotki_model.h5
│       ├── Umerkot_model.h5
│       ├── Sanghar_model.h5
│       └── Khairpur_model.h5
└── figures/
        ├── missing_value_distribution.png
        ├── correlation_heatmap.png
        ├── validation_karachi.png
        ├── validation_hyderabad.png
        ├── validation_jacobabad.png
        ├── rmse_heatmap.png
        └── seasonal_error_heatmap.png
```

---

## Dataset

- **Source:** CMCC-CM2-VHR4 high-resolution climate model (CMIP6)
- **Time Span:** 2000–2024 (daily records)
- **Base City:** Karachi (real data); other 14 cities synthetically extended with city-specific biases
- **Features:** Temperature (°C), Wind Speed (km/h), Relative Humidity (%), Precipitation (mm)
- **Missing Value Indicator:** -999 (replaced via per-city mean imputation)
- **Cities (15):** Karachi, Hyderabad, Sukkur, Larkana, Jacobabad, Dadu, Thatta, Badin, Mirpurkhas, Tharparkar, Shikarpur, Ghotki, Umerkot, Sanghar, Khairpur

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Jupyter Notebook
```bash
Jupyter Notebook.ipynb
```

### 3. Run all cells top-to-bottom
The notebook is self-contained and runs end-to-end without errors. It will:
- Load and clean the base Karachi CSV data
- Generate synthetic multi-city data
- Inject and impute -999 missing values
- Train 15 city-specific LSTM models
- Evaluate and compare performance (RMSE, R²)
- Generate and display all required visualizations

> **Note:** Place the raw data file `1958 to 2025.csv` in `/content/` (Google Colab) or update the path in Cell 1 to your local path.

---

## Model Architecture

Each city uses a two-layer stacked LSTM:

```
Input → LSTM(N, return_sequences=True) → Dropout(0.2)
      → LSTM(N/2) → Dropout(0.2)
      → Dense(1)
```

Where N = 64 (low-complexity cities) or 128 (high-complexity cities).

| Hyperparameter | Range |
|---|---|
| LSTM Units | 64 or 128 |
| Lookback Window | 7, 10, or 14 days |
| Epochs | 15, 20, or 25 |
| Batch Size | 32 |
| Optimizer | Adam (lr=0.001) |
| Loss | Mean Squared Error |

---

## Key Results Summary

| Category | Cities | Avg RMSE | Avg R² |
|---|---|---|---|
| Easiest to predict | Badin, Thatta, Karachi | ~0.4°C | ~0.98 |
| Moderate | Hyderabad, Dadu, Ghotki | ~0.7°C | ~0.94 |
| Hardest to predict | Jacobabad, Tharparkar | ~1.6°C | ~0.82 |

Highest prediction error occurs in **Summer** and **Monsoon** seasons across all cities.

---

## Reproducibility

- Random seed fixed: `np.random.seed(42)` and `tf.random.set_seed(42)`
- All visualizations are generated inline within the notebook
- Model weights saved to `checkpoints/project_b_models/` after training

---

## Assignment Information

| Attribute | Details |
|---|---|
| Course | CS730 — Advanced Deep Learning |
| Semester | Spring 2026 |
| Project | B — Multi-City Climate Prediction System |
| Submission Format | Jupyter Notebook (.ipynb) + Technical Report (.docx) |
| Deadline | 25 May 2026 |
