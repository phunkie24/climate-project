# Quick Start Guide

Get started with the Climate Rainfall Analysis project in 5 minutes!

## Prerequisites

- Python 3.9 or higher
- pip package manager
- 4GB RAM minimum

## Installation

### Option 1: Local Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/climate-rainfall-analysis.git
cd climate-rainfall-analysis

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Option 2: Docker

```bash
# Build and run with docker-compose
docker-compose up -d dashboard

# Access dashboard at http://localhost:8501
```

## Quick Run

### 1. Process Data (2 minutes)

```bash
python scripts/process_data.py
```

This will:
- Load/create sample climate data
- Clean and validate data
- Engineer features
- Save processed data to `data/processed/`

### 2. Train Models (5 minutes)

```bash
python scripts/train_model.py
```

This will:
- Train Linear Regression, Random Forest, and XGBoost
- Evaluate models
- Save best model to `models/`

### 3. Launch Dashboard (instant)

```bash
streamlit run scripts/dashboard.py
```

Open your browser to **http://localhost:8501**

## What You Can Do

### Explore the Dashboard
- View rainfall trends over time
- Compare different regions
- Analyze country-level patterns
- Explore correlations
- Download filtered data

### Make Predictions

```python
from src.models import load_predictor

# Load trained model
predictor = load_predictor('xgboost')

# Make prediction
rainfall = predictor.predict({
    'avg_temp_c': 25.0,
    'avg_humidity(%)': 65.0,
    'atmospheric_co2(ppm)': 420.0,
    'cloud_cover(%)': 50.0,
    'years_since_1991': 34  # 2025
})

print(f"Predicted rainfall: {rainfall:.1f} mm")
```

### Custom Analysis

```python
from src.data import load_climate_data
import pandas as pd

# Load data
df = load_climate_data('data/processed/climate_data_clean.csv')

# Analyze specific country
nigeria = df[df['country'] == 'Nigeria']
avg_rainfall = nigeria['precipitation(mm)'].mean()
print(f"Nigeria average rainfall: {avg_rainfall:.1f} mm")

# Calculate trend
import numpy as np
years = nigeria['year'].values
precip = nigeria['precipitation(mm)'].values
trend = np.polyfit(years, precip, 1)[0]
print(f"Trend: {trend:.2f} mm/year")
```

## Project Structure

```
climate-project/
├── data/               # Data files
│   ├── raw/           # Original data
│   └── processed/     # Cleaned data
├── models/            # Trained models
├── scripts/           # Executable scripts
│   ├── process_data.py
│   ├── train_model.py
│   └── dashboard.py
├── src/               # Source code
│   ├── data/          # Data processing
│   ├── features/      # Feature engineering
│   ├── models/        # ML models
│   └── visualization/ # Plotting
└── notebooks/         # Jupyter notebooks
```

## Next Steps

1. **Explore notebooks/**
   - `01_data_exploration.ipynb` - EDA
   - `02_feature_engineering.ipynb` - Features
   - `03_model_training.ipynb` - Models

2. **Read documentation**
   - [Data Processing](docs/data_processing.md)
   - [Model Training](docs/model_training.md)
   - [API Reference](docs/api_reference.md)

3. **Customize**
   - Modify `config/config.yaml` for settings
   - Add your own data in `data/raw/`
   - Experiment with model parameters

## Common Issues

### Issue: Module not found
```bash
# Solution: Install in development mode
pip install -e .
```

### Issue: Data file not found
```bash
# Solution: Run data processing first
python scripts/process_data.py
```

### Issue: Model not trained
```bash
# Solution: Train models
python scripts/train_model.py
```

## Getting Help

- **Documentation**: See `docs/` folder
- **Issues**: Open an issue on GitHub
- **Examples**: Check `notebooks/` folder

## Tips

- Use **Docker** for easiest setup
- Run **process_data.py** first, always
- Check **config.yaml** for customization
- Use **dashboard** for quick insights
- Use **notebooks** for detailed analysis

Happy analyzing! 🌧️📊
