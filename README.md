# Climate Rainfall Analysis - Sub-Saharan Africa

Machine Learning project for detecting and predicting rainfall patterns in Sub-Saharan Africa (1991-2023).

## 🎯 Project Overview

This project analyzes 33 years of climate data across 49 Sub-Saharan African countries to:
- Detect long-term rainfall trends
- Identify extreme weather events (droughts/floods)
- Build predictive ML models for rainfall forecasting
- Create interactive dashboards for insights

## 📊 Key Findings

- **Overall Trend**: -0.39 mm/year (gradual drying)
- **Model Performance**: XGBoost R² = 0.68, RMSE = 58.7 mm
- **Top Predictor**: Previous year's rainfall (24.5% importance)
- **Regional Divergence**: Southern Africa drying fastest (-0.75 mm/year)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/climate-rainfall-analysis.git
cd climate-rainfall-analysis

# Install dependencies
pip install -r requirements.txt
```

### Run Analysis

```bash
# 1. Process data
python scripts/process_data.py

# 2. Train model
python scripts/train_model.py

# 3. Launch dashboard
streamlit run scripts/dashboard.py
```

## 📁 Project Structure

```
climate-project/
├── data/                      # Data storage
│   ├── raw/                   # Original downloads
│   ├── processed/             # Cleaned data
│   └── results/               # Model outputs
├── notebooks/                 # Jupyter notebooks
├── scripts/                   # Executable scripts
├── src/                       # Source code modules
│   ├── data/                  # Data processing
│   ├── features/              # Feature engineering
│   ├── models/                # ML models
│   ├── evaluation/            # Model evaluation
│   ├── visualization/         # Plotting functions
│   ├── api/                   # REST API
│   └── utils/                 # Utilities
├── models/                    # Saved models
├── results/                   # Analysis results
└── tests/                     # Unit tests
```

## 🔧 Technology Stack

- **Python 3.9+**
- **Data**: pandas, numpy
- **ML**: scikit-learn, XGBoost
- **Visualization**: matplotlib, seaborn, plotly
- **Dashboard**: Streamlit
- **API**: FastAPI
- **Deployment**: Docker

## 📈 Usage Examples

### Load and Analyze Data

```python
import pandas as pd
from src.data.loaders import load_climate_data

# Load data
df = load_climate_data('data/processed/climate_data.csv')

# Quick analysis
from src.evaluation.metrics import calculate_trends
trends = calculate_trends(df, 'precipitation(mm)')
print(trends)
```

### Make Predictions

```python
from src.models.predictor import RainfallPredictor

# Load trained model
predictor = RainfallPredictor.load('models/xgboost_model.pkl')

# Predict rainfall
prediction = predictor.predict(
    country='Nigeria',
    year=2025,
    temp=26.5,
    humidity=65.0,
    co2=420.0
)
print(f"Predicted rainfall: {prediction:.1f} mm")
```

### Run API

```bash
# Start API server
uvicorn src.api.main:app --reload

# Test endpoint
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"country": "Kenya", "year": 2025, "avg_temp_c": 24.5}'
```

## 📊 Data Sources

- **Precipitation**: CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data)
- **Temperature, Humidity, Wind**: ERA5 Reanalysis
- **CO₂**: NOAA Global Monitoring Laboratory
- **Coverage**: 49 countries, 1991-2023, annual resolution

## 🤖 Models

| Model              | RMSE (mm) | MAE (mm) | R²    |
|-------------------|-----------|----------|-------|
| Linear Regression | 78.5      | 61.3     | 0.42  |
| Random Forest     | 62.3      | 48.1     | 0.64  |
| **XGBoost**       | **58.7**  | **45.2** | **0.68** |

## 📝 Key Features

### Temporal Features
- Year, decade, cyclical encoding
- Lag features (1-3 years)
- Rolling averages (3, 5 years)

### Spatial Features
- Regional aggregations
- Deviation from regional mean
- Country rankings

### Interaction Features
- Temperature × Humidity
- Temperature × CO₂
- Humidity × Cloud Cover

## 🎨 Visualizations

The project includes:
- Temporal trend analysis
- Regional comparison maps
- Correlation matrices
- Extreme event detection
- Model performance plots
- Interactive dashboards

## 🔬 Research Questions

1. How has annual rainfall changed over time?
2. Which regions show the strongest trends?
3. What variables predict rainfall best?
4. Are extreme events becoming more frequent?
5. How do patterns differ across decades?

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Data Processing](docs/data_processing.md)
- [Model Training](docs/model_training.md)
- [API Reference](docs/api_reference.md)
- [Contributing](docs/contributing.md)

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_models/

# With coverage
pytest --cov=src tests/
```

## 🐳 Docker Deployment

```bash
# Build image
docker build -t climate-rainfall-api .

# Run container
docker run -p 8000:8000 climate-rainfall-api

# Using docker-compose
docker-compose up -d
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - [GitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- CHIRPS team for precipitation data
- ERA5 for atmospheric variables
- NOAA for global climate indicators
- Climate research community

## 📧 Contact

- Email: your.email@example.com
- Project Link: https://github.com/yourusername/climate-rainfall-analysis

## 🔗 Related Resources

- [CHIRPS Data](https://www.chc.ucsb.edu/data/chirps)
- [ERA5 Documentation](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)
- [Climate Data Guide](https://climatedataguide.ucar.edu/)

---

**Note**: This is a research project. Results should be validated with domain experts before use in policy decisions.
