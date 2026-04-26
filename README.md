<img width="1894" height="632" alt="image" src="https://github.com/user-attachments/assets/86a17493-30dd-4cfd-85e1-f78b18bdc766" />
<img width="1603" height="765" alt="image" src="https://github.com/user-attachments/assets/1c724756-1799-4564-b48d-664a1cfd72df" />
<img width="1497" height="935" alt="image" src="https://github.com/user-attachments/assets/61783336-f756-4b3f-9eab-da4b43c644c2" />




# Smart Home Energy Consumption Forecasting

End-to-end mini project to forecast smart home energy usage using classic ML and optional LSTM.

## Project Structure
```
smart_energy_forecasting/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── plots/
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── generate_sample_data.py
│   ├── models.py
│   ├── predict.py
│   └── train.py
├── run_all.py
├── run.ps1
├── requirements.txt
└── README.md
```

## Quickstart (Windows PowerShell)

1) Open PowerShell in this folder.

2) Run the one-liner below to create a virtual env, install dependencies, generate sample data (90 days), and train a Random Forest model with 24-hour horizon:

```
./run.ps1
```

This will create artifacts in `models/` and plots in `plots/`.

If execution policy blocks scripts, run:
```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
./run.ps1
```

## Alternative: Run with Python

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_all.py --days 90 --model random_forest --horizon 24
```

## Frontend (Streamlit UI)

You can run a simple web UI to generate data, train models, and produce forecasts interactively.

Quick start:

```
./run_frontend.ps1
```

Then open the app in your browser at:

- http://localhost:8501

From the UI you can:
- Generate sample data or upload your CSV.
- Choose model type and forecast horizon, optionally enable hyperparameter tuning.
- Train the model and view metrics and prediction plots.
- Generate multi-step forecasts and view the forecast plot/table.

Supported models (choose with `--model`):
- random_forest (default)
- gradient_boosting
- linear, lasso, ridge
- xgboost (optional dependency)
- lightgbm (optional dependency)
- lstm (optional dependency: tensorflow)

To use optional models, install the extra package, e.g.:
```
pip install xgboost
# or
pip install lightgbm
# or
pip install tensorflow
```

## Using your own dataset

- CSV must include a `timestamp` column and a target column (default `energy_consumption`).
- Update and run:
```
python -m src.train --data path\to\your.csv --model random_forest --target energy_consumption --horizon 24
```

## Make future forecasts from a saved model

Find latest model and processor files in `models/`, then:
```
python -m src.predict --model models/random_forest_model_YYYYMMDD_HHMMSS.pkl --processor models/processor_YYYYMMDD_HHMMSS.pkl --steps 168 --output forecasts.csv
```

## Notes
- Feature engineering in `src/data_processing.py` adds time features, lags, and rolling windows.
- Scaling is handled consistently; predictions are inverse-transformed to original units.
- LSTM is supported but optional; by default, classic ML is faster and simpler to run.

## Troubleshooting
- If `ModuleNotFoundError: src...`, ensure you run from the project root. For scripts inside `src/`, we use absolute imports (`from src...`).
- On first run, large packages (like numpy) may take time to compile wheels.
- If you hit memory limits training LSTM, prefer `random_forest` or `gradient_boosting`.
# Smart-home-energy-consumption

