import os
import sys
import glob
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure src is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.generate_sample_data import save_sample_data
from src.train import set_seeds, load_and_preprocess_data, train_model
from src.predict import EnergyConsumptionPredictor

st.set_page_config(page_title="Smart Home Energy Forecasting", layout="wide")
st.title("🏠 Smart Home Energy Consumption Forecasting")

# Sidebar controls
st.sidebar.header("Configuration")
model_type = st.sidebar.selectbox(
    "Model",
    ["random_forest", "gradient_boosting", "linear", "lasso", "ridge", "xgboost", "lightgbm", "lstm"],
    index=0,
)
forecast_horizon = st.sidebar.number_input("Forecast horizon (hours)", min_value=1, max_value=168, value=24)
tune = st.sidebar.checkbox("Hyperparameter tuning (where supported)", value=False)
seed = st.sidebar.number_input("Random seed", min_value=0, value=42)

# Data section
st.subheader("1) Data")
col1, col2 = st.columns(2)

data_source = col1.radio("Select data source", ["Generate sample data", "Upload CSV"], index=0)

uploaded_file = None
sample_days = 90
if data_source == "Generate sample data":
    sample_days = col2.slider("Days of sample data", min_value=14, max_value=730, value=90, step=7)
else:
    uploaded_file = col2.file_uploader("Upload CSV with 'timestamp' and 'energy_consumption' columns", type=["csv"]) 

run_training = st.button("Train Model", type="primary")

# Placeholder for outputs
metrics_placeholder = st.empty()
plot_placeholder = st.empty()
forecast_placeholder = st.empty()

if run_training:
    try:
        set_seeds(int(seed))
        os.makedirs(os.path.join(CURRENT_DIR, "data/raw"), exist_ok=True)
        os.makedirs(os.path.join(CURRENT_DIR, "models"), exist_ok=True)
        os.makedirs(os.path.join(CURRENT_DIR, "plots"), exist_ok=True)

        # Prepare data path
        if data_source == "Generate sample data":
            data_csv = os.path.join(CURRENT_DIR, "data/raw/sample_energy_data.csv")
            save_sample_data(filepath=data_csv, num_days=int(sample_days))
        else:
            if uploaded_file is None:
                st.error("Please upload a CSV file.")
                st.stop()
            data_csv = os.path.join(CURRENT_DIR, "data/raw", "uploaded_data.csv")
            df_up = pd.read_csv(uploaded_file)
            df_up.to_csv(data_csv, index=False)

        # Load and preprocess
        with st.spinner("Loading and preprocessing data..."):
            data_dict = load_and_preprocess_data(
                data_csv, target_column="energy_consumption", forecast_horizon=int(forecast_horizon)
            )

        # Train
        with st.spinner(f"Training {model_type} model..."):
            model, test_metrics = train_model(
                data_dict, model_type=model_type, tune_hyperparams=bool(tune)
            )

        # Show metrics
        st.success("Training complete")
        nice_metrics = {k: float(v) for k, v in test_metrics.items() if k not in ("y_true", "y_pred")}
        metrics_placeholder.subheader("Metrics")
        metrics_placeholder.json(nice_metrics)

        # Plot predictions inline
        y_test = data_dict['y_test_orig']
        y_pred_scaled = model.predict(data_dict['X_test'])
        if hasattr(y_pred_scaled, 'flatten'):
            y_pred_scaled = y_pred_scaled.flatten()
        y_pred = data_dict['processor'].inverse_transform_target(np.array(y_pred_scaled))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(y_test[:200], label='Actual', alpha=0.8)
        ax.plot(y_pred[:200], label='Predicted', alpha=0.8)
        ax.set_title(f'Actual vs Predicted - {model_type}')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Energy Consumption')
        ax.legend()
        plot_placeholder.pyplot(fig, clear_figure=True)

        # Save state
        st.session_state['last_model_type'] = model_type
        st.session_state['last_data_csv'] = data_csv

    except Exception as e:
        st.exception(e)

st.subheader("2) Forecast")
colf1, colf2 = st.columns([2,1])
forecast_steps = colf2.number_input("Forecast steps (hours)", min_value=1, max_value=7*24, value=24)
run_forecast = colf2.button("Generate Forecast")

if run_forecast:
    try:
        # Find latest model and processor saved by the training step
        def latest(pattern):
            files = glob.glob(os.path.join(CURRENT_DIR, pattern))
            if not files:
                return None
            return max(files, key=os.path.getctime)

        # Try model types in order of preference (use the last trained model type first if available)
        model_paths_patterns = []
        if 'last_model_type' in st.session_state:
            mt = st.session_state['last_model_type']
            ext = 'h5' if mt == 'lstm' else 'pkl'
            model_paths_patterns.append(f"models/{mt}_model_*.{ext}")
        # Fallbacks
        model_paths_patterns += [
            "models/random_forest_model_*.pkl",
            "models/gradient_boosting_model_*.pkl",
            "models/linear_model_*.pkl",
            "models/lasso_model_*.pkl",
            "models/ridge_model_*.pkl",
            "models/xgboost_model_*.pkl",
            "models/lightgbm_model_*.pkl",
            "models/lstm_model_*.h5",
        ]

        model_path = None
        for pat in model_paths_patterns:
            model_path = latest(pat)
            if model_path:
                break

        processor_path = latest("models/processor_*.pkl")
        if not model_path or not processor_path:
            st.error("No saved model/processor found. Train a model first.")
            st.stop()

        predictor = EnergyConsumptionPredictor(model_path, processor_path)
        with st.spinner("Generating forecast..."):
            forecast_df = predictor.forecast(steps=int(forecast_steps))

        # Show forecast
        if not forecast_df.empty:
            colf1.write(forecast_df[['timestamp', 'energy_consumption']])
            # Plot
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            ax2.plot(pd.to_datetime(forecast_df['timestamp']), forecast_df['energy_consumption'], marker='o')
            ax2.set_title(f"{forecast_steps}-hour Forecast")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Energy Consumption")
            fig2.autofmt_xdate(rotation=45)
            colf1.pyplot(fig2, clear_figure=True)
        else:
            st.warning("Forecast returned empty results.")

    except Exception as e:
        st.exception(e)
