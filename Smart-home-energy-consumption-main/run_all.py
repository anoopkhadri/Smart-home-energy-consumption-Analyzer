import os
import sys
import argparse

# Ensure src is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.generate_sample_data import save_sample_data
from src.train import set_seeds, load_and_preprocess_data, train_model


def main():
    parser = argparse.ArgumentParser(description="End-to-end runner for Smart Home Energy Forecasting")
    parser.add_argument("--days", type=int, default=90, help="Number of days to simulate for sample data")
    parser.add_argument("--model", type=str, default="random_forest", choices=[
        "random_forest", "xgboost", "lightgbm", "gradient_boosting", "linear", "lasso", "ridge", "lstm"
    ], help="Model type to train")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon in hours")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning where supported")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Create folders
    for d in ["data/raw", "data/processed", "models", "plots"]:
        os.makedirs(os.path.join(CURRENT_DIR, d), exist_ok=True)

    data_csv = os.path.join(CURRENT_DIR, "data/raw/sample_energy_data.csv")
    print(f"Generating sample data: {data_csv}")
    save_sample_data(filepath=data_csv, num_days=args.days)

    print("\nLoading data and training model...")
    set_seeds(args.seed)
    data_dict = load_and_preprocess_data(
        data_csv,
        target_column="energy_consumption",
        forecast_horizon=args.horizon,
    )

    model, metrics = train_model(
        data_dict,
        model_type=args.model,
        tune_hyperparams=args.tune,
    )

    print("\nFinished. Key metrics:")
    for k in ["rmse", "mae", "r2"]:
        v = metrics.get(k)
        if v is not None and not isinstance(v, (list, tuple)):
            print(f"{k.upper()}: {v:.4f}")

    print("\nArtifacts saved:")
    print(f"- Models in {os.path.join(CURRENT_DIR, 'models')}\n- Plots in {os.path.join(CURRENT_DIR, 'plots')}")


if __name__ == "__main__":
    main()
