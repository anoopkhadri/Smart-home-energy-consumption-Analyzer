import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from src.data_processing import EnergyDataProcessor
from src.models import EnergyConsumptionModels, get_default_param_grid
import joblib

# Set random seeds for reproducibility
def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

def load_and_preprocess_data(data_path, target_column='energy_consumption', forecast_horizon=24):
    """Load and preprocess the data."""
    print("Loading and preprocessing data...")
    processor = EnergyDataProcessor(data_path)
    data = processor.load_data()
    df_features = processor.create_features(target_column=target_column, forecast_horizon=forecast_horizon)
    
    # Prepare training data
    X_train, X_val, X_test, y_train, y_val, y_test, y_train_orig, y_val_orig, y_test_orig = processor.prepare_training_data(
        df_features, target_column='target', test_size=0.2, val_size=0.1
    )
    
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'y_train_orig': y_train_orig, 'y_val_orig': y_val_orig, 'y_test_orig': y_test_orig,
        'processor': processor, 'data': data, 'df_features': df_features
    }

def train_model(data_dict, model_type='random_forest', tune_hyperparams=False):
    """Train the specified model."""
    print(f"\nTraining {model_type} model...")
    
    # Get data
    X_train, X_val, X_test = data_dict['X_train'], data_dict['X_val'], data_dict['X_test']
    y_train, y_val, y_test = data_dict['y_train'], data_dict['y_val'], data_dict['y_test']
    processor = data_dict['processor']
    
    # Initialize model
    input_shape = (1, X_train.shape[1]) if model_type == 'lstm' else None
    model = EnergyConsumptionModels(model_type=model_type, input_shape=input_shape)
    
    # Hyperparameter tuning if requested
    if tune_hyperparams and model_type in ['random_forest', 'xgboost', 'lightgbm']:
        print(f"Performing hyperparameter tuning for {model_type}...")
        param_grid = get_default_param_grid(model_type)
        best_params = model.hyperparameter_tuning(
            np.vstack([X_train, X_val]),  # Combine train and val for CV
            np.concatenate([y_train, y_val]),
            param_grid
        )
        print(f"Best parameters: {best_params}")
    
    # Train the model
    if model_type == 'lstm':
        model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
    else:
        # For non-LSTM models, combine train and validation sets
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])
        model.train(X_train_full, y_train_full)
    
    # Evaluate on test set
    test_metrics = model.evaluate(X_test, y_test, processor.scaler)
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'models/{model_type}_model_{timestamp}.pkl'
    
    if model_type == 'lstm':
        # Save Keras model
        model.model.save(f'models/{model_type}_model_{timestamp}.h5')
    else:
        # Save scikit-learn model
        joblib.dump(model.model, model_path)
    
    # Save the processor for later use in predictions
    joblib.dump(processor, f'models/processor_{timestamp}.pkl')
    
    # Save test metrics (make JSON-serializable)
    serializable_metrics = {}
    for k, v in test_metrics.items():
        if k in ('y_true', 'y_pred'):
            # Skip large arrays in JSON; they are available in memory if needed
            continue
        # Cast numpy types to Python types
        try:
            if hasattr(v, 'item'):
                serializable_metrics[k] = v.item()
            else:
                serializable_metrics[k] = float(v)
        except Exception:
            serializable_metrics[k] = v

    with open(f'models/{model_type}_metrics_{timestamp}.json', 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"\nTest Metrics for {model_type}:")
    for metric, value in test_metrics.items():
        if metric not in ['y_true', 'y_pred']:
            print(f"{metric.upper()}: {value:.4f}")
    
    return model, test_metrics

def plot_predictions(y_true, y_pred, model_name, save_path=None):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:200], label='Actual', alpha=0.7)
    plt.plot(y_pred[:200], label='Predicted', alpha=0.7)
    plt.title(f'Actual vs Predicted Energy Consumption - {model_name}')
    plt.xlabel('Time Step')
    plt.ylabel('Energy Consumption')
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train energy consumption forecasting model')
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--model', type=str, default='random_forest',
                       choices=['random_forest', 'xgboost', 'lstm', 'linear', 'lasso', 'ridge', 'gradient_boosting', 'lightgbm'],
                       help='Model type to train')
    parser.add_argument('--target', type=str, default='energy_consumption',
                       help='Name of the target column')
    parser.add_argument('--horizon', type=int, default=24,
                       help='Forecast horizon in hours')
    parser.add_argument('--tune', action='store_true',
                       help='Whether to perform hyperparameter tuning')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds
    set_seeds(args.seed)
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    try:
        # Load and preprocess data
        data_dict = load_and_preprocess_data(
            args.data, 
            target_column=args.target,
            forecast_horizon=args.horizon
        )
        
        # Train model
        model, metrics = train_model(
            data_dict, 
            model_type=args.model,
            tune_hyperparams=args.tune
        )
        
        # Plot predictions
        y_test = data_dict['y_test_orig']
        # Get predictions (scaled space)
        y_pred_scaled = model.predict(data_dict['X_test'])
        if hasattr(y_pred_scaled, 'flatten'):
            y_pred_scaled = y_pred_scaled.flatten()

        # Inverse transform predictions to original scale
        y_pred = data_dict['processor'].inverse_transform_target(np.array(y_pred_scaled))
        
        # Save prediction plot
        plot_save_path = f'plots/{args.model}_predictions.png'
        plot_predictions(y_test, y_pred, args.model, plot_save_path)
        print(f"\nPrediction plot saved to {plot_save_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
