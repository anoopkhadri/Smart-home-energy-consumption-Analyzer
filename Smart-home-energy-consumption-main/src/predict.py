import os
import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class EnergyConsumptionPredictor:
    def __init__(self, model_path, processor_path):
        """
        Initialize the predictor with trained model and processor.
        
        Args:
            model_path (str): Path to the trained model file
            processor_path (str): Path to the data processor file
        """
        self.model_path = model_path
        self.processor_path = processor_path
        self.model = None
        self.processor = None
        
        # Load model and processor
        self._load_model_and_processor()
    
    def _load_model_and_processor(self):
        """Load the trained model and processor."""
        # Load processor
        self.processor = joblib.load(self.processor_path)
        
        # Load model based on file extension
        if self.model_path.endswith('.h5'):
            from tensorflow.keras.models import load_model
            self.model = load_model(self.model_path)
            self.model_type = 'lstm'
        else:
            self.model = joblib.load(self.model_path)
            self.model_type = 'ml'
    
    def prepare_input_data(self, input_data):
        """
        Prepare input data for prediction.
        
        Args:
            input_data (pd.DataFrame): Input data with features
            
        Returns:
            np.array: Prepared input data for the model
        """
        # If input is a single row, convert to DataFrame
        if isinstance(input_data, (list, dict, np.ndarray)):
            input_data = pd.DataFrame(input_data)
        
        # Ensure timestamp is set as index if present
        if 'timestamp' in input_data.columns:
            input_data['timestamp'] = pd.to_datetime(input_data['timestamp'])
            input_data.set_index('timestamp', inplace=True)
        
        # Create features (without target shift for inference)
        df_features = self.processor.create_features(include_target=False)
        
        # Get the most recent rows for prediction
        if len(df_features) > 0:
            # Get the last row for making next prediction
            X_pred = df_features.iloc[[-1]].drop(columns=['target'], errors='ignore')
            # Scale features
            X_pred_scaled = self.processor.feature_scaler.transform(X_pred)
            
            # Reshape for LSTM if needed
            if self.model_type == 'lstm':
                X_pred_scaled = X_pred_scaled.reshape((1, 1, X_pred_scaled.shape[1]))
                # Use the last available timestamp as context
            return X_pred_scaled, X_pred.index[0]
        
        return None, None
    
    def predict(self, input_data):
        """
        Make predictions using the trained model.
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            dict: Prediction results with timestamp and energy consumption
        """
        X_pred, timestamp = self.prepare_input_data(input_data)
        
        if X_pred is None:
            return {"error": "Could not prepare input data for prediction"}
        
        # Make prediction
        if self.model_type == 'lstm':
            y_pred_scaled = self.model.predict(X_pred)[0][0]
        else:
            y_pred_scaled = self.model.predict(X_pred)[0]
        
        # Inverse transform the prediction to original scale
        y_pred = float(self.processor.inverse_transform_target(np.array([y_pred_scaled]))[0])
        
        # Create prediction timestamp (next hour)
        if timestamp is None:
            timestamp = datetime.now()
        else:
            # Get the last timestamp and add one hour
            timestamp = timestamp + pd.Timedelta(hours=1)
        
        return {
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'energy_consumption': float(y_pred),
            'model': os.path.basename(self.model_path)
        }
    
    def forecast(self, steps=24, initial_data=None):
        """
        Generate multi-step forecasts.
        
        Args:
            steps (int): Number of steps to forecast
            initial_data (pd.DataFrame): Initial data to start forecasting from
            
        Returns:
            pd.DataFrame: DataFrame with timestamps and forecasted values
        """
        if initial_data is None:
            # Use the last available data point from the stored processor data
            df_features = self.processor.create_features(include_target=False)
            initial_data = df_features.iloc[[-1]]
        
        forecasts = []
        current_data = initial_data.copy()

        # Establish a deterministic starting timestamp (last known point)
        if self.processor.data is not None and len(self.processor.data) > 0:
            base_last_ts = self.processor.data.index[-1]
        else:
            base_last_ts = current_data.index[-1]
        
        for step in range(steps):
            # Make prediction for next step
            prediction = self.predict(current_data)
            
            if 'error' in prediction:
                print(f"Error in prediction: {prediction['error']}")
                break
                
            # Advance timestamp deterministically by 1 hour each step
            next_ts = pd.to_datetime(base_last_ts) + pd.Timedelta(hours=step + 1)

            # Override returned timestamp to our scheduled next_ts
            prediction['timestamp'] = next_ts.strftime('%Y-%m-%d %H:%M:%S')

            forecasts.append(prediction)

            # Build a minimal next row to extend the underlying data used for features
            next_energy = prediction['energy_consumption']

            # Append the predicted point to the processor's underlying data to roll features forward
            base_cols = ['energy_consumption']
            new_row = pd.DataFrame({'energy_consumption': [next_energy]}, index=[next_ts])

            # Ensure processor has a DataFrame initialized
            if self.processor.data is None or len(self.processor.data) == 0:
                self.processor.data = new_row.copy()
            else:
                # Concat and keep sorted by timestamp
                self.processor.data = pd.concat([self.processor.data, new_row])
                self.processor.data = self.processor.data[~self.processor.data.index.duplicated(keep='last')]
                self.processor.data.sort_index(inplace=True)

            # Recreate features (without target) and set current_data to the last row context
            df_features = self.processor.create_features(include_target=False)
            current_data = df_features.iloc[[-1]]
        
        return pd.DataFrame(forecasts)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Make predictions using a trained energy consumption model')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--processor', type=str, required=True, help='Path to the processor file')
    parser.add_argument('--steps', type=int, default=24, help='Number of steps to forecast')
    parser.add_argument('--output', type=str, help='Output file path for saving predictions')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = EnergyConsumptionPredictor(args.model, args.processor)
        
        # Generate forecasts
        print(f"Generating {args.steps}-step forecast...")
        forecasts = predictor.forecast(steps=args.steps)
        
        # Print forecasts
        print("\nForecasted Energy Consumption:")
        print(forecasts[['timestamp', 'energy_consumption']].to_string(index=False))
        
        # Save to file if output path is provided
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            forecasts.to_csv(args.output, index=False)
            print(f"\nForecasts saved to {args.output}")
        
        # Plot the forecast
        plt.figure(figsize=(12, 6))
        plt.plot(pd.to_datetime(forecasts['timestamp']), forecasts['energy_consumption'], 
                marker='o', linestyle='-', color='b')
        plt.title(f'Energy Consumption Forecast ({args.steps} hours)')
        plt.xlabel('Time')
        plt.ylabel('Energy Consumption')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_path = args.output.replace('.csv', '.png') if args.output else 'forecast_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Forecast plot saved to {plot_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
