import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

class EnergyDataProcessor:
    def __init__(self, data_path):
        """
        Initialize the EnergyDataProcessor with the path to the dataset.
        
        Args:
            data_path (str): Path to the CSV file containing energy consumption data
        """
        self.data_path = data_path
        self.data = None
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        
    def load_data(self):
        """Load and preprocess the energy consumption data."""
        # Load the dataset
        self.data = pd.read_csv(self.data_path)
        
        # Convert timestamp to datetime
        if 'timestamp' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data.set_index('timestamp', inplace=True)
        
        # Handle missing values
        self._handle_missing_values()
        
        return self.data
    
    def _handle_missing_values(self):
        """Handle missing values in the dataset."""
        # Forward fill for time series data
        self.data.ffill(inplace=True)
        self.data.bfill(inplace=True)  # Backward fill any remaining NaNs
    
    def create_features(self, target_column='energy_consumption', forecast_horizon=24, include_target=True):
        """
        Create time series features from the timestamp index.
        
        Args:
            target_column (str): Name of the target variable column
            forecast_horizon (int): Number of hours to forecast ahead
            include_target (bool): If True, create a shifted 'target' column for training. If False, omit target.
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        df = self.data.copy()
        
        # Extract time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['is_weekend'] = df.index.dayofweek >= 5
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7.0)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7.0)
        
        # Create lag features
        for lag in [1, 2, 3, 24, 48, 168]:  # 1h, 2h, 3h, 24h, 48h, 1 week lags
            df[f'lag_{lag}'] = df[target_column].shift(lag)
        
        # Create rolling statistics (use min_periods=1 so latest row is not NaN during inference)
        windows = [3, 24, 168]  # 3h, 24h, 1 week windows
        for window in windows:
            df[f'rolling_mean_{window}'] = df[target_column].rolling(window=window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df[target_column].rolling(window=window, min_periods=1).std()
        
        # Create target variable (shifted by forecast_horizon) for training only
        if include_target:
            df['target'] = df[target_column].shift(-forecast_horizon)

        # Drop rows with NaN values created by lag/rolling (and target if included)
        # During training (include_target=True), it's OK to drop NaNs. During inference, keep all rows.
        if include_target:
            df.dropna(inplace=True)
        
        return df
    
    def prepare_training_data(self, df, target_column='target', test_size=0.2, val_size=0.1, random_state=42):
        """
        Prepare train, validation, and test sets.
        
        Args:
            df (pd.DataFrame): DataFrame with features and target
            target_column (str): Name of the target column
            test_size (float): Proportion of data to use for testing
            val_size (float): Proportion of training data to use for validation
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Separate features and target
        X = df.drop(columns=[target_column, 'target'] if 'target' in df.columns and target_column != 'target' else target_column)
        y = df[target_column]
        
        # Split into train and test sets (temporal split for time series)
        train_size = 1 - test_size
        train_idx = int(len(X) * train_size)
        
        X_train_val, X_test = X.iloc[:train_idx], X.iloc[train_idx:]
        y_train_val, y_test = y.iloc[:train_idx], y.iloc[train_idx:]
        
        # Further split training data into train and validation sets
        val_idx = int(len(X_train_val) * (1 - val_size))
        X_train, X_val = X_train_val.iloc[:val_idx], X_train_val.iloc[val_idx:]
        y_train, y_val = y_train_val.iloc[:val_idx], y_train_val.iloc[val_idx:]
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Scale target
        y_train = y_train.values.reshape(-1, 1)
        y_val = y_val.values.reshape(-1, 1)
        y_test = y_test.values.reshape(-1, 1)
        
        y_train_scaled = self.scaler.fit_transform(y_train)
        y_val_scaled = self.scaler.transform(y_val)
        y_test_scaled = self.scaler.transform(y_test)
        
        return (
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_scaled.ravel(), y_val_scaled.ravel(), y_test_scaled.ravel(),
            y_train.ravel(), y_val.ravel(), y_test.ravel()
        )
    
    def inverse_transform_target(self, y_scaled):
        """Inverse transform the scaled target variable."""
        return self.scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
