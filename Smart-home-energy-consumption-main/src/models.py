import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

class EnergyConsumptionModels:
    def __init__(self, model_type='random_forest', input_shape=None):
        """
        Initialize the model.
        
        Args:
            model_type (str): Type of model to use. Options: 'random_forest', 'xgboost', 'lstm', 'linear', 'lasso', 'ridge', 'gradient_boosting', 'lightgbm'
            input_shape (tuple): Shape of input data (required for LSTM)
        """
        self.model_type = model_type
        self.model = None
        self.input_shape = input_shape
        self.scaler = None
        self.history = None
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == 'xgboost':
            try:
                import xgboost as xgb
            except ImportError as e:
                raise ImportError("xgboost is not installed. Install it with 'pip install xgboost' or choose another model.") from e
            self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        elif model_type == 'lightgbm':
            try:
                import lightgbm as lgb
            except ImportError as e:
                raise ImportError("lightgbm is not installed. Install it with 'pip install lightgbm' or choose another model.") from e
            self.model = lgb.LGBMRegressor(objective='regression', random_state=42)
        elif model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'lasso':
            self.model = Lasso(alpha=0.1, random_state=42)
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0, random_state=42)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == 'lstm':
            if input_shape is None:
                raise ValueError("input_shape must be provided for LSTM model")
            self.model = self._build_lstm_model(input_shape)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _build_lstm_model(self, input_shape):
        """Build an LSTM model for time series forecasting."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
        except ImportError as e:
            raise ImportError("TensorFlow is not installed. Install it with 'pip install tensorflow' to use LSTM model.") from e

        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            **kwargs: Additional arguments for model training
            
        Returns:
            The trained model
        """
        if self.model_type == 'lstm':
            # Reshape input for LSTM [samples, timesteps, features]
            if len(X_train.shape) == 2:
                X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                if X_val is not None:
                    X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
            
            try:
                from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
            except ImportError:
                EarlyStopping = ModelCheckpoint = None

            callbacks = None
            if EarlyStopping and ModelCheckpoint and X_val is not None:
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')
                ]
            
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=kwargs.get('epochs', 100),
                batch_size=kwargs.get('batch_size', 32),
                callbacks=callbacks,
                verbose=1
            )
        else:
            self.model.fit(X_train, y_train)
            
        return self.model
    
    def predict(self, X):
        """Make predictions using the trained model."""
        if self.model_type == 'lstm' and len(X.shape) == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test, scaler=None):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: True test values
            scaler: Scaler object to inverse transform the predictions (optional)
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        # Inverse transform if scaler is provided
        if scaler is not None:
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'y_true': y_test,
            'y_pred': y_pred
        }
    
    def hyperparameter_tuning(self, X_train, y_train, param_dist, n_iter=20, cv=3):
        """
        Perform hyperparameter tuning using RandomizedSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training target
            param_dist: Dictionary of hyperparameter distributions
            n_iter: Number of parameter settings to sample
            cv: Number of cross-validation folds
            
        Returns:
            dict: Best parameters found
        """
        if self.model_type == 'lstm':
            print("Hyperparameter tuning not implemented for LSTM. Using default parameters.")
            return {}
            
        random_search = RandomizedSearchCV(
            self.model, 
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=42
        )
        
        random_search.fit(X_train, y_train)
        self.model = random_search.best_estimator_
        
        return random_search.best_params_

def get_default_param_grid(model_type):
    """Get default parameter grid for hyperparameter tuning."""
    if model_type == 'random_forest':
        return {
            'n_estimators': sp_randint(50, 200),
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': sp_randint(2, 11),
            'min_samples_leaf': sp_randint(1, 5)
        }
    elif model_type == 'xgboost':
        return {
            'n_estimators': sp_randint(50, 200),
            'max_depth': sp_randint(3, 10),
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
    elif model_type == 'lightgbm':
        return {
            'num_leaves': sp_randint(20, 50),
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': sp_randint(50, 200),
            'min_child_samples': sp_randint(10, 30)
        }
    else:
        return {}
