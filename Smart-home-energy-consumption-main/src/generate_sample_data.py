import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_data(num_days=365, freq='h'):
    """
    Generate sample energy consumption data with daily and weekly seasonality.
    
    Args:
        num_days (int): Number of days to generate data for
        freq (str): Frequency of data points ('h' for hourly, '30T' for 30 minutes, etc.)
        
    Returns:
        pd.DataFrame: Generated time series data
    """
    # Create datetime index
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days)
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Base consumption (daily pattern)
    hours = (date_range.hour + date_range.minute / 60).values
    daily_pattern = 2 + np.sin(2 * np.pi * (hours - 9) / 24)  # Peak during the day
    
    # Weekly pattern (higher on weekdays)
    day_of_week = date_range.dayofweek.values
    weekly_pattern = np.where(day_of_week < 5, 1.2, 0.8)  # Higher on weekdays
    
    # Temperature effect (simulated)
    temp_effect = 0.5 * np.sin(2 * np.pi * (date_range.dayofyear / 365)) + 0.5  # Seasonal variation
    
    # Random noise
    noise = np.random.normal(0, 0.1, size=len(date_range))
    
    # Generate energy consumption
    energy = 10 * daily_pattern * weekly_pattern * (1 + 0.3 * temp_effect) * (1 + 0.1 * noise)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': date_range,
        'energy_consumption': np.maximum(energy, 0.1),  # Ensure no negative values
        'temperature': 15 + 15 * temp_effect + np.random.normal(0, 2, size=len(date_range)),
        'humidity': 50 + 30 * np.sin(2 * np.pi * (hours - 6) / 24) + np.random.normal(0, 5, size=len(date_range)),
        'occupancy': np.random.poisson(0.5, size=len(date_range))  # Simulated occupancy
    })
    
    # Add some anomalies
    anomaly_indices = np.random.choice(len(df), size=len(df)//100, replace=False)  # ~1% anomalies
    df.loc[anomaly_indices, 'energy_consumption'] *= np.random.uniform(1.5, 3, size=len(anomaly_indices))
    
    return df

def save_sample_data(filepath='data/raw/sample_energy_data.csv', num_days=365):
    """Generate and save sample data to a CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = generate_sample_data(num_days=num_days)
    df.to_csv(filepath, index=False)
    print(f"Sample data saved to {filepath}")

if __name__ == "__main__":
    save_sample_data()
