import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

class WildfireDataProcessor:
    """
    Data processor for wildfire prediction dataset.
    Handles data loading, preprocessing, and feature engineering.
    """
    def __init__(self, lag_days=15, test_size=0.2, random_state=42):
        self.lag_days = lag_days
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = RobustScaler()
        
    def load_data(self, file_path):
        """
        Load and perform initial preprocessing of the dataset.
        """
        df = pd.read_parquet(file_path)
        
        # Convert date column to datetime
        df['acq_date'] = pd.to_datetime(df['acq_date'])
        return df
    
    def create_temporal_features(self, df):
        """
        Create temporal features including cyclical time encodings.
        """
        # Day of year features (cyclical encoding)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Month features (cyclical encoding)
        df['month'] = df['acq_date'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def create_lagged_features(self, df, columns):
        """
        Create lagged features for specified columns.
        """
        for col in columns:
            for lag in range(1, self.lag_days + 1):
                df[f'{col}_lag_{lag}'] = df.groupby(['latitude', 'longitude'])[col].shift(lag)
        return df
    
    def create_rolling_features(self, df, columns):
        """
        Create rolling statistics features.
        """
        windows = [7, 14, 30]  # Rolling windows in days
        
        for col in columns:
            for window in windows:
                df[f'{col}_rolling_mean_{window}d'] = df.groupby(
                    ['latitude', 'longitude'])[col].rolling(window).mean().reset_index(0, drop=True)
                df[f'{col}_rolling_std_{window}d'] = df.groupby(
                    ['latitude', 'longitude'])[col].rolling(window).std().reset_index(0, drop=True)
        
        return df
    
    def prepare_features(self, df):
        """
        Prepare features for model training.
        """
        # Temporal features
        df = self.create_temporal_features(df)
        
        # Weather-related columns for lagging
        weather_columns = [
            'average_temperature', 'precipitation',
            'wind_speed', 'relative_humidity'
        ]
        
        # Create lagged features
        df = self.create_lagged_features(df, weather_columns)
        
        # Create rolling features
        df = self.create_rolling_features(df, weather_columns)
        
        return df
    
    def split_data(self, df, target_col='is_fire'):
        """
        Split data into training and validation sets.
        """
        # Sort by date
        df = df.sort_values('acq_date')
        
        # Split chronologically
        train_df = df[df['acq_date'] < '2022-01-01']
        valid_df = df[df['acq_date'] >= '2022-01-01']
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ['acq_date', target_col]]
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_valid = valid_df[feature_cols]
        y_valid = valid_df[target_col]
        
        return X_train, X_valid, y_train, y_valid
    
    def scale_features(self, X_train, X_valid):
        """
        Scale features using RobustScaler.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_valid_scaled = self.scaler.transform(X_valid)
        
        return X_train_scaled, X_valid_scaled
    
    def prepare_data(self, file_path):
        """
        Complete data preparation pipeline.
        """
        # Load data
        df = self.load_data(file_path)
        
        # Prepare features
        df = self.prepare_features(df)
        
        # Split data
        X_train, X_valid, y_train, y_valid = self.split_data(df)
        
        # Scale features
        X_train_scaled, X_valid_scaled = self.scale_features(X_train, X_valid)
        
        return X_train_scaled, X_valid_scaled, y_train, y_valid

def load_and_prepare_data(file_path):
    """
    Convenience function to load and prepare data.
    """
    processor = WildfireDataProcessor()
    return processor.prepare_data(file_path)