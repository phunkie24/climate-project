"""
Feature engineering for climate data
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class FeatureEngineer:
    """Feature engineering pipeline for climate data"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.feature_names = []
    
    def create_temporal_features(self) -> 'FeatureEngineer':
        """Create time-based features"""
        # Basic temporal
        self.df['decade'] = (self.df['year'] // 10) * 10
        self.df['years_since_1991'] = self.df['year'] - 1991
        
        # Cyclical encoding
        self.df['year_in_decade'] = self.df['year'] % 10
        self.df['year_sin'] = np.sin(2 * np.pi * self.df['year_in_decade'] / 10)
        self.df['year_cos'] = np.cos(2 * np.pi * self.df['year_in_decade'] / 10)
        
        new_features = ['decade', 'years_since_1991', 'year_sin', 'year_cos']
        self.feature_names.extend(new_features)
        print(f"[OK] Created {len(new_features)} temporal features")
        
        return self
    
    def create_lag_features(self, 
                          target_col: str = 'precipitation(mm)',
                          lags: List[int] = [1, 2, 3]) -> 'FeatureEngineer':
        """Create lagged features"""
        self.df = self.df.sort_values(['country', 'year'])
        
        for lag in lags:
            col_name = f'{target_col.replace("(", "_").replace(")", "").replace("%", "pct")}_lag_{lag}'
            self.df[col_name] = self.df.groupby('country')[target_col].shift(lag)
            self.feature_names.append(col_name)
        
        print(f"[OK] Created {len(lags)} lag features")
        return self
    
    def create_rolling_features(self,
                               target_col: str = 'precipitation(mm)',
                               windows: List[int] = [3, 5]) -> 'FeatureEngineer':
        """Create rolling statistics"""
        self.df = self.df.sort_values(['country', 'year'])
        
        base_name = target_col.replace("(", "_").replace(")", "").replace("%", "pct")
        
        for window in windows:
            # Rolling mean
            mean_col = f'{base_name}_rolling_mean_{window}'
            self.df[mean_col] = self.df.groupby('country')[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            self.feature_names.append(mean_col)
            
            # Rolling std
            std_col = f'{base_name}_rolling_std_{window}'
            self.df[std_col] = self.df.groupby('country')[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            self.feature_names.append(std_col)
        
        print(f"[OK] Created {len(windows) * 2} rolling features")
        return self
    
    def create_change_features(self,
                             target_col: str = 'precipitation(mm)') -> 'FeatureEngineer':
        """Create change/delta features"""
        self.df = self.df.sort_values(['country', 'year'])
        
        base_name = target_col.replace("(", "_").replace(")", "").replace("%", "pct")
        
        # Absolute change
        change_col = f'{base_name}_change'
        self.df[change_col] = self.df.groupby('country')[target_col].diff()
        self.feature_names.append(change_col)
        
        # Percentage change
        pct_change_col = f'{base_name}_pct_change'
        self.df[pct_change_col] = self.df.groupby('country')[target_col].pct_change() * 100
        self.feature_names.append(pct_change_col)
        
        print(f"[OK] Created 2 change features")
        return self
    
    def create_interaction_features(self) -> 'FeatureEngineer':
        """Create interaction features between variables"""
        interactions = []
        
        if 'avg_temp_c' in self.df.columns and 'avg_humidity(%)' in self.df.columns:
            self.df['temp_humidity'] = self.df['avg_temp_c'] * self.df['avg_humidity(%)']
            interactions.append('temp_humidity')
        
        if 'avg_temp_c' in self.df.columns and 'atmospheric_co2(ppm)' in self.df.columns:
            self.df['temp_co2'] = self.df['avg_temp_c'] * self.df['atmospheric_co2(ppm)']
            interactions.append('temp_co2')
        
        if 'avg_humidity(%)' in self.df.columns and 'cloud_cover(%)' in self.df.columns:
            self.df['humidity_cloud'] = self.df['avg_humidity(%)'] * self.df['cloud_cover(%)']
            interactions.append('humidity_cloud')
        
        self.feature_names.extend(interactions)
        print(f"[OK] Created {len(interactions)} interaction features")
        return self
    
    def create_regional_features(self,
                                target_col: str = 'precipitation(mm)') -> 'FeatureEngineer':
        """Create regional aggregation features"""
        if 'region' not in self.df.columns:
            print("[WARN] Warning: 'region' column not found, skipping regional features")
            return self
        
        base_name = target_col.replace("(", "_").replace(")", "").replace("%", "pct")
        
        # Regional mean
        regional_mean_col = f'regional_mean_{base_name}'
        regional_means = self.df.groupby(['region', 'year'])[target_col].transform('mean')
        self.df[regional_mean_col] = regional_means
        self.feature_names.append(regional_mean_col)
        
        # Deviation from regional mean
        deviation_col = f'deviation_from_regional_{base_name}'
        self.df[deviation_col] = self.df[target_col] - self.df[regional_mean_col]
        self.feature_names.append(deviation_col)
        
        # Regional rank
        rank_col = f'regional_rank_{base_name}'
        self.df[rank_col] = self.df.groupby(['region', 'year'])[target_col].rank(ascending=False)
        self.feature_names.append(rank_col)
        
        print(f"[OK] Created 3 regional features")
        return self
    
    def engineer_all(self) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        print("="*60)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        self.create_temporal_features()
        self.create_lag_features()
        self.create_rolling_features()
        self.create_change_features()
        self.create_interaction_features()
        self.create_regional_features()
        
        print("\n" + "="*60)
        print(f"COMPLETE: Created {len(self.feature_names)} new features")
        print(f"Total columns: {len(self.df.columns)}")
        print("="*60)
        
        return self.df
    
    def get_feature_names(self) -> List[str]:
        """Get list of created feature names"""
        return self.feature_names


def engineer_features(df: pd.DataFrame,
                     temporal: bool = True,
                     lags: Optional[List[int]] = [1, 2, 3],
                     rolling_windows: Optional[List[int]] = [3, 5],
                     changes: bool = True,
                     interactions: bool = True,
                     regional: bool = True) -> pd.DataFrame:
    """
    Convenience function for feature engineering
    
    Args:
        df: Input DataFrame
        temporal: Create temporal features
        lags: List of lag periods (None to skip)
        rolling_windows: List of rolling window sizes (None to skip)
        changes: Create change features
        interactions: Create interaction features
        regional: Create regional features
        
    Returns:
        DataFrame with engineered features
    """
    engineer = FeatureEngineer(df)
    
    if temporal:
        engineer.create_temporal_features()
    
    if lags:
        engineer.create_lag_features(lags=lags)
    
    if rolling_windows:
        engineer.create_rolling_features(windows=rolling_windows)
    
    if changes:
        engineer.create_change_features()
    
    if interactions:
        engineer.create_interaction_features()
    
    if regional:
        engineer.create_regional_features()
    
    return engineer.df
