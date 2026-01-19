"""
Data transformation utilities
"""

import pandas as pd
import numpy as np
from typing import Optional, List


def clean_data(df: pd.DataFrame, 
               drop_duplicates: bool = True,
               fill_method: Optional[str] = 'ffill') -> pd.DataFrame:
    """
    Clean climate data
    
    Args:
        df: Input DataFrame
        drop_duplicates: Remove duplicate country-year combinations
        fill_method: Method for filling missing values ('ffill', 'bfill', 'mean', None)
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    print(f"Initial shape: {df.shape}")
    
    # Remove duplicates
    if drop_duplicates and 'country' in df.columns and 'year' in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=['country', 'year'])
        removed = before - len(df)
        if removed > 0:
            print(f"Removed {removed} duplicate records")
    
    # Handle missing values
    if fill_method:
        if fill_method in ['ffill', 'bfill']:
            df = df.sort_values(['country', 'year'])
            if fill_method == 'ffill':
                df = df.groupby('country', group_keys=False).apply(lambda g: g.ffill())
            else:
                df = df.groupby('country', group_keys=False).apply(lambda g: g.bfill())
        elif fill_method == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        print(f"Filled missing values using: {fill_method}")
    
    # Validate ranges
    if 'precipitation(mm)' in df.columns:
        df = df[df['precipitation(mm)'] >= 0]
    
    if 'avg_temp_c' in df.columns:
        df = df[(df['avg_temp_c'] >= -50) & (df['avg_temp_c'] <= 60)]
    
    if 'avg_humidity(%)' in df.columns:
        df['avg_humidity(%)'] = df['avg_humidity(%)'].clip(0, 100)
    
    if 'cloud_cover(%)' in df.columns:
        df['cloud_cover(%)'] = df['cloud_cover(%)'].clip(0, 100)
    
    print(f"Final shape: {df.shape}")
    
    return df


def add_regional_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add regional classification to countries
    
    Args:
        df: DataFrame with 'country' column
        
    Returns:
        DataFrame with 'region' column added
    """
    region_map = {
        # West Africa
        'Benin': 'West Africa', 'Burkina Faso': 'West Africa', 'Cape Verde': 'West Africa',
        'Ivory Coast': 'West Africa', 'Gambia': 'West Africa', 'Ghana': 'West Africa',
        'Guinea': 'West Africa', 'Guinea-Bissau': 'West Africa', 'Liberia': 'West Africa',
        'Mali': 'West Africa', 'Mauritania': 'West Africa', 'Niger': 'West Africa',
        'Nigeria': 'West Africa', 'Senegal': 'West Africa', 'Sierra Leone': 'West Africa',
        'Togo': 'West Africa',
        
        # East Africa
        'Burundi': 'East Africa', 'Comoros': 'East Africa', 'Djibouti': 'East Africa',
        'Eritrea': 'East Africa', 'Ethiopia': 'East Africa', 'Kenya': 'East Africa',
        'Madagascar': 'East Africa', 'Mauritius': 'East Africa', 'Rwanda': 'East Africa',
        'Seychelles': 'East Africa', 'Somalia': 'East Africa', 'South Sudan': 'East Africa',
        'Sudan': 'East Africa', 'Tanzania': 'East Africa', 'Uganda': 'East Africa',
        
        # Central Africa
        'Angola': 'Central Africa', 'Cameroon': 'Central Africa',
        'Central African Republic': 'Central Africa', 'Chad': 'Central Africa',
        'Congo': 'Central Africa', 'Democratic Republic of Congo': 'Central Africa',
        'DRC': 'Central Africa', 'Equatorial Guinea': 'Central Africa',
        'Gabon': 'Central Africa', 'Sao Tome and Principe': 'Central Africa',
        
        # Southern Africa
        'Botswana': 'Southern Africa', 'Eswatini': 'Southern Africa',
        'Lesotho': 'Southern Africa', 'Malawi': 'Southern Africa',
        'Mozambique': 'Southern Africa', 'Namibia': 'Southern Africa',
        'South Africa': 'Southern Africa', 'Zambia': 'Southern Africa',
        'Zimbabwe': 'Southern Africa'
    }
    
    df = df.copy()
    df['region'] = df['country'].map(region_map)
    
    # Check for unmapped countries
    unmapped = df[df['region'].isnull()]['country'].unique()
    if len(unmapped) > 0:
        print(f"Warning: {len(unmapped)} countries not mapped to regions: {unmapped}")
    
    return df


def aggregate_to_annual(df: pd.DataFrame, 
                       date_col: str = 'date',
                       value_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Aggregate data to annual level
    
    Args:
        df: DataFrame with datetime column
        date_col: Name of date column
        value_cols: Columns to aggregate (if None, aggregate all numeric)
        
    Returns:
        Annually aggregated DataFrame
    """
    df = df.copy()
    
    # Extract year
    if date_col in df.columns:
        df['year'] = pd.to_datetime(df[date_col]).dt.year
    
    # Determine columns to aggregate
    if value_cols is None:
        value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        value_cols = [col for col in value_cols if col != 'year']
    
    # Group by country and year
    if 'country' in df.columns:
        agg_df = df.groupby(['country', 'year'])[value_cols].mean().reset_index()
    else:
        agg_df = df.groupby('year')[value_cols].mean().reset_index()
    
    return agg_df


def normalize_features(df: pd.DataFrame, 
                      columns: Optional[List[str]] = None,
                      method: str = 'standard') -> pd.DataFrame:
    """
    Normalize numerical features
    
    Args:
        df: Input DataFrame
        columns: Columns to normalize (if None, normalize all numeric)
        method: Normalization method ('standard', 'minmax')
        
    Returns:
        DataFrame with normalized features
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col in df.columns:
            if method == 'standard':
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[f'{col}_normalized'] = (df[col] - mean) / std
            elif method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val)
    
    return df
