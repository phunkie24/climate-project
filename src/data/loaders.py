"""
Data loading utilities
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union


def load_climate_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load climate data from CSV file
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with climate data
    """
    df = pd.read_csv(filepath)
    
    # Convert year to datetime if needed
    if 'year' in df.columns and df['year'].dtype in ['int64', 'int32']:
        df['date'] = pd.to_datetime(df['year'], format='%Y')
    
    return df


def load_multiple_files(data_dir: Union[str, Path], pattern: str = "*.csv") -> pd.DataFrame:
    """
    Load and concatenate multiple CSV files
    
    Args:
        data_dir: Directory containing files
        pattern: File pattern to match
        
    Returns:
        Concatenated DataFrame
    """
    data_dir = Path(data_dir)
    files = list(data_dir.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} found in {data_dir}")
    
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


def save_processed_data(df: pd.DataFrame, filepath: Union[str, Path], format: str = 'csv'):
    """
    Save processed data
    
    Args:
        df: DataFrame to save
        filepath: Output path
        format: File format ('csv' or 'parquet')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'csv':
        df.to_csv(filepath, index=False)
    elif format == 'parquet':
        df.to_parquet(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"[OK] Saved data to {filepath}")


def get_country_list(region: Optional[str] = None) -> list:
    """
    Get list of Sub-Saharan African countries
    
    Args:
        region: Optional region filter (West/East/Central/Southern Africa)
        
    Returns:
        List of country names
    """
    countries = {
        'West Africa': [
            'Benin', 'Burkina Faso', 'Cape Verde', 'Ivory Coast', 'Gambia',
            'Ghana', 'Guinea', 'Guinea-Bissau', 'Liberia', 'Mali',
            'Mauritania', 'Niger', 'Nigeria', 'Senegal', 'Sierra Leone', 'Togo'
        ],
        'East Africa': [
            'Burundi', 'Comoros', 'Djibouti', 'Eritrea', 'Ethiopia',
            'Kenya', 'Madagascar', 'Mauritius', 'Rwanda', 'Seychelles',
            'Somalia', 'South Sudan', 'Sudan', 'Tanzania', 'Uganda'
        ],
        'Central Africa': [
            'Angola', 'Cameroon', 'Central African Republic', 'Chad',
            'Congo', 'Democratic Republic of Congo', 'Equatorial Guinea',
            'Gabon', 'Sao Tome and Principe'
        ],
        'Southern Africa': [
            'Botswana', 'Eswatini', 'Lesotho', 'Malawi', 'Mozambique',
            'Namibia', 'South Africa', 'Zambia', 'Zimbabwe'
        ]
    }
    
    if region:
        return countries.get(region, [])
    
    # Return all countries
    all_countries = []
    for region_countries in countries.values():
        all_countries.extend(region_countries)
    
    return sorted(all_countries)
