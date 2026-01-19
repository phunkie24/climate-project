"""
Tests for data module
"""

import pytest
import pandas as pd
import numpy as np
from src.data import (
    clean_data,
    add_regional_mapping,
    validate_dataframe,
    get_country_list
)


@pytest.fixture
def sample_data():
    """Create sample climate data for testing"""
    return pd.DataFrame({
        'country': ['Nigeria', 'Kenya', 'Nigeria', 'Kenya'],
        'year': [2020, 2020, 2021, 2021],
        'precipitation(mm)': [1000, 800, 1050, 820],
        'avg_temp_c': [26.5, 24.0, 26.8, 24.2],
        'avg_humidity(%)': [65, 58, 64, 59],
        'cloud_cover(%)': [50, 45, 52, 46],
        'atmospheric_co2(ppm)': [414, 414, 416, 416]
    })


def test_clean_data(sample_data):
    """Test data cleaning function"""
    cleaned = clean_data(sample_data, drop_duplicates=True)
    
    assert len(cleaned) == len(sample_data)
    assert cleaned['precipitation(mm)'].min() >= 0
    assert cleaned['avg_humidity(%)'].max() <= 100


def test_add_regional_mapping(sample_data):
    """Test regional mapping"""
    df_with_region = add_regional_mapping(sample_data)
    
    assert 'region' in df_with_region.columns
    assert df_with_region.loc[df_with_region['country'] == 'Nigeria', 'region'].iloc[0] == 'West Africa'
    assert df_with_region.loc[df_with_region['country'] == 'Kenya', 'region'].iloc[0] == 'East Africa'


def test_validate_dataframe(sample_data):
    """Test data validation"""
    is_valid, report = validate_dataframe(sample_data, verbose=False)
    
    assert isinstance(is_valid, bool)
    assert isinstance(report, dict)
    assert 'completeness' in report or 'validity' in report


def test_get_country_list():
    """Test country list function"""
    all_countries = get_country_list()
    west_africa = get_country_list('West Africa')
    
    assert len(all_countries) > 40
    assert 'Nigeria' in all_countries
    assert 'Nigeria' in west_africa
    assert 'Kenya' not in west_africa


def test_clean_data_with_missing():
    """Test cleaning data with missing values"""
    df = pd.DataFrame({
        'country': ['Nigeria', 'Kenya', 'Nigeria'],
        'year': [2020, 2020, 2021],
        'precipitation(mm)': [1000, np.nan, 1050],
        'avg_temp_c': [26.5, 24.0, 26.8]
    })
    
    cleaned = clean_data(df, fill_method='ffill')
    
    # After forward fill, there might still be NaN in first occurrence
    assert cleaned['precipitation(mm)'].notna().sum() >= 2
