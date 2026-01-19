"""
Data processing module
"""

from .loaders import (
    load_climate_data,
    load_multiple_files,
    save_processed_data,
    get_country_list
)

from .validators import (
    DataValidator,
    validate_dataframe
)

from .transformers import (
    clean_data,
    add_regional_mapping,
    aggregate_to_annual,
    normalize_features
)

__all__ = [
    # Loaders
    'load_climate_data',
    'load_multiple_files',
    'save_processed_data',
    'get_country_list',
    
    # Validators
    'DataValidator',
    'validate_dataframe',
    
    # Transformers
    'clean_data',
    'add_regional_mapping',
    'aggregate_to_annual',
    'normalize_features',
]
