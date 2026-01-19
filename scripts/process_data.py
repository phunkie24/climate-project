# Databricks notebook source
import sys
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (
    load_climate_data,
    clean_data,
    validate_dataframe,
    add_regional_mapping,
    save_processed_data,
)

from src.features import engineer_features


def main():
    df = load_climate_data("data/raw/climate_data_raw.csv")

    df = clean_data(df)

    validate_dataframe(df)

    df = add_regional_mapping(df)

    print("COLUMNS BEFORE FEATURES:", df.columns.tolist())

    df_features = engineer_features(
        df,
        temporal=True,
        lags=[1, 3],
        rolling_windows=[3],
        changes=True,
        interactions=True,
        regional=True,
    )

    save_processed_data(df_features, "data/processed/climate_data_featured.csv")

    print("\nPIPELINE SUCCESSFUL")
    print(df_features.head())


if __name__ == "__main__":
    main()
