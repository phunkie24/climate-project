# Databricks notebook source
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    print("LOADED COLUMNS:", df.columns.tolist())

    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    print("NORMALIZED COLUMNS:", df.columns.tolist())

    # 🔥 HARD GUARANTEE
    if "country" not in df.columns:
        raise KeyError(
            "CRITICAL ERROR: 'country' column not found after normalization.\n"
            f"Available columns: {df.columns.tolist()}\n"
            "Fix your CSV header."
        )

    return df


def add_regional_mapping(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    print("COLUMNS BEFORE REGIONAL:", df.columns.tolist())

    if "country" not in df.columns:
        raise KeyError(
            "'country' missing before regional mapping.\n"
            f"Columns: {df.columns.tolist()}"
        )

    df["country"] = df["country"].astype(str).str.strip().str.lower()

    region_map = {
        "nigeria": "West Africa",
        "ghana": "West Africa",
        "kenya": "East Africa",
        "ethiopia": "East Africa",
        "south africa": "Southern Africa",
    }

    df["region"] = df["country"].map(region_map).fillna("Unknown")

    return df
