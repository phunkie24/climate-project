#!/usr/bin/env python3
"""
Model training script
Train and evaluate machine learning models
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_climate_data
from src.models import train_all_models


def main():
    """Main model training pipeline"""
    
    print("="*70)
    print("MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Load featured data
    print("\n[Step 1/3] Loading processed data...")
    try:
        df = load_climate_data('data/processed/climate_data_featured.csv')
        print(f"[OK] Loaded {len(df)} records with {len(df.columns)} features")
    except FileNotFoundError:
        print("[ERROR] Error: Processed data not found!")
        print("  Please run: python scripts/process_data.py first")
        return
    
    # Train models
    print("\n[Step 2/3] Training models...")
    trainer, comparison = train_all_models(
        df,
        target_col='precipitation(mm)',
        test_year=2018
    )
    
    # Save best model (XGBoost)
    print("\n[Step 3/3] Saving models...")
    trainer.save_model('xgboost', output_dir='models')
    trainer.save_model('random_forest', output_dir='models')
    trainer.save_model('linear', output_dir='models')
    
    # Save comparison results
    comparison.to_csv('results/model_comparison.csv', index=False)
    print(f"[OK] Saved comparison to results/model_comparison.csv")
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best model: XGBoost")
    print(f"Test RMSE: {trainer.results['xgboost']['test_rmse']:.2f} mm")
    print(f"Test R²: {trainer.results['xgboost']['test_r2']:.3f}")
    print(f"\nModels saved to: models/")
    print("="*70)


if __name__ == '__main__':
    main()
