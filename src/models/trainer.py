"""
Model training utilities
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib


class ModelTrainer:
    """Train and evaluate machine learning models"""
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'precipitation(mm)'):
        self.df = df
        self.target_col = target_col
        self.models = {}
        self.results = {}
        self.scalers = {}
    
    def prepare_data(self, 
                     feature_cols: Optional[List[str]] = None,
                     test_year: int = 2018) -> Tuple:
        """
        Prepare train/test splits using temporal split
        
        Args:
            feature_cols: List of feature column names
            test_year: Year to split train/test (before = train, >= test)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, features)
        """
        # Remove rows with missing target
        df_clean = self.df.dropna(subset=[self.target_col])
        
        # Define features if not provided
        if feature_cols is None:
            # Auto-select numeric columns except target
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = [self.target_col, 'year']
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Remove features with missing values
        available_features = [f for f in feature_cols if f in df_clean.columns]
        df_clean = df_clean.dropna(subset=available_features)
        
        # Temporal split
        train = df_clean[df_clean['year'] < test_year]
        test = df_clean[df_clean['year'] >= test_year]
        
        X_train = train[available_features]
        y_train = train[self.target_col]
        X_test = test[available_features]
        y_test = test[self.target_col]
        
        print(f"Training set: {len(X_train)} samples ({train['year'].min()}-{train['year'].max()})")
        print(f"Test set: {len(X_test)} samples ({test['year'].min()}-{test['year'].max()})")
        print(f"Features: {len(available_features)}")
        
        return X_train, X_test, y_train, y_test, available_features
    
    def train_linear_model(self, X_train, y_train, X_test, y_test, alpha: float = 1.0):
        """Train Ridge regression model"""
        print("\nTraining Linear Regression (Ridge)...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Evaluate
        results = self._calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)
        
        self.models['linear'] = model
        self.scalers['linear'] = scaler
        self.results['linear'] = results
        
        self._print_results('Linear Regression', results)
        
        return model, scaler, results
    
    def train_random_forest(self, X_train, y_train, X_test, y_test,
                           n_estimators: int = 200, max_depth: int = 15):
        """Train Random Forest model"""
        print("\nTraining Random Forest...")
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Evaluate
        results = self._calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)
        results['feature_importance'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.models['random_forest'] = model
        self.results['random_forest'] = results
        
        self._print_results('Random Forest', results)
        
        return model, results
    
    def train_xgboost(self, X_train, y_train, X_test, y_test,
                     n_estimators: int = 500, learning_rate: float = 0.01):
        """Train XGBoost model"""
        print("\nTraining XGBoost...")
        
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            eval_metric='rmse',
            early_stopping_rounds=50,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Evaluate
        results = self._calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)
        results['best_iteration'] = model.best_iteration
        results['feature_importance'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.models['xgboost'] = model
        self.results['xgboost'] = results
        
        self._print_results('XGBoost', results)
        
        return model, results
    
    def _calculate_metrics(self, y_train, y_train_pred, y_test, y_test_pred) -> Dict:
        """Calculate evaluation metrics"""
        return {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'test_predictions': y_test_pred
        }
    
    def _print_results(self, model_name: str, results: Dict):
        """Print model results"""
        print(f"\n{'='*50}")
        print(f"{model_name.upper()} RESULTS")
        print(f"{'='*50}")
        print(f"Train RMSE: {results['train_rmse']:.2f} mm")
        print(f"Test RMSE:  {results['test_rmse']:.2f} mm")
        print(f"Train MAE:  {results['train_mae']:.2f} mm")
        print(f"Test MAE:   {results['test_mae']:.2f} mm")
        print(f"Train R²:   {results['train_r2']:.3f}")
        print(f"Test R²:    {results['test_r2']:.3f}")
        
        if 'feature_importance' in results:
            print(f"\nTop 10 Features:")
            print(results['feature_importance'].head(10).to_string(index=False))
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models"""
        comparison_data = []
        
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Train RMSE': results['train_rmse'],
                'Test RMSE': results['test_rmse'],
                'Train MAE': results['train_mae'],
                'Test MAE': results['test_mae'],
                'Train R²': results['train_r2'],
                'Test R²': results['test_r2']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print(f"\n{'='*70}")
        print("MODEL COMPARISON")
        print(f"{'='*70}")
        print(comparison_df.to_string(index=False))
        print(f"{'='*70}\n")
        
        return comparison_df
    
    def save_model(self, model_name: str, output_dir: str = 'models'):
        """Save trained model"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Save model
        model_path = f"{output_dir}/{model_name}_model.pkl"
        joblib.dump(self.models[model_name], model_path)
        print(f"[OK] Saved model to {model_path}")
        
        # Save scaler if exists
        if model_name in self.scalers:
            scaler_path = f"{output_dir}/{model_name}_scaler.pkl"
            joblib.dump(self.scalers[model_name], scaler_path)
            print(f"[OK] Saved scaler to {scaler_path}")


def train_all_models(df: pd.DataFrame, 
                     target_col: str = 'precipitation(mm)',
                     test_year: int = 2018) -> Tuple[ModelTrainer, pd.DataFrame]:
    """
    Convenience function to train all models
    
    Args:
        df: DataFrame with features
        target_col: Target variable name
        test_year: Year for train/test split
        
    Returns:
        Tuple of (trainer, comparison_df)
    """
    trainer = ModelTrainer(df, target_col)
    
    # Prepare data
    X_train, X_test, y_train, y_test, features = trainer.prepare_data(test_year=test_year)
    
    # Train models
    trainer.train_linear_model(X_train, y_train, X_test, y_test)
    trainer.train_random_forest(X_train, y_train, X_test, y_test)
    trainer.train_xgboost(X_train, y_train, X_test, y_test)
    
    # Compare
    comparison = trainer.compare_models()
    
    return trainer, comparison
