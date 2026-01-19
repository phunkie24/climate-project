"""
Model prediction utilities
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, Union, Optional
from pathlib import Path


class RainfallPredictor:
    """Make predictions using trained models"""
    
    def __init__(self, model, scaler=None, features=None):
        self.model = model
        self.scaler = scaler
        self.features = features
    
    @classmethod
    def load(cls, model_path: Union[str, Path], 
             scaler_path: Optional[Union[str, Path]] = None,
             features_path: Optional[Union[str, Path]] = None):
        """
        Load trained model from disk
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler (optional)
            features_path: Path to saved feature list (optional)
            
        Returns:
            RainfallPredictor instance
        """
        model = joblib.load(model_path)
        
        scaler = None
        if scaler_path and Path(scaler_path).exists():
            scaler = joblib.load(scaler_path)
        
        features = None
        if features_path and Path(features_path).exists():
            features = joblib.load(features_path)
        
        return cls(model, scaler, features)
    
    def predict(self, X: Union[pd.DataFrame, Dict]) -> Union[float, np.ndarray]:
        """
        Make rainfall prediction
        
        Args:
            X: Features as DataFrame or dict
            
        Returns:
            Predicted rainfall in mm
        """
        # Convert dict to DataFrame if needed
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        
        # Ensure feature order matches training
        if self.features is not None:
            # Check for missing features
            missing_features = set(self.features) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            X = X[self.features]
        
        # Apply scaling if needed
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
        else:
            predictions = self.model.predict(X)
        
        # Return single value if single prediction
        if len(predictions) == 1:
            return float(predictions[0])
        
        return predictions
    
    def predict_for_country(self,
                           country: str,
                           year: int,
                           temp: float,
                           humidity: float,
                           co2: float,
                           cloud_cover: float = 50.0,
                           **kwargs) -> float:
        """
        Predict rainfall for specific country and conditions
        
        Args:
            country: Country name
            year: Year
            temp: Average temperature (°C)
            humidity: Average humidity (%)
            co2: Atmospheric CO2 (ppm)
            cloud_cover: Cloud cover (%)
            **kwargs: Additional features
            
        Returns:
            Predicted rainfall in mm
        """
        features = {
            'avg_temp_c': temp,
            'avg_humidity(%)': humidity,
            'atmospheric_co2(ppm)': co2,
            'cloud_cover(%)': cloud_cover,
            'years_since_1991': year - 1991,
            **kwargs
        }
        
        return self.predict(features)
    
    def predict_with_uncertainty(self, X: Union[pd.DataFrame, Dict],
                                n_samples: int = 100) -> Dict:
        """
        Predict with uncertainty estimation (for ensemble models)
        
        Args:
            X: Features
            n_samples: Number of bootstrap samples
            
        Returns:
            Dict with mean, std, and confidence intervals
        """
        # Convert dict to DataFrame if needed
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        
        # Get predictions from all trees (if Random Forest)
        try:
            if hasattr(self.model, 'estimators_'):
                predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
                
                return {
                    'mean': float(predictions.mean()),
                    'std': float(predictions.std()),
                    'confidence_95_lower': float(np.percentile(predictions, 2.5)),
                    'confidence_95_upper': float(np.percentile(predictions, 97.5))
                }
        except:
            pass
        
        # Fallback to point prediction
        prediction = self.predict(X)
        return {
            'mean': float(prediction),
            'std': None,
            'confidence_95_lower': None,
            'confidence_95_upper': None
        }


def load_predictor(model_name: str = 'xgboost', 
                  models_dir: str = 'models') -> RainfallPredictor:
    """
    Convenience function to load a predictor
    
    Args:
        model_name: Name of model ('xgboost', 'random_forest', 'linear')
        models_dir: Directory containing saved models
        
    Returns:
        RainfallPredictor instance
    """
    model_path = f"{models_dir}/{model_name}_model.pkl"
    scaler_path = f"{models_dir}/{model_name}_scaler.pkl"
    features_path = f"{models_dir}/{model_name}_features.pkl"
    
    return RainfallPredictor.load(model_path, scaler_path, features_path)
