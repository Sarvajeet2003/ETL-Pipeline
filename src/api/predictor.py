"""
Model prediction service
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import joblib
import json
from datetime import datetime

from src.utils.config import config_manager
from src.utils.logging import get_logger
from src.utils.exceptions import ModelLoadException, MLException
from src.ml.preprocessor import MLPreprocessor

logger = get_logger()

class ModelPredictor:
    """Handle model loading and predictions"""
    
    def __init__(self):
        self.config = config_manager.get_api_config()['api']
        self.ml_config = config_manager.get_ml_config()['ml']
        
        self.model = None
        self.preprocessor = None
        self.model_metadata = None
        self.model_name = None
        self.last_loaded = None
        
        # Load model on initialization
        self.load_model()
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load the trained model and preprocessors
        
        Args:
            model_path: Optional path to model directory
        """
        try:
            if model_path is None:
                model_path = self.config['models']['model_path']
            
            model_dir = Path(model_path)
            
            if not model_dir.exists():
                raise ModelLoadException(f"Model directory not found: {model_dir}")
            
            # Load model metadata
            metadata_path = model_dir / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                self.model_name = self.model_metadata.get('best_model_name', 'unknown')
            else:
                logger.warning("Model metadata not found, using default model")
                self.model_name = 'unknown'
            
            # Load the best model
            model_file = self.config['models']['default_model']
            model_file_path = model_dir / model_file
            
            if not model_file_path.exists():
                raise ModelLoadException(f"Model file not found: {model_file_path}")
            
            self.model = joblib.load(model_file_path)
            logger.info(f"Loaded model from {model_file_path}")
            
            # Load preprocessors
            self.preprocessor = MLPreprocessor(self.ml_config['data'])
            self.preprocessor.load_preprocessors(str(model_dir))
            
            self.last_loaded = datetime.now()
            logger.info(f"Model and preprocessors loaded successfully at {self.last_loaded}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise ModelLoadException(f"Failed to load model: {str(e)}")
    
    def preprocess_input(self, features: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess input features for prediction
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([features])
            
            # Apply the same preprocessing as during training
            df_processed = self.preprocessor.encode_categorical_features(df, fit=False)
            df_processed = self.preprocessor.scale_numerical_features(df_processed, fit=False)
            
            # Ensure all expected features are present (use final feature names after encoding)
            expected_features = getattr(self.preprocessor, 'final_feature_names', None)
            if expected_features:
                # Add missing columns with default values
                for col in expected_features:
                    if col not in df_processed.columns:
                        df_processed[col] = 0
                
                # Reorder columns to match training data
                df_processed = df_processed.reindex(columns=expected_features, fill_value=0)
            else:
                # Fallback: try to get feature names from the model if available
                if hasattr(self.model, 'feature_names_in_'):
                    model_features = list(self.model.feature_names_in_)
                    df_processed = df_processed.reindex(columns=model_features, fill_value=0)
            
            logger.debug(f"Preprocessed input shape: {df_processed.shape}")
            return df_processed
            
        except Exception as e:
            logger.error(f"Error preprocessing input: {str(e)}")
            raise MLException(f"Failed to preprocess input: {str(e)}")
    
    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a single prediction
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if self.model is None:
                raise MLException("Model not loaded")
            
            # Preprocess input
            X = self.preprocess_input(features)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            
            # Get prediction probability if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)[0]
                probability = float(probabilities[1])  # Probability of positive class
            else:
                probability = float(prediction)  # For models without probability
            
            # Determine confidence level
            confidence = self._get_confidence_level(probability)
            
            result = {
                'prediction': int(prediction),
                'probability': probability,
                'confidence': confidence,
                'model_used': self.model_name,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Prediction made: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise MLException(f"Failed to make prediction: {str(e)}")
    
    def predict_batch(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make batch predictions
        
        Args:
            instances: List of feature dictionaries
            
        Returns:
            List of prediction results
        """
        try:
            if self.model is None:
                raise MLException("Model not loaded")
            
            results = []
            
            for i, features in enumerate(instances):
                try:
                    result = self.predict_single(features)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error predicting instance {i}: {str(e)}")
                    # Add error result for this instance
                    results.append({
                        'prediction': -1,
                        'probability': 0.0,
                        'confidence': 'Error',
                        'model_used': self.model_name,
                        'timestamp': datetime.now().isoformat(),
                        'error': str(e)
                    })
            
            logger.info(f"Batch prediction completed for {len(instances)} instances")
            return results
            
        except Exception as e:
            logger.error(f"Error making batch predictions: {str(e)}")
            raise MLException(f"Failed to make batch predictions: {str(e)}")
    
    def _get_confidence_level(self, probability: float) -> str:
        """
        Determine confidence level based on probability
        
        Args:
            probability: Prediction probability
            
        Returns:
            Confidence level string
        """
        if probability >= 0.8 or probability <= 0.2:
            return "High"
        elif probability >= 0.6 or probability <= 0.4:
            return "Medium"
        else:
            return "Low"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        try:
            info = {
                'model_name': self.model_name,
                'model_type': type(self.model).__name__ if self.model else 'Not loaded',
                'last_loaded': self.last_loaded.isoformat() if self.last_loaded else None,
                'model_loaded': self.model is not None
            }
            
            # Add performance metrics if available
            if self.model_metadata:
                info['metadata'] = self.model_metadata
            
            # Add feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                feature_names = self.preprocessor.feature_columns if self.preprocessor else None
                if feature_names and len(feature_names) == len(self.model.feature_importances_):
                    importance_dict = dict(zip(feature_names, self.model.feature_importances_))
                    # Sort by importance
                    sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
                    info['feature_importance'] = {k: float(v) for k, v in list(sorted_importance.items())[:10]}  # Top 10
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {
                'model_name': 'unknown',
                'model_type': 'unknown',
                'last_loaded': None,
                'model_loaded': False,
                'error': str(e)
            }
    
    def reload_model(self) -> bool:
        """
        Reload the model (useful for auto-reload functionality)
        
        Returns:
            True if reload was successful
        """
        try:
            logger.info("Reloading model...")
            self.load_model()
            return True
        except Exception as e:
            logger.error(f"Error reloading model: {str(e)}")
            return False
    
    def is_model_loaded(self) -> bool:
        """
        Check if model is loaded and ready for predictions
        
        Returns:
            True if model is loaded
        """
        return self.model is not None and self.preprocessor is not None

# Global predictor instance
predictor = ModelPredictor()