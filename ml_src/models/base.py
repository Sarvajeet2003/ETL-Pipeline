"""Base model class for all ML models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger


class BaseModel(ABC):
    """Abstract base class for all ML models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model with configuration.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None
        self.is_fitted = False
        self.logger = logger.bind(model=self.__class__.__name__)
        
    @abstractmethod
    def build_model(self) -> Any:
        """Build the model with specified parameters.
        
        Returns:
            Model instance
        """
        pass
    
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Fit the model to training data.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Get prediction probabilities (for classification models).
        
        Args:
            X: Features to predict on
            
        Returns:
            Prediction probabilities or None if not supported
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores.
        
        Returns:
            Feature importance array or None if not supported
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_).flatten()
        return None
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return self.config.get('parameters', {})
    
    def get_training_history(self) -> Optional[Dict[str, Any]]:
        """Get training history if available.
        
        Returns:
            Training history dictionary or None
        """
        return None
    
    @property
    def model_type(self) -> str:
        """Get model type identifier.
        
        Returns:
            Model type string
        """
        return self.__class__.__name__
    
    def is_classifier(self) -> bool:
        """Check if model is a classifier.
        
        Returns:
            True if classifier, False if regressor
        """
        # Default implementation - override in subclasses if needed
        return hasattr(self.model, 'predict_proba')
    
    def save_model(self, filepath: str) -> None:
        """Save model to file.
        
        Args:
            filepath: Path to save the model
        """
        import joblib
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        try:
            joblib.dump(self.model, filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load model from file.
        
        Args:
            filepath: Path to load the model from
        """
        import joblib
        
        try:
            self.model = joblib.load(filepath)
            self.is_fitted = True
            self.logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise