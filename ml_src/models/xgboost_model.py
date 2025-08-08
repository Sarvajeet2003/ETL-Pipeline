"""XGBoost model implementation."""

from typing import Dict, Any, Optional
import numpy as np
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from .base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost model implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize XGBoost model."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install it with: pip install xgboost")
        
        super().__init__(config)
        self.training_history = {}
    
    def build_model(self) -> Any:
        """Build XGBoost model."""
        params = self.config.get('parameters', {})
        
        # Set default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(params)
        
        # The actual model type will be determined at fit time
        self.model_params = default_params
        self.logger.info("XGBoost model parameters configured")
        return None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Fit XGBoost model."""
        if self.model is None:
            self.build_model()
        
        # Determine if this is classification or regression
        is_classification = self._is_classification_task(y_train)
        
        if is_classification:
            self.model = xgb.XGBClassifier(**self.model_params)
            self.logger.info("Using XGBClassifier")
        else:
            self.model = xgb.XGBRegressor(**self.model_params)
            self.logger.info("Using XGBRegressor")
        
        # Prepare evaluation set for early stopping
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        else:
            eval_set = [(X_train, y_train)]
        
        # Configure early stopping
        early_stopping_config = self.config.get('early_stopping', {})
        fit_params = {}
        
        if early_stopping_config.get('enabled', False) and X_val is not None:
            fit_params.update({
                'eval_set': eval_set,
                'early_stopping_rounds': early_stopping_config.get('patience', 10),
                'verbose': False
            })
        
        # Fit the model
        self.model.fit(X_train, y_train, **fit_params)
        self.is_fitted = True
        
        # Store training history
        if hasattr(self.model, 'evals_result_'):
            self.training_history = self.model.evals_result_
        
        # Log training performance
        train_predictions = self.model.predict(X_train)
        if is_classification:
            from sklearn.metrics import accuracy_score
            train_score = accuracy_score(y_train, train_predictions)
            self.logger.info(f"Training accuracy: {train_score:.4f}")
        else:
            from sklearn.metrics import r2_score
            train_score = r2_score(y_train, train_predictions)
            self.logger.info(f"Training R² score: {train_score:.4f}")
        
        if X_val is not None and y_val is not None:
            val_predictions = self.model.predict(X_val)
            if is_classification:
                val_score = accuracy_score(y_val, val_predictions)
                self.logger.info(f"Validation accuracy: {val_score:.4f}")
            else:
                val_score = r2_score(y_val, val_predictions)
                self.logger.info(f"Validation R² score: {val_score:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def get_training_history(self) -> Optional[Dict[str, Any]]:
        """Get training history."""
        return self.training_history if self.training_history else None
    
    def _is_classification_task(self, y: np.ndarray) -> bool:
        """Determine if this is a classification task."""
        unique_values = len(np.unique(y))
        return unique_values < 20 and (y.dtype == 'object' or unique_values <= 10)
    
    def is_classifier(self) -> bool:
        """Check if model is a classifier."""
        return isinstance(self.model, xgb.XGBClassifier) if self.model else False