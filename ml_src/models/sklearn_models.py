"""Scikit-learn model implementations."""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

from .base import BaseModel


class SklearnRandomForest(BaseModel):
    """Random Forest model using scikit-learn."""
    
    def build_model(self) -> Any:
        """Build Random Forest model."""
        params = self.config.get('parameters', {})
        
        # Determine if this should be a classifier or regressor
        # This will be determined at fit time based on target variable
        self.logger.info("Random Forest model will be configured at fit time")
        return None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Fit Random Forest model."""
        params = self.config.get('parameters', {})
        
        # Determine if this is classification or regression
        is_classification = self._is_classification_task(y_train)
        
        if is_classification:
            self.model = RandomForestClassifier(**params)
            self.logger.info("Using RandomForestClassifier")
        else:
            self.model = RandomForestRegressor(**params)
            self.logger.info("Using RandomForestRegressor")
        
        # Fit the model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Log training performance
        train_score = self.model.score(X_train, y_train)
        self.logger.info(f"Training score: {train_score:.4f}")
        
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            self.logger.info(f"Validation score: {val_score:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def _is_classification_task(self, y: np.ndarray) -> bool:
        """Determine if this is a classification task."""
        unique_values = len(np.unique(y))
        return unique_values < 20 and (y.dtype == 'object' or unique_values <= 10)


class SklearnLinearRegression(BaseModel):
    """Linear Regression model using scikit-learn."""
    
    def build_model(self) -> LinearRegression:
        """Build Linear Regression model."""
        params = self.config.get('parameters', {})
        self.model = LinearRegression(**params)
        self.logger.info("Linear Regression model created")
        return self.model
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Fit Linear Regression model."""
        if self.model is None:
            self.build_model()
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Log training performance
        train_score = self.model.score(X_train, y_train)
        self.logger.info(f"Training R² score: {train_score:.4f}")
        
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            self.logger.info(f"Validation R² score: {val_score:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def is_classifier(self) -> bool:
        """Linear regression is not a classifier."""
        return False


class SklearnLogisticRegression(BaseModel):
    """Logistic Regression model using scikit-learn."""
    
    def build_model(self) -> LogisticRegression:
        """Build Logistic Regression model."""
        params = self.config.get('parameters', {})
        
        # Set default parameters for better convergence
        default_params = {
            'max_iter': 1000,
            'random_state': 42
        }
        default_params.update(params)
        
        self.model = LogisticRegression(**default_params)
        self.logger.info("Logistic Regression model created")
        return self.model
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Fit Logistic Regression model."""
        if self.model is None:
            self.build_model()
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Log training performance
        train_score = self.model.score(X_train, y_train)
        self.logger.info(f"Training accuracy: {train_score:.4f}")
        
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            self.logger.info(f"Validation accuracy: {val_score:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def is_classifier(self) -> bool:
        """Logistic regression is a classifier."""
        return True