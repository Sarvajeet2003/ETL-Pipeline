"""Feature engineering and preprocessing module."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, LabelEncoder, OrdinalEncoder
)
from sklearn.feature_selection import (
    mutual_info_regression, mutual_info_classif,
    f_regression, f_classif, RFE, SelectKBest
)
from sklearn.linear_model import LassoCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from loguru import logger

from .config import FeatureEngineeringConfig


class FeatureEngineer:
    """Handles feature engineering and preprocessing."""
    
    def __init__(self, config: FeatureEngineeringConfig):
        """Initialize feature engineer with configuration.
        
        Args:
            config: Feature engineering configuration
        """
        self.config = config
        self.logger = logger.bind(component="FeatureEngineer")
        self.preprocessor = None
        self.feature_names = None
        
    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series, 
                     X_val: Optional[pd.DataFrame] = None, 
                     X_test: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Fit feature engineering pipeline and transform datasets.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            X_test: Test features (optional)
            
        Returns:
            Tuple of transformed (X_train, X_val, X_test)
        """
        if not self.config.enabled:
            self.logger.info("Feature engineering disabled, returning original data")
            return X_train.values, X_val.values if X_val is not None else None, X_test.values if X_test is not None else None
        
        self.logger.info("Starting feature engineering pipeline")
        
        # Create custom features first
        X_train_custom = self._create_custom_features(X_train)
        X_val_custom = self._create_custom_features(X_val) if X_val is not None else None
        X_test_custom = self._create_custom_features(X_test) if X_test is not None else None
        
        # Build preprocessing pipeline
        self.preprocessor = self._build_preprocessor(X_train_custom)
        
        # Fit and transform training data
        X_train_transformed = self.preprocessor.fit_transform(X_train_custom)
        
        # Transform validation and test data
        X_val_transformed = None
        X_test_transformed = None
        
        if X_val_custom is not None:
            X_val_transformed = self.preprocessor.transform(X_val_custom)
        
        if X_test_custom is not None:
            X_test_transformed = self.preprocessor.transform(X_test_custom)
        
        # Store feature names for later use
        self._extract_feature_names()
        
        # Apply feature selection if enabled
        if self.config.feature_selection.get('enabled', False):
            X_train_transformed, X_val_transformed, X_test_transformed = self._apply_feature_selection(
                X_train_transformed, y_train, X_val_transformed, X_test_transformed
            )
        
        self.logger.info(f"Feature engineering completed. Final feature count: {X_train_transformed.shape[1]}")
        
        return X_train_transformed, X_val_transformed, X_test_transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        # Apply custom features
        X_custom = self._create_custom_features(X)
        
        # Transform using fitted preprocessor
        X_transformed = self.preprocessor.transform(X_custom)
        
        return X_transformed
    
    def _create_custom_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create custom features based on configuration.
        
        Args:
            X: Input features
            
        Returns:
            DataFrame with custom features added
        """
        if not self.config.custom_features:
            return X
        
        X_custom = X.copy()
        
        for feature_config in self.config.custom_features:
            try:
                feature_name = feature_config['name']
                formula = feature_config['formula']
                
                self.logger.info(f"Creating custom feature: {feature_name}")
                
                # Evaluate formula in the context of the DataFrame
                # Note: This is a simplified implementation. In production, 
                # consider using a safer expression evaluator
                X_custom[feature_name] = eval(formula, {"__builtins__": {}}, {
                    **{col: X_custom[col] for col in X_custom.columns},
                    'np': np, 'pd': pd
                })
                
                self.logger.info(f"Custom feature '{feature_name}' created successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to create custom feature '{feature_name}': {e}")
                continue
        
        return X_custom
    
    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Build preprocessing pipeline based on configuration.
        
        Args:
            X: Training features
            
        Returns:
            Fitted ColumnTransformer
        """
        # Identify column types
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self.logger.info(f"Identified {len(numeric_columns)} numeric and {len(categorical_columns)} categorical columns")
        
        transformers = []
        
        # Numerical preprocessing
        if numeric_columns:
            numeric_transformer = self._build_numeric_transformer()
            transformers.append(('num', numeric_transformer, numeric_columns))
        
        # Categorical preprocessing
        if categorical_columns:
            categorical_transformer = self._build_categorical_transformer()
            transformers.append(('cat', categorical_transformer, categorical_columns))
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough',  # Keep other columns as-is
            sparse_threshold=0  # Return dense array
        )
        
        return preprocessor
    
    def _build_numeric_transformer(self) -> Pipeline:
        """Build numeric preprocessing pipeline.
        
        Returns:
            Numeric preprocessing pipeline
        """
        scaling_config = self.config.numerical_scaling
        scaling_method = scaling_config.get('method', 'standard')
        
        steps = []
        
        # Add scaler based on configuration
        if scaling_method == 'standard':
            steps.append(('scaler', StandardScaler()))
        elif scaling_method == 'minmax':
            steps.append(('scaler', MinMaxScaler()))
        elif scaling_method == 'robust':
            steps.append(('scaler', RobustScaler()))
        elif scaling_method == 'none':
            pass  # No scaling
        else:
            self.logger.warning(f"Unknown scaling method: {scaling_method}, using standard")
            steps.append(('scaler', StandardScaler()))
        
        if not steps:
            steps.append(('passthrough', 'passthrough'))
        
        return Pipeline(steps)
    
    def _build_categorical_transformer(self) -> Pipeline:
        """Build categorical preprocessing pipeline.
        
        Returns:
            Categorical preprocessing pipeline
        """
        encoding_config = self.config.categorical_encoding
        encoding_method = encoding_config.get('method', 'one_hot')
        
        steps = []
        
        # Add encoder based on configuration
        if encoding_method == 'one_hot':
            encoder = OneHotEncoder(
                handle_unknown=encoding_config.get('handle_unknown', 'ignore'),
                drop=encoding_config.get('drop_first', 'first') if encoding_config.get('drop_first', True) else None,
                sparse_output=False
            )
            steps.append(('encoder', encoder))
        elif encoding_method == 'label':
            steps.append(('encoder', LabelEncoder()))
        elif encoding_method == 'ordinal':
            steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
        else:
            self.logger.warning(f"Unknown encoding method: {encoding_method}, using one_hot")
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            steps.append(('encoder', encoder))
        
        return Pipeline(steps)
    
    def _extract_feature_names(self) -> None:
        """Extract feature names from the fitted preprocessor."""
        try:
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                self.feature_names = self.preprocessor.get_feature_names_out().tolist()
            else:
                # Fallback for older sklearn versions
                self.feature_names = [f'feature_{i}' for i in range(self.preprocessor.transform(pd.DataFrame()).shape[1])]
            
            self.logger.info(f"Extracted {len(self.feature_names)} feature names")
            
        except Exception as e:
            self.logger.warning(f"Could not extract feature names: {e}")
            self.feature_names = None
    
    def _apply_feature_selection(self, X_train: np.ndarray, y_train: pd.Series,
                               X_val: Optional[np.ndarray], X_test: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Apply feature selection based on configuration.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            X_test: Test features
            
        Returns:
            Tuple of feature-selected datasets
        """
        selection_config = self.config.feature_selection
        method = selection_config.get('method', 'mutual_info')
        k_features = selection_config.get('k_features', 10)
        
        self.logger.info(f"Applying feature selection: {method}, k={k_features}")
        
        # Determine if this is a classification or regression problem
        is_classification = y_train.nunique() < 20 and y_train.dtype in ['object', 'category', 'bool']
        
        try:
            if method == 'mutual_info':
                if is_classification:
                    score_func = mutual_info_classif
                else:
                    score_func = mutual_info_regression
                selector = SelectKBest(score_func=score_func, k=k_features)
                
            elif method == 'f_test':
                if is_classification:
                    score_func = f_classif
                else:
                    score_func = f_regression
                selector = SelectKBest(score_func=score_func, k=k_features)
                
            elif method == 'rfe':
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                if is_classification:
                    estimator = RandomForestClassifier(n_estimators=50, random_state=42)
                else:
                    estimator = RandomForestRegressor(n_estimators=50, random_state=42)
                selector = RFE(estimator=estimator, n_features_to_select=k_features)
                
            elif method == 'lasso':
                if is_classification:
                    self.logger.warning("Lasso feature selection not suitable for classification, using mutual_info")
                    selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
                else:
                    # Use Lasso for feature selection
                    lasso = LassoCV(cv=5, random_state=42)
                    lasso.fit(X_train, y_train)
                    
                    # Select features with non-zero coefficients
                    selected_features = np.abs(lasso.coef_) > 1e-5
                    
                    if selected_features.sum() == 0:
                        self.logger.warning("No features selected by Lasso, keeping all features")
                        return X_train, X_val, X_test
                    
                    # Apply selection
                    X_train_selected = X_train[:, selected_features]
                    X_val_selected = X_val[:, selected_features] if X_val is not None else None
                    X_test_selected = X_test[:, selected_features] if X_test is not None else None
                    
                    self.logger.info(f"Lasso selected {selected_features.sum()} features")
                    return X_train_selected, X_val_selected, X_test_selected
            else:
                self.logger.warning(f"Unknown feature selection method: {method}")
                return X_train, X_val, X_test
            
            # Fit selector and transform data
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_val_selected = selector.transform(X_val) if X_val is not None else None
            X_test_selected = selector.transform(X_test) if X_test is not None else None
            
            # Update feature names if available
            if self.feature_names is not None and hasattr(selector, 'get_support'):
                selected_mask = selector.get_support()
                self.feature_names = [name for name, selected in zip(self.feature_names, selected_mask) if selected]
            
            self.logger.info(f"Feature selection completed: {X_train_selected.shape[1]} features selected")
            
            return X_train_selected, X_val_selected, X_test_selected
            
        except Exception as e:
            self.logger.error(f"Feature selection failed: {e}")
            return X_train, X_val, X_test
    
    def get_feature_names(self) -> Optional[List[str]]:
        """Get feature names after preprocessing.
        
        Returns:
            List of feature names or None if not available
        """
        return self.feature_names
    
    def get_feature_importance_from_preprocessor(self) -> Optional[Dict[str, float]]:
        """Get feature importance from preprocessing steps if available.
        
        Returns:
            Dictionary of feature importances or None
        """
        if self.feature_names is None:
            return None
        
        # This is a placeholder - actual implementation would depend on
        # the specific preprocessing steps used
        return None