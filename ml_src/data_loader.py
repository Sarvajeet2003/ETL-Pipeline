"""Data loading and preprocessing module."""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
from sklearn.model_selection import train_test_split
from loguru import logger

from .config import DataConfig


class DataLoader:
    """Handles data loading and initial preprocessing."""
    
    def __init__(self, config: DataConfig):
        """Initialize data loader with configuration.
        
        Args:
            config: Data configuration object
        """
        self.config = config
        self.logger = logger.bind(component="DataLoader")
        
    def load_data(self) -> pd.DataFrame:
        """Load data from the specified source.
        
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data format is not supported
        """
        source_path = Path(self.config.source_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Data file not found: {source_path}")
        
        self.logger.info(f"Loading data from: {source_path}")
        
        try:
            # Determine file format and load accordingly
            if source_path.suffix.lower() == '.csv':
                df = pd.read_csv(source_path)
            elif source_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(source_path)
            elif source_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(source_path)
            elif source_path.suffix.lower() == '.json':
                df = pd.read_json(source_path)
            else:
                raise ValueError(f"Unsupported file format: {source_path.suffix}")
            
            self.logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Log basic data info
            self._log_data_info(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise
    
    def prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features_df, target_series)
            
        Raises:
            ValueError: If target column is not found or no features remain
        """
        # Check if target column exists
        if self.config.target_column not in df.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found in data")
        
        # Extract target variable
        y = df[self.config.target_column].copy()
        
        # Determine feature columns
        if self.config.feature_columns == "auto":
            # Use all columns except target and excluded columns
            feature_columns = [
                col for col in df.columns 
                if col != self.config.target_column and col not in self.config.exclude_columns
            ]
        else:
            # Use specified feature columns
            feature_columns = [
                col for col in self.config.feature_columns 
                if col in df.columns and col != self.config.target_column
            ]
        
        if not feature_columns:
            raise ValueError("No feature columns available after filtering")
        
        # Extract features
        X = df[feature_columns].copy()
        
        self.logger.info(f"Features prepared: {len(feature_columns)} columns")
        self.logger.info(f"Target variable: {self.config.target_column}")
        self.logger.debug(f"Feature columns: {feature_columns}")
        
        # Log target variable info
        self._log_target_info(y)
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split data into train, validation, and test sets.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        split_config = self.config.split
        
        test_size = split_config.get('test_size', 0.2)
        val_size = split_config.get('validation_size', 0.2)
        random_state = split_config.get('random_state', 42)
        stratify_col = split_config.get('stratify', False)
        
        # Determine stratification
        stratify_y = None
        if stratify_col:
            if stratify_col == True:
                stratify_y = y
            elif isinstance(stratify_col, str) and stratify_col in X.columns:
                stratify_y = X[stratify_col]
        
        self.logger.info(f"Splitting data: test_size={test_size}, val_size={val_size}")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_y
        )
        
        # Second split: separate train and validation from remaining data
        if val_size > 0:
            # Adjust validation size relative to remaining data
            val_size_adjusted = val_size / (1 - test_size)
            
            # Update stratification for remaining data
            stratify_temp = None
            if stratify_y is not None:
                if stratify_col == True:
                    stratify_temp = y_temp
                elif isinstance(stratify_col, str):
                    stratify_temp = X_temp[stratify_col]
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=stratify_temp
            )
        else:
            # No validation set
            X_train, X_val = X_temp, pd.DataFrame()
            y_train, y_val = y_temp, pd.Series(dtype=y.dtype)
        
        # Log split results
        self.logger.info(f"Data split completed:")
        self.logger.info(f"  Training set: {X_train.shape[0]} samples")
        if not X_val.empty:
            self.logger.info(f"  Validation set: {X_val.shape[0]} samples")
        self.logger.info(f"  Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _log_data_info(self, df: pd.DataFrame) -> None:
        """Log basic information about the dataset.
        
        Args:
            df: DataFrame to analyze
        """
        self.logger.info("Dataset overview:")
        self.logger.info(f"  Shape: {df.shape}")
        self.logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data types
        dtype_counts = df.dtypes.value_counts()
        self.logger.info(f"  Data types: {dict(dtype_counts)}")
        
        # Missing values
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        if len(missing_cols) > 0:
            self.logger.warning(f"  Missing values found in {len(missing_cols)} columns:")
            for col, count in missing_cols.items():
                pct = (count / len(df)) * 100
                self.logger.warning(f"    {col}: {count} ({pct:.1f}%)")
        else:
            self.logger.info("  No missing values found")
        
        # Duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            self.logger.warning(f"  Duplicate rows: {duplicate_count}")
        else:
            self.logger.info("  No duplicate rows found")
    
    def _log_target_info(self, y: pd.Series) -> None:
        """Log information about the target variable.
        
        Args:
            y: Target variable Series
        """
        self.logger.info("Target variable overview:")
        self.logger.info(f"  Name: {y.name}")
        self.logger.info(f"  Data type: {y.dtype}")
        self.logger.info(f"  Non-null values: {y.count()}/{len(y)}")
        
        if y.dtype in ['int64', 'float64']:
            # Numerical target
            self.logger.info(f"  Statistics:")
            self.logger.info(f"    Mean: {y.mean():.4f}")
            self.logger.info(f"    Std: {y.std():.4f}")
            self.logger.info(f"    Min: {y.min():.4f}")
            self.logger.info(f"    Max: {y.max():.4f}")
            
            # Check for potential classification (few unique values)
            unique_values = y.nunique()
            if unique_values <= 20:
                self.logger.info(f"    Unique values: {unique_values}")
                value_counts = y.value_counts().head(10)
                self.logger.info(f"    Value distribution: {dict(value_counts)}")
        else:
            # Categorical target
            unique_values = y.nunique()
            self.logger.info(f"  Unique values: {unique_values}")
            
            if unique_values <= 20:
                value_counts = y.value_counts()
                self.logger.info(f"  Value distribution: {dict(value_counts)}")
                
                # Check for class imbalance
                if unique_values > 1:
                    min_class_pct = (value_counts.min() / len(y)) * 100
                    max_class_pct = (value_counts.max() / len(y)) * 100
                    if min_class_pct < 10:
                        self.logger.warning(f"  Potential class imbalance detected:")
                        self.logger.warning(f"    Smallest class: {min_class_pct:.1f}%")
                        self.logger.warning(f"    Largest class: {max_class_pct:.1f}%")
    
    def get_data_summary(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Get comprehensive data summary for logging.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Dict containing data summary statistics
        """
        summary = {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'target_column': y.name,
            'feature_columns': list(X.columns),
            'data_types': dict(X.dtypes.astype(str)),
            'missing_values': dict(X.isnull().sum()),
            'target_type': str(y.dtype),
            'target_unique_values': y.nunique()
        }
        
        # Add target statistics
        if y.dtype in ['int64', 'float64']:
            summary['target_stats'] = {
                'mean': float(y.mean()),
                'std': float(y.std()),
                'min': float(y.min()),
                'max': float(y.max()),
                'median': float(y.median())
            }
        else:
            summary['target_distribution'] = dict(y.value_counts().head(10))
        
        return summary