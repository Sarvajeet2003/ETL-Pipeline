"""
Data preprocessing module for ML pipeline
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

from src.utils.logging import get_logger
from src.utils.exceptions import MLException

logger = get_logger()

class MLPreprocessor:
    """Preprocess data for machine learning"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.target_column = config['target_column']
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load cleaned data from ETL pipeline
        
        Args:
            file_path: Path to the cleaned data file
            
        Returns:
            Loaded DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded data with shape: {df.shape}")
            
            # Remove source_file column if it exists (added by ETL)
            if 'source_file' in df.columns:
                df = df.drop('source_file', axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise MLException(f"Failed to load data: {str(e)}")
    
    def split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split data into features and target
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features, target)
        """
        try:
            if self.target_column not in df.columns:
                raise MLException(f"Target column '{self.target_column}' not found in data")
            
            X = df.drop(self.target_column, axis=1)
            y = df[self.target_column]
            
            # Remove ID columns if they exist
            id_columns = [col for col in X.columns if 'id' in col.lower()]
            if id_columns:
                X = X.drop(id_columns, axis=1)
                logger.info(f"Removed ID columns: {id_columns}")
            
            self.feature_columns = list(X.columns)
            logger.info(f"Features: {self.feature_columns}")
            logger.info(f"Target: {self.target_column}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error splitting features and target: {str(e)}")
            raise MLException(f"Failed to split features and target: {str(e)}")
    
    def encode_categorical_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            X: Features DataFrame
            fit: Whether to fit encoders (True for training, False for inference)
            
        Returns:
            DataFrame with encoded features
        """
        try:
            X_encoded = X.copy()
            encoding_method = self.config.get('encoding', 'onehot')
            
            categorical_columns = X_encoded.select_dtypes(include=['object', 'category']).columns
            
            if len(categorical_columns) == 0:
                logger.info("No categorical columns found")
                return X_encoded
            
            logger.info(f"Encoding categorical columns: {list(categorical_columns)}")
            
            if encoding_method == 'onehot':
                for col in categorical_columns:
                    if fit:
                        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                        encoded_data = encoder.fit_transform(X_encoded[[col]])
                        self.encoders[col] = encoder
                    else:
                        if col not in self.encoders:
                            raise MLException(f"Encoder for column {col} not found")
                        encoder = self.encoders[col]
                        encoded_data = encoder.transform(X_encoded[[col]])
                    
                    # Create column names for one-hot encoded features
                    feature_names = [f"{col}_{category}" for category in encoder.categories_[0]]
                    encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=X_encoded.index)
                    
                    # Replace original column with encoded columns
                    X_encoded = X_encoded.drop(col, axis=1)
                    X_encoded = pd.concat([X_encoded, encoded_df], axis=1)
            
            elif encoding_method == 'label':
                for col in categorical_columns:
                    if fit:
                        encoder = LabelEncoder()
                        X_encoded[col] = encoder.fit_transform(X_encoded[col].astype(str))
                        self.encoders[col] = encoder
                    else:
                        if col not in self.encoders:
                            raise MLException(f"Encoder for column {col} not found")
                        encoder = self.encoders[col]
                        # Handle unknown categories
                        X_encoded[col] = X_encoded[col].astype(str)
                        mask = X_encoded[col].isin(encoder.classes_)
                        X_encoded.loc[mask, col] = encoder.transform(X_encoded.loc[mask, col])
                        X_encoded.loc[~mask, col] = -1  # Unknown category
            
            logger.info(f"Categorical encoding completed. New shape: {X_encoded.shape}")
            
            # Save final feature names after encoding (only when fitting)
            if fit:
                self.final_feature_names = list(X_encoded.columns)
            
            return X_encoded
            
        except Exception as e:
            logger.error(f"Error encoding categorical features: {str(e)}")
            raise MLException(f"Failed to encode categorical features: {str(e)}")
    
    def scale_numerical_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            X: Features DataFrame
            fit: Whether to fit scalers (True for training, False for inference)
            
        Returns:
            DataFrame with scaled features
        """
        try:
            X_scaled = X.copy()
            scaling_method = self.config.get('scaling', 'standard')
            
            if scaling_method == 'none':
                logger.info("Feature scaling disabled")
                return X_scaled
            
            numerical_columns = X_scaled.select_dtypes(include=[np.number]).columns
            
            if len(numerical_columns) == 0:
                logger.info("No numerical columns found")
                return X_scaled
            
            logger.info(f"Scaling numerical columns: {list(numerical_columns)}")
            
            # Choose scaler based on configuration
            if scaling_method == 'standard':
                scaler_class = StandardScaler
            elif scaling_method == 'minmax':
                scaler_class = MinMaxScaler
            elif scaling_method == 'robust':
                scaler_class = RobustScaler
            else:
                raise MLException(f"Unknown scaling method: {scaling_method}")
            
            if fit:
                scaler = scaler_class()
                X_scaled[numerical_columns] = scaler.fit_transform(X_scaled[numerical_columns])
                self.scalers['numerical'] = scaler
            else:
                if 'numerical' not in self.scalers:
                    raise MLException("Numerical scaler not found")
                scaler = self.scalers['numerical']
                X_scaled[numerical_columns] = scaler.transform(X_scaled[numerical_columns])
            
            logger.info(f"Numerical scaling completed using {scaling_method} method")
            return X_scaled
            
        except Exception as e:
            logger.error(f"Error scaling numerical features: {str(e)}")
            raise MLException(f"Failed to scale numerical features: {str(e)}")
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        try:
            test_size = self.config.get('test_size', 0.2)
            val_size = self.config.get('validation_size', 0.2)
            random_state = self.config.get('random_state', 42)
            
            # First split: separate test set
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Second split: separate train and validation from remaining data
            val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
            )
            
            logger.info(f"Data split completed:")
            logger.info(f"  Train: {X_train.shape[0]} samples")
            logger.info(f"  Validation: {X_val.shape[0]} samples")
            logger.info(f"  Test: {X_test.shape[0]} samples")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise MLException(f"Failed to split data: {str(e)}")
    
    def save_preprocessors(self, model_path: str) -> None:
        """
        Save fitted preprocessors for later use
        
        Args:
            model_path: Directory to save preprocessors
        """
        try:
            model_dir = Path(model_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save scalers
            if self.scalers:
                scalers_path = model_dir / "scalers.joblib"
                joblib.dump(self.scalers, scalers_path)
                logger.info(f"Saved scalers to {scalers_path}")
            
            # Save encoders
            if self.encoders:
                encoders_path = model_dir / "encoders.joblib"
                joblib.dump(self.encoders, encoders_path)
                logger.info(f"Saved encoders to {encoders_path}")
            
            # Save feature columns (original features before encoding)
            features_path = model_dir / "feature_columns.joblib"
            joblib.dump(self.feature_columns, features_path)
            logger.info(f"Saved feature columns to {features_path}")
            
            # Also save the final processed feature names (after encoding)
            # This will be set during the preprocessing step
            if hasattr(self, 'final_feature_names'):
                final_features_path = model_dir / "final_feature_names.joblib"
                joblib.dump(self.final_feature_names, final_features_path)
                logger.info(f"Saved final feature names to {final_features_path}")
            
        except Exception as e:
            logger.error(f"Error saving preprocessors: {str(e)}")
            raise MLException(f"Failed to save preprocessors: {str(e)}")
    
    def load_preprocessors(self, model_path: str) -> None:
        """
        Load fitted preprocessors
        
        Args:
            model_path: Directory containing preprocessors
        """
        try:
            model_dir = Path(model_path)
            
            # Load scalers
            scalers_path = model_dir / "scalers.joblib"
            if scalers_path.exists():
                self.scalers = joblib.load(scalers_path)
                logger.info(f"Loaded scalers from {scalers_path}")
            
            # Load encoders
            encoders_path = model_dir / "encoders.joblib"
            if encoders_path.exists():
                self.encoders = joblib.load(encoders_path)
                logger.info(f"Loaded encoders from {encoders_path}")
            
            # Load feature columns
            features_path = model_dir / "feature_columns.joblib"
            if features_path.exists():
                self.feature_columns = joblib.load(features_path)
                logger.info(f"Loaded feature columns from {features_path}")
            
            # Load final feature names (after encoding)
            final_features_path = model_dir / "final_feature_names.joblib"
            if final_features_path.exists():
                self.final_feature_names = joblib.load(final_features_path)
                logger.info(f"Loaded final feature names from {final_features_path}")
            else:
                self.final_feature_names = None
            
        except Exception as e:
            logger.error(f"Error loading preprocessors: {str(e)}")
            raise MLException(f"Failed to load preprocessors: {str(e)}")