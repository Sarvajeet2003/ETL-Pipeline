"""
Data cleaning module for ETL pipeline
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from scipy import stats

from src.utils.logging import get_logger
from src.utils.exceptions import ETLException

logger = get_logger()

class DataCleaner:
    """Clean and preprocess data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cleaning_stats = {}
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with duplicates removed
        """
        try:
            initial_rows = len(df)
            
            if self.config.get('drop_duplicates', True):
                # Keep first occurrence of duplicates
                df_cleaned = df.drop_duplicates(keep='first')
                duplicates_removed = initial_rows - len(df_cleaned)
                
                logger.info(f"Removed {duplicates_removed} duplicate rows")
                self.cleaning_stats['duplicates_removed'] = duplicates_removed
                
                return df_cleaned
            else:
                logger.info("Duplicate removal disabled in config")
                self.cleaning_stats['duplicates_removed'] = 0
                return df.copy()
                
        except Exception as e:
            logger.error(f"Error removing duplicates: {str(e)}")
            raise ETLException(f"Failed to remove duplicates: {str(e)}")
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values based on configuration
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        try:
            df_cleaned = df.copy()
            missing_before = df_cleaned.isnull().sum().sum()
            
            handle_method = self.config.get('handle_missing', 'drop')
            
            if handle_method == 'drop':
                # Drop rows with any missing values
                df_cleaned = df_cleaned.dropna()
                logger.info(f"Dropped rows with missing values")
                
            elif handle_method == 'fill_mean':
                # Fill numeric columns with mean
                numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    if df_cleaned[col].isnull().any():
                        mean_value = df_cleaned[col].mean()
                        df_cleaned[col].fillna(mean_value, inplace=True)
                        logger.info(f"Filled missing values in {col} with mean: {mean_value:.2f}")
                
                # Fill categorical columns with mode
                categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
                for col in categorical_columns:
                    if df_cleaned[col].isnull().any():
                        mode_value = df_cleaned[col].mode().iloc[0] if not df_cleaned[col].mode().empty else 'Unknown'
                        df_cleaned[col].fillna(mode_value, inplace=True)
                        logger.info(f"Filled missing values in {col} with mode: {mode_value}")
            
            elif handle_method == 'fill_median':
                # Fill numeric columns with median
                numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    if df_cleaned[col].isnull().any():
                        median_value = df_cleaned[col].median()
                        df_cleaned[col].fillna(median_value, inplace=True)
                        logger.info(f"Filled missing values in {col} with median: {median_value:.2f}")
            
            missing_after = df_cleaned.isnull().sum().sum()
            self.cleaning_stats['missing_values_handled'] = missing_before - missing_after
            
            logger.info(f"Handled {missing_before - missing_after} missing values using method: {handle_method}")
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise ETLException(f"Failed to handle missing values: {str(e)}")
    
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers from numeric columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with outliers removed
        """
        try:
            df_cleaned = df.copy()
            initial_rows = len(df_cleaned)
            
            outlier_method = self.config.get('outlier_method', 'iqr')
            
            if outlier_method == 'none':
                logger.info("Outlier removal disabled")
                self.cleaning_stats['outliers_removed'] = 0
                return df_cleaned
            
            numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
            outlier_indices = set()
            
            for col in numeric_columns:
                if outlier_method == 'iqr':
                    # Interquartile Range method
                    Q1 = df_cleaned[col].quantile(0.25)
                    Q3 = df_cleaned[col].quantile(0.75)
                    IQR = Q3 - Q1
                    threshold = self.config.get('outlier_threshold', 1.5)
                    
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    col_outliers = df_cleaned[
                        (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
                    ].index
                    
                elif outlier_method == 'zscore':
                    # Z-score method
                    threshold = self.config.get('outlier_threshold', 3)
                    z_scores = np.abs(stats.zscore(df_cleaned[col].dropna()))
                    col_outliers = df_cleaned[col].dropna().index[z_scores > threshold]
                
                outlier_indices.update(col_outliers)
                logger.info(f"Found {len(col_outliers)} outliers in column {col}")
            
            # Remove outliers
            df_cleaned = df_cleaned.drop(index=outlier_indices)
            outliers_removed = initial_rows - len(df_cleaned)
            
            self.cleaning_stats['outliers_removed'] = outliers_removed
            logger.info(f"Removed {outliers_removed} outlier rows using {outlier_method} method")
            
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Error removing outliers: {str(e)}")
            raise ETLException(f"Failed to remove outliers: {str(e)}")
    
    def standardize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize data types for consistency
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized data types
        """
        try:
            df_cleaned = df.copy()
            
            # Convert string representations of numbers to numeric
            for col in df_cleaned.columns:
                if df_cleaned[col].dtype == 'object':
                    # Try to convert to numeric
                    numeric_series = pd.to_numeric(df_cleaned[col], errors='coerce')
                    if not numeric_series.isnull().all():
                        # If conversion was successful for most values
                        non_null_ratio = numeric_series.notna().sum() / len(numeric_series)
                        if non_null_ratio > 0.8:  # 80% of values are numeric
                            df_cleaned[col] = numeric_series
                            logger.info(f"Converted column {col} to numeric type")
            
            # Ensure categorical columns are properly typed
            categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if df_cleaned[col].nunique() < len(df_cleaned) * 0.5:  # Less than 50% unique values
                    df_cleaned[col] = df_cleaned[col].astype('category')
                    logger.info(f"Converted column {col} to category type")
            
            self.cleaning_stats['data_types_standardized'] = True
            logger.info("Data types standardization completed")
            
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Error standardizing data types: {str(e)}")
            raise ETLException(f"Failed to standardize data types: {str(e)}")
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """Get comprehensive cleaning report"""
        return {
            'timestamp': pd.Timestamp.now().isoformat(),
            'cleaning_stats': self.cleaning_stats,
            'config_used': self.config
        }