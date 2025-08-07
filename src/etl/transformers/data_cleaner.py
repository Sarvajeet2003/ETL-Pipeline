"""
Data cleaning and transformation utilities
"""

import re
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from ..logger import ETLLogger


class DataCleaner:
    """Handles data cleaning and basic transformations"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data cleaner with configuration
        
        Args:
            config: Data cleaning configuration
        """
        self.config = config
        self.logger = ETLLogger.get_logger(self.__class__.__name__)
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all configured cleaning operations to DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info(f"Cleaning DataFrame with {len(df)} rows")
        
        # Make a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Remove duplicates
        if self.config.get('remove_duplicates', False):
            initial_rows = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            removed_rows = initial_rows - len(cleaned_df)
            if removed_rows > 0:
                self.logger.info(f"Removed {removed_rows} duplicate rows")
        
        # Handle missing values
        cleaned_df = self._handle_missing_values(cleaned_df)
        
        # Clean text columns
        text_config = self.config.get('text_cleaning', {})
        if text_config:
            cleaned_df = self._clean_text_columns(cleaned_df, text_config)
        
        self.logger.info(f"Cleaning completed. Final DataFrame has {len(cleaned_df)} rows")
        return cleaned_df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on configuration"""
        strategy = self.config.get('handle_missing_values', 'drop')
        fill_value = self.config.get('fill_value')
        
        if strategy == 'drop':
            initial_rows = len(df)
            df = df.dropna()
            removed_rows = initial_rows - len(df)
            if removed_rows > 0:
                self.logger.info(f"Dropped {removed_rows} rows with missing values")
                
        elif strategy == 'fill':
            if fill_value is not None:
                df = df.fillna(fill_value)
                self.logger.info(f"Filled missing values with: {fill_value}")
            else:
                # Fill with appropriate defaults based on column type
                for column in df.columns:
                    if df[column].dtype in ['object', 'string']:
                        df[column] = df[column].fillna('')
                    elif df[column].dtype in ['int64', 'float64']:
                        df[column] = df[column].fillna(0)
                    elif df[column].dtype == 'bool':
                        df[column] = df[column].fillna(False)
                self.logger.info("Filled missing values with type-appropriate defaults")
                
        elif strategy == 'interpolate':
            # Only interpolate numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                df[column] = df[column].interpolate()
            self.logger.info(f"Interpolated missing values in {len(numeric_columns)} numeric columns")
        
        return df
    
    def _clean_text_columns(self, df: pd.DataFrame, text_config: Dict[str, Any]) -> pd.DataFrame:
        """Clean text columns based on configuration"""
        # Identify text columns
        text_columns = df.select_dtypes(include=['object', 'string']).columns
        
        for column in text_columns:
            if column.startswith('_'):  # Skip metadata columns
                continue
                
            self.logger.debug(f"Cleaning text column: {column}")
            
            # Convert to string and handle NaN
            df[column] = df[column].astype(str).replace('nan', '')
            
            if text_config.get('remove_extra_whitespace', False):
                df[column] = df[column].str.strip()
                df[column] = df[column].str.replace(r'\s+', ' ', regex=True)
            
            if text_config.get('lowercase', False):
                df[column] = df[column].str.lower()
            
            if text_config.get('remove_special_chars', False):
                # Keep alphanumeric, spaces, and basic punctuation
                df[column] = df[column].str.replace(r'[^\w\s\.\,\!\?\-]', '', regex=True)
        
        return df
    
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names (lowercase, replace spaces with underscores)
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized column names
        """
        # Store original column mapping for reference
        original_columns = df.columns.tolist()
        
        # Standardize column names
        new_columns = []
        for col in df.columns:
            # Convert to lowercase and replace spaces/special chars with underscores
            new_col = re.sub(r'[^\w]', '_', col.lower())
            # Remove multiple consecutive underscores
            new_col = re.sub(r'_+', '_', new_col)
            # Remove leading/trailing underscores
            new_col = new_col.strip('_')
            new_columns.append(new_col)
        
        df.columns = new_columns
        
        # Log column name changes
        changes = [(orig, new) for orig, new in zip(original_columns, new_columns) if orig != new]
        if changes:
            self.logger.info(f"Standardized {len(changes)} column names")
            for orig, new in changes:
                self.logger.debug(f"  '{orig}' -> '{new}'")
        
        return df