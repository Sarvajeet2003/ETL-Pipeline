"""
Schema validation for data quality assurance
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from ..logger import ETLLogger


class SchemaValidator:
    """Validates DataFrame schema and data quality"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize schema validator with configuration
        
        Args:
            config: Schema validation configuration
        """
        self.config = config
        self.logger = ETLLogger.get_logger(self.__class__.__name__)
        self.enabled = config.get('enabled', True)
        self.strict_mode = config.get('strict_mode', False)
    
    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame against schema requirements
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if not self.enabled:
            return True, []
        
        self.logger.info(f"Validating DataFrame schema with {len(df)} rows, {len(df.columns)} columns")
        
        errors = []
        
        # Validate required columns
        errors.extend(self._validate_required_columns(df))
        
        # Validate column types
        errors.extend(self._validate_column_types(df))
        
        # Validate data quality
        errors.extend(self._validate_data_quality(df))
        
        is_valid = len(errors) == 0
        
        if errors:
            self.logger.warning(f"Schema validation found {len(errors)} issues")
            for error in errors:
                self.logger.warning(f"  - {error}")
        else:
            self.logger.info("Schema validation passed")
        
        # In strict mode, raise exception on validation errors
        if self.strict_mode and not is_valid:
            raise ValueError(f"Schema validation failed: {'; '.join(errors)}")
        
        return is_valid, errors
    
    def _validate_required_columns(self, df: pd.DataFrame) -> List[str]:
        """Validate that all required columns are present"""
        errors = []
        required_columns = self.config.get('required_columns', [])
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {list(missing_columns)}")
        
        return errors
    
    def _validate_column_types(self, df: pd.DataFrame) -> List[str]:
        """Validate column data types"""
        errors = []
        expected_types = self.config.get('column_types', {})
        
        for column, expected_type in expected_types.items():
            if column not in df.columns:
                continue
            
            actual_type = str(df[column].dtype)
            
            # Map pandas dtypes to expected type names
            type_mapping = {
                'int64': ['int', 'integer', 'int64'],
                'float64': ['float', 'number', 'float64'],
                'object': ['string', 'text', 'object'],
                'bool': ['boolean', 'bool'],
                'datetime64[ns]': ['datetime', 'timestamp']
            }
            
            # Check if actual type matches expected type
            type_match = False
            for pandas_type, expected_names in type_mapping.items():
                if actual_type.startswith(pandas_type) and expected_type.lower() in expected_names:
                    type_match = True
                    break
            
            if not type_match:
                errors.append(f"Column '{column}' has type '{actual_type}', expected '{expected_type}'")
        
        return errors
    
    def _validate_data_quality(self, df: pd.DataFrame) -> List[str]:
        """Validate data quality metrics"""
        errors = []
        
        # Check for completely empty columns
        empty_columns = df.columns[df.isnull().all()].tolist()
        if empty_columns:
            errors.append(f"Completely empty columns found: {empty_columns}")
        
        # Check for columns with high null percentage
        null_threshold = self.config.get('max_null_percentage', 0.5)
        high_null_columns = []
        
        for column in df.columns:
            null_percentage = df[column].isnull().sum() / len(df)
            if null_percentage > null_threshold:
                high_null_columns.append(f"{column} ({null_percentage:.1%})")
        
        if high_null_columns:
            errors.append(f"Columns with high null percentage (>{null_threshold:.0%}): {high_null_columns}")
        
        # Check for duplicate rows (if not intentionally allowed)
        if not self.config.get('allow_duplicates', True):
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                errors.append(f"Found {duplicate_count} duplicate rows")
        
        return errors
    
    def get_data_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate data profile summary
        
        Args:
            df: DataFrame to profile
            
        Returns:
            Dictionary with data profile information
        """
        profile = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'columns': {}
        }
        
        for column in df.columns:
            col_profile = {
                'dtype': str(df[column].dtype),
                'null_count': int(df[column].isnull().sum()),
                'null_percentage': float(df[column].isnull().sum() / len(df)),
                'unique_count': int(df[column].nunique())
            }
            
            # Add type-specific statistics
            if df[column].dtype in ['int64', 'float64']:
                col_profile.update({
                    'min': float(df[column].min()) if not df[column].empty else None,
                    'max': float(df[column].max()) if not df[column].empty else None,
                    'mean': float(df[column].mean()) if not df[column].empty else None,
                    'std': float(df[column].std()) if not df[column].empty else None
                })
            elif df[column].dtype == 'object':
                col_profile.update({
                    'avg_length': float(df[column].astype(str).str.len().mean()) if not df[column].empty else None,
                    'max_length': int(df[column].astype(str).str.len().max()) if not df[column].empty else None
                })
            
            profile['columns'][column] = col_profile
        
        return profile