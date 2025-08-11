"""
Data validation module for ETL pipeline
"""
import pandas as pd
from typing import List, Dict, Any, Optional
import numpy as np

from src.utils.logging import get_logger
from src.utils.exceptions import DataValidationException

logger = get_logger()

class DataValidator:
    """Validate data quality and structure"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_results = {}
    
    def validate_structure(self, df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> bool:
        """
        Validate data structure and required columns
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if validation passes
        """
        try:
            logger.info("Starting data structure validation")
            
            # Check if DataFrame is empty
            if df.empty:
                raise DataValidationException("DataFrame is empty")
            
            # Check minimum number of rows
            min_rows = self.config.get('min_rows', 1)
            if len(df) < min_rows:
                raise DataValidationException(f"Dataset has {len(df)} rows, minimum required: {min_rows}")
            
            # Check required columns
            if required_columns:
                missing_columns = set(required_columns) - set(df.columns)
                if missing_columns:
                    raise DataValidationException(f"Missing required columns: {missing_columns}")
            
            self.validation_results['structure'] = {
                'passed': True,
                'rows': len(df),
                'columns': len(df.columns),
                'required_columns_present': not bool(required_columns) or set(required_columns).issubset(set(df.columns))
            }
            
            logger.info("Data structure validation passed")
            return True
            
        except DataValidationException:
            raise
        except Exception as e:
            logger.error(f"Error during structure validation: {str(e)}")
            raise DataValidationException(f"Structure validation failed: {str(e)}")
    
    def validate_data_quality(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality metrics
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if validation passes
        """
        try:
            logger.info("Starting data quality validation")
            
            # Check missing values percentage
            max_missing_pct = self.config.get('max_missing_percentage', 0.5)
            missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
            
            if missing_pct > max_missing_pct:
                raise DataValidationException(
                    f"Missing values percentage ({missing_pct:.2%}) exceeds threshold ({max_missing_pct:.2%})"
                )
            
            # Check for completely empty columns
            empty_columns = df.columns[df.isnull().all()].tolist()
            if empty_columns:
                logger.warning(f"Found completely empty columns: {empty_columns}")
            
            # Check data types consistency
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df[col].dtype == 'object':
                    logger.warning(f"Column {col} should be numeric but has object dtype")
            
            self.validation_results['quality'] = {
                'passed': True,
                'missing_percentage': missing_pct,
                'empty_columns': empty_columns,
                'numeric_columns': len(numeric_columns)
            }
            
            logger.info("Data quality validation passed")
            return True
            
        except DataValidationException:
            raise
        except Exception as e:
            logger.error(f"Error during quality validation: {str(e)}")
            raise DataValidationException(f"Quality validation failed: {str(e)}")
    
    def validate_business_rules(self, df: pd.DataFrame) -> bool:
        """
        Validate business-specific rules
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if validation passes
        """
        try:
            logger.info("Starting business rules validation")
            
            issues = []
            
            # Example business rules - customize based on your data
            if 'age' in df.columns:
                invalid_ages = df[(df['age'] < 0) | (df['age'] > 120)]
                if not invalid_ages.empty:
                    issues.append(f"Found {len(invalid_ages)} records with invalid ages")
            
            if 'monthly_charges' in df.columns:
                negative_charges = df[df['monthly_charges'] < 0]
                if not negative_charges.empty:
                    issues.append(f"Found {len(negative_charges)} records with negative charges")
            
            if 'tenure' in df.columns:
                invalid_tenure = df[df['tenure'] < 0]
                if not invalid_tenure.empty:
                    issues.append(f"Found {len(invalid_tenure)} records with negative tenure")
            
            # Log issues but don't fail validation (can be handled in cleaning)
            if issues:
                for issue in issues:
                    logger.warning(f"Business rule issue: {issue}")
            
            self.validation_results['business_rules'] = {
                'passed': True,
                'issues': issues
            }
            
            logger.info("Business rules validation completed")
            return True
            
        except Exception as e:
            logger.error(f"Error during business rules validation: {str(e)}")
            raise DataValidationException(f"Business rules validation failed: {str(e)}")
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get comprehensive validation report"""
        return {
            'timestamp': pd.Timestamp.now().isoformat(),
            'results': self.validation_results,
            'overall_status': all(
                result.get('passed', False) 
                for result in self.validation_results.values()
            )
        }