"""
Data extraction module for ETL pipeline
"""
import pandas as pd
from pathlib import Path
from typing import List, Optional
import glob

from src.utils.logging import get_logger
from src.utils.exceptions import ETLException

logger = get_logger()

class DataExtractor:
    """Extract data from various sources"""
    
    def __init__(self, source_path: str, file_pattern: str = "*.csv"):
        self.source_path = Path(source_path)
        self.file_pattern = file_pattern
        
    def extract_csv_files(self) -> pd.DataFrame:
        """
        Extract data from CSV files in the source directory
        
        Returns:
            Combined DataFrame from all matching CSV files
        """
        try:
            # Find all matching files
            pattern = str(self.source_path / self.file_pattern)
            files = glob.glob(pattern)
            
            if not files:
                raise ETLException(f"No files found matching pattern: {pattern}")
            
            logger.info(f"Found {len(files)} files to extract: {files}")
            
            # Read and combine all CSV files
            dataframes = []
            for file_path in files:
                logger.info(f"Reading file: {file_path}")
                df = pd.read_csv(file_path)
                df['source_file'] = Path(file_path).name
                dataframes.append(df)
                logger.info(f"Loaded {len(df)} rows from {file_path}")
            
            # Combine all dataframes
            combined_df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Combined dataset shape: {combined_df.shape}")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error during data extraction: {str(e)}")
            raise ETLException(f"Data extraction failed: {str(e)}")
    
    def extract_single_csv(self, file_path: str) -> pd.DataFrame:
        """
        Extract data from a single CSV file
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with the CSV data
        """
        try:
            full_path = self.source_path / file_path
            logger.info(f"Reading single CSV file: {full_path}")
            
            df = pd.read_csv(full_path)
            df['source_file'] = Path(file_path).name
            
            logger.info(f"Loaded {len(df)} rows from {full_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {str(e)}")
            raise ETLException(f"Failed to read CSV file: {str(e)}")
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get basic information about the extracted data
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with data information
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
            'missing_values': {col: int(count) for col, count in df.isnull().sum().to_dict().items()},
            'memory_usage': int(df.memory_usage(deep=True).sum()),
            'duplicate_rows': int(df.duplicated().sum())
        }
        
        logger.info(f"Data info: {info}")
        return info