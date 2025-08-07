"""
CSV file extractor
"""

import glob
from typing import Iterator, Dict, Any
import pandas as pd
from pathlib import Path
from .base import BaseExtractor


class CSVExtractor(BaseExtractor):
    """Extractor for CSV files"""
    
    def validate_config(self) -> bool:
        """Validate CSV extractor configuration"""
        required_keys = ['paths']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        return True
    
    def extract(self) -> Iterator[pd.DataFrame]:
        """
        Extract data from CSV files
        
        Yields:
            DataFrame for each CSV file
        """
        self.validate_config()
        
        paths = self.config['paths']
        encoding = self.config.get('encoding', 'utf-8')
        delimiter = self.config.get('delimiter', ',')
        chunk_size = self.config.get('chunk_size', None)
        
        for path_pattern in paths:
            files = glob.glob(path_pattern)
            
            if not files:
                self.logger.warning(f"No files found matching pattern: {path_pattern}")
                continue
            
            for file_path in files:
                self.logger.info(f"Extracting data from: {file_path}")
                
                try:
                    # Read CSV in chunks if specified
                    if chunk_size:
                        for chunk in pd.read_csv(
                            file_path,
                            encoding=encoding,
                            delimiter=delimiter,
                            chunksize=chunk_size
                        ):
                            # Add source metadata
                            chunk['_source_file'] = Path(file_path).name
                            chunk['_source_type'] = 'csv'
                            yield chunk
                    else:
                        df = pd.read_csv(
                            file_path,
                            encoding=encoding,
                            delimiter=delimiter
                        )
                        # Add source metadata
                        df['_source_file'] = Path(file_path).name
                        df['_source_type'] = 'csv'
                        yield df
                        
                except Exception as e:
                    self.logger.error(f"Error reading CSV file {file_path}: {e}")
                    continue