"""
CSV file loader
"""

from typing import Dict, Any
import pandas as pd
from pathlib import Path
from .base import BaseLoader


class CSVLoader(BaseLoader):
    """Loader for CSV files"""
    
    def validate_config(self) -> bool:
        """Validate CSV loader configuration"""
        required_keys = ['path']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        return True
    
    def load(self, df: pd.DataFrame, metadata: Dict[str, Any] = None) -> bool:
        """
        Load DataFrame to CSV file
        
        Args:
            df: DataFrame to save
            metadata: Optional metadata (used for filename generation)
            
        Returns:
            True if save was successful
        """
        self.validate_config()
        
        output_path = Path(self.config['path'])
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        if metadata and 'source_name' in metadata:
            filename = f"processed_{metadata['source_name']}.csv"
        else:
            filename = "processed_data.csv"
        
        file_path = output_path / filename
        
        try:
            # Save DataFrame to CSV
            df.to_csv(
                file_path,
                index=self.config.get('include_index', False),
                encoding=self.config.get('encoding', 'utf-8')
            )
            
            self.logger.info(f"Successfully saved {len(df)} rows to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving CSV file {file_path}: {e}")
            return False