"""
Database loader
"""

from typing import Dict, Any
import pandas as pd
from sqlalchemy import create_engine, text
from .base import BaseLoader


class DatabaseLoader(BaseLoader):
    """Loader for database tables"""
    
    def validate_config(self) -> bool:
        """Validate database loader configuration"""
        required_keys = ['connection_string', 'table_name']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        return True
    
    def load(self, df: pd.DataFrame, metadata: Dict[str, Any] = None) -> bool:
        """
        Load DataFrame to database table
        
        Args:
            df: DataFrame to save
            metadata: Optional metadata
            
        Returns:
            True if save was successful
        """
        self.validate_config()
        
        connection_string = self.config['connection_string']
        table_name = self.config['table_name']
        if_exists = self.config.get('if_exists', 'append')  # append, replace, fail
        
        try:
            engine = create_engine(connection_string)
            
            # Save DataFrame to database
            df.to_sql(
                table_name,
                engine,
                if_exists=if_exists,
                index=self.config.get('include_index', False),
                chunksize=self.config.get('chunksize', 10000)
            )
            
            self.logger.info(f"Successfully saved {len(df)} rows to table: {table_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving to database table {table_name}: {e}")
            return False