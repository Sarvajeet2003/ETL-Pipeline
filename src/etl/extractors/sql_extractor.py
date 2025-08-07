"""
SQL database extractor
"""

from typing import Iterator, Dict, Any
import pandas as pd
from sqlalchemy import create_engine, text
from .base import BaseExtractor


class SQLExtractor(BaseExtractor):
    """Extractor for SQL databases"""
    
    def validate_config(self) -> bool:
        """Validate SQL extractor configuration"""
        required_keys = ['connection_string', 'query']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        return True
    
    def extract(self) -> Iterator[pd.DataFrame]:
        """
        Extract data from SQL database
        
        Yields:
            DataFrame chunks from database query
        """
        self.validate_config()
        
        connection_string = self.config['connection_string']
        query = self.config['query']
        chunk_size = self.config.get('chunk_size', 10000)
        
        self.logger.info(f"Connecting to database: {connection_string.split('@')[-1] if '@' in connection_string else 'database'}")
        
        try:
            engine = create_engine(connection_string)
            
            # Execute query in chunks
            with engine.connect() as conn:
                for chunk in pd.read_sql(
                    text(query),
                    conn,
                    chunksize=chunk_size
                ):
                    # Add source metadata
                    chunk['_source_type'] = 'sql'
                    chunk['_source_query'] = query[:100] + '...' if len(query) > 100 else query
                    yield chunk
                    
        except Exception as e:
            self.logger.error(f"Error executing SQL query: {e}")
            raise