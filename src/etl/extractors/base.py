"""
Base extractor class for all data sources
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Iterator
import pandas as pd
from ..logger import ETLLogger


class BaseExtractor(ABC):
    """Abstract base class for all data extractors"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize extractor with configuration
        
        Args:
            config: Extractor-specific configuration
        """
        self.config = config
        self.logger = ETLLogger.get_logger(self.__class__.__name__)
    
    @abstractmethod
    def extract(self) -> Iterator[pd.DataFrame]:
        """
        Extract data from source
        
        Yields:
            DataFrame chunks of extracted data
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate extractor configuration
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the extraction
        
        Returns:
            Dictionary containing extraction metadata
        """
        return {
            'extractor_type': self.__class__.__name__,
            'config': self.config
        }