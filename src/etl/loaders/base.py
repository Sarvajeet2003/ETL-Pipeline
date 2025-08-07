"""
Base loader class for all output destinations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
from ..logger import ETLLogger


class BaseLoader(ABC):
    """Abstract base class for all data loaders"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize loader with configuration
        
        Args:
            config: Loader-specific configuration
        """
        self.config = config
        self.logger = ETLLogger.get_logger(self.__class__.__name__)
    
    @abstractmethod
    def load(self, df: pd.DataFrame, metadata: Dict[str, Any] = None) -> bool:
        """
        Load DataFrame to destination
        
        Args:
            df: DataFrame to load
            metadata: Optional metadata about the data
            
        Returns:
            True if load was successful
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate loader configuration
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        pass