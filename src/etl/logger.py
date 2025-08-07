"""
Logging configuration for ETL pipeline
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


class ETLLogger:
    """Centralized logging configuration for ETL pipeline"""
    
    def __init__(self, config: dict):
        """
        Initialize logger with configuration
        
        Args:
            config: Logging configuration dictionary
        """
        self.config = config
        self._setup_logger()
    
    def _setup_logger(self):
        """Configure logging with file and console handlers"""
        # Create logs directory if it doesn't exist
        log_file = self.config.get('file', 'logs/etl.log')
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.config.get('level', 'INFO')),
            format=self.config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=[
                # File handler with rotation
                logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=5
                ),
                # Console handler
                logging.StreamHandler()
            ]
        )
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Get logger instance for specific module
        
        Args:
            name: Logger name (typically __name__)
            
        Returns:
            Logger instance
        """
        return logging.getLogger(name)