"""
Logging utilities for the data science pipeline
"""
import sys
import logging
from pathlib import Path
from typing import Optional

def setup_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    component: str = "pipeline"
) -> None:
    """
    Setup logger with file and console output
    
    Args:
        log_file: Path to log file
        level: Logging level
        component: Component name for log formatting
    """
    # Create logger
    logger = logging.getLogger(component)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    logger.info(f"Logger initialized for {component} component")

def get_logger():
    """Get the configured logger instance"""
    # Return the root logger or create a default one
    logger = logging.getLogger()
    if not logger.handlers:
        # Setup basic logging if no handlers exist
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    return logger