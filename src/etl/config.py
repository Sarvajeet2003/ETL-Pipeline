"""
Configuration management for ETL pipeline
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv


class ConfigManager:
    """Manages configuration loading and environment variable substitution"""
    
    def __init__(self, config_path: str = "config/etl_config.yaml"):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to the YAML configuration file
        """
        load_dotenv()  # Load environment variables from .env file
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and parse YAML configuration file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            return self._substitute_env_vars(config)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """Recursively substitute environment variables in configuration"""
        if isinstance(obj, dict):
            return {key: self._substitute_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            env_var = obj[2:-1]
            return os.getenv(env_var, obj)
        return obj
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key
        
        Args:
            key: Configuration key (e.g., 'logging.level')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_data_sources(self) -> Dict[str, Dict[str, Any]]:
        """Get all enabled data sources configuration"""
        sources = self.get('data_sources', {})
        return {name: config for name, config in sources.items() 
                if config.get('enabled', False)}
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration dictionary"""
        return self._config