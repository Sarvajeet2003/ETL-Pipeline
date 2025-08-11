"""
Configuration management utilities
"""
import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._configs = {}
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_name in self._configs:
            return self._configs[config_name]
        
        config_path = self.config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        self._configs[config_name] = config
        return config
    
    def get_etl_config(self) -> Dict[str, Any]:
        """Get ETL configuration"""
        return self.load_config("etl_config")
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Get ML configuration"""
        return self.load_config("ml_config")
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return self.load_config("api_config")

# Global config manager instance
config_manager = ConfigManager()