"""Configuration management for ML training pipeline."""

import os
import yaml
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator
from loguru import logger


class DataConfig(BaseModel):
    """Data configuration model."""
    source_path: str
    target_column: str
    feature_columns: Union[str, List[str]] = "auto"
    exclude_columns: List[str] = Field(default_factory=list)
    split: Dict[str, Any] = Field(default_factory=dict)


class FeatureEngineeringConfig(BaseModel):
    """Feature engineering configuration model."""
    enabled: bool = True
    categorical_encoding: Dict[str, Any] = Field(default_factory=dict)
    numerical_scaling: Dict[str, Any] = Field(default_factory=dict)
    feature_selection: Dict[str, Any] = Field(default_factory=dict)
    custom_features: List[Dict[str, str]] = Field(default_factory=list)


class ModelConfig(BaseModel):
    """Model configuration model."""
    type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    hyperparameter_tuning: Dict[str, Any] = Field(default_factory=dict)


class TrainingConfig(BaseModel):
    """Training configuration model."""
    cross_validation: Dict[str, Any] = Field(default_factory=dict)
    early_stopping: Dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    """Evaluation configuration model."""
    metrics: Dict[str, List[str]] = Field(default_factory=dict)
    interpretation: Dict[str, Any] = Field(default_factory=dict)


class ModelPersistenceConfig(BaseModel):
    """Model persistence configuration model."""
    save_path: str = "models/"
    model_format: str = "joblib"
    versioning: Dict[str, Any] = Field(default_factory=dict)
    registry: Dict[str, Any] = Field(default_factory=dict)


class LoggingConfig(BaseModel):
    """Logging configuration model."""
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}"
    file: str = "logs/ml_training.log"
    mlflow: Dict[str, Any] = Field(default_factory=dict)


class VisualizationConfig(BaseModel):
    """Visualization configuration model."""
    enabled: bool = True
    save_plots: bool = True
    plot_format: str = "png"
    plots_path: str = "plots/"
    plots: List[str] = Field(default_factory=list)


class ExperimentConfig(BaseModel):
    """Experiment configuration model."""
    name: str
    description: str = ""
    tags: List[str] = Field(default_factory=list)


class MLConfig(BaseModel):
    """Main ML configuration model."""
    experiment: ExperimentConfig
    data: DataConfig
    feature_engineering: FeatureEngineeringConfig = Field(default_factory=FeatureEngineeringConfig)
    model: ModelConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    model_persistence: ModelPersistenceConfig = Field(default_factory=ModelPersistenceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)


class MLConfigManager:
    """Manages ML configuration loading and validation."""
    
    def __init__(self, config_path: str = "ml_config.yaml"):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to the ML configuration file
        """
        self.config_path = Path(config_path)
        load_dotenv()  # Load environment variables from .env file
        
    def load_config(self) -> MLConfig:
        """Load and validate ML configuration from YAML file.
        
        Returns:
            MLConfig: Validated configuration object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"ML configuration file not found: {self.config_path}")
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                raw_config = yaml.safe_load(file)
                
            # Substitute environment variables
            processed_config = self._substitute_env_vars(raw_config)
            
            # Validate and create config object
            config = MLConfig(**processed_config)
            
            logger.info(f"ML configuration loaded successfully from {self.config_path}")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in ML configuration file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading ML configuration: {e}")
            raise
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """Recursively substitute environment variables in configuration.
        
        Args:
            obj: Configuration object (dict, list, or string)
            
        Returns:
            Object with environment variables substituted
        """
        if isinstance(obj, dict):
            return {key: self._substitute_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            env_var = obj[2:-1]  # Remove ${ and }
            return os.getenv(env_var, obj)  # Return original if env var not found
        else:
            return obj
    
    def save_config(self, config: MLConfig, output_path: str) -> None:
        """Save configuration to file for reproducibility.
        
        Args:
            config: Configuration object to save
            output_path: Path to save the configuration
        """
        try:
            config_dict = config.dict()
            
            with open(output_path, 'w', encoding='utf-8') as file:
                yaml.dump(config_dict, file, default_flow_style=False, indent=2)
                
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise