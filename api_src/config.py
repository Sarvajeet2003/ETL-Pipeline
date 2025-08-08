"""Configuration management for ML Model Serving API."""

import os
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv


class AppConfig(BaseModel):
    """Application configuration."""
    title: str = "ML Model Serving API"
    description: str = "Production-ready API for serving machine learning models"
    version: str = "1.0.0"
    debug: bool = False
    environment: str = "production"


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    log_level: str = "info"
    access_log: bool = True


class APIKeyConfig(BaseModel):
    """API Key authentication configuration."""
    enabled: bool = True
    header_name: str = "X-API-Key"
    valid_keys: List[str] = Field(default_factory=list)


class OAuth2Config(BaseModel):
    """OAuth2 authentication configuration."""
    enabled: bool = False
    secret_key: str = ""
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7


class CORSConfig(BaseModel):
    """CORS configuration."""
    enabled: bool = True
    allow_origins: List[str] = Field(default_factory=lambda: ["*"])
    allow_credentials: bool = True
    allow_methods: List[str] = Field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    allow_headers: List[str] = Field(default_factory=lambda: ["*"])


class RateLimitingConfig(BaseModel):
    """Rate limiting configuration."""
    enabled: bool = True
    requests_per_minute: int = 100
    burst_size: int = 20


class SecurityConfig(BaseModel):
    """Security configuration."""
    api_key: APIKeyConfig = Field(default_factory=APIKeyConfig)
    oauth2: OAuth2Config = Field(default_factory=OAuth2Config)
    cors: CORSConfig = Field(default_factory=CORSConfig)
    rate_limiting: RateLimitingConfig = Field(default_factory=RateLimitingConfig)


class ModelConfig(BaseModel):
    """Individual model configuration."""
    name: str
    path: str
    type: str = "ml_pipeline"
    warm_up: bool = True
    cache_predictions: bool = False
    max_batch_size: int = 1000


class DatabaseConfig(BaseModel):
    """Database configuration."""
    enabled: bool = False
    url: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False


class RedisConfig(BaseModel):
    """Redis configuration."""
    enabled: bool = False
    url: str = ""
    password: str = ""
    db: int = 0
    max_connections: int = 10


class RequestLoggingConfig(BaseModel):
    """Request logging configuration."""
    enabled: bool = True
    include_body: bool = False
    include_headers: bool = False


class PerformanceLoggingConfig(BaseModel):
    """Performance logging configuration."""
    enabled: bool = True
    slow_request_threshold: float = 1.0


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"
    file: str = "logs/api.log"
    max_size: str = "10MB"
    backup_count: int = 5
    request_logging: RequestLoggingConfig = Field(default_factory=RequestLoggingConfig)
    performance_logging: PerformanceLoggingConfig = Field(default_factory=PerformanceLoggingConfig)


class MetricsConfig(BaseModel):
    """Metrics configuration."""
    enabled: bool = True
    endpoint: str = "/metrics"
    include_default_metrics: bool = True


class HealthCheckConfig(BaseModel):
    """Health check configuration."""
    enabled: bool = True
    endpoint: str = "/health"
    detailed_endpoint: str = "/health/detailed"


class PrometheusConfig(BaseModel):
    """Prometheus configuration."""
    enabled: bool = True
    registry: str = "default"


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    health_check: HealthCheckConfig = Field(default_factory=HealthCheckConfig)
    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)


class TrainingJobConfig(BaseModel):
    """Training job configuration."""
    config_path: str = "ml_config.yaml"
    output_path: str = "models/"
    notification_webhook: str = ""


class AsyncTrainingConfig(BaseModel):
    """Async training configuration."""
    enabled: bool = False
    broker_url: str = ""
    result_backend: str = ""


class TrainingConfig(BaseModel):
    """Training configuration."""
    enabled: bool = True
    trigger_endpoint: str = "/retrain"
    job_config: TrainingJobConfig = Field(default_factory=TrainingJobConfig)
    async_training: AsyncTrainingConfig = Field(default_factory=AsyncTrainingConfig)


class InputValidationConfig(BaseModel):
    """Input validation configuration."""
    enabled: bool = True
    strict_mode: bool = True
    max_payload_size: str = "10MB"


class OutputValidationConfig(BaseModel):
    """Output validation configuration."""
    enabled: bool = True
    include_confidence: bool = True
    include_metadata: bool = True


class ValidationConfig(BaseModel):
    """Validation configuration."""
    input_validation: InputValidationConfig = Field(default_factory=InputValidationConfig)
    output_validation: OutputValidationConfig = Field(default_factory=OutputValidationConfig)


class CachingConfig(BaseModel):
    """Caching configuration."""
    enabled: bool = False
    backend: str = "redis"
    ttl: int = 3600
    max_size: int = 1000


class FeaturesConfig(BaseModel):
    """Feature flags configuration."""
    batch_prediction: bool = True
    model_explanation: bool = False
    a_b_testing: bool = False
    model_monitoring: bool = True


class APIConfig(BaseModel):
    """Main API configuration."""
    app: AppConfig = Field(default_factory=AppConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    models: Dict[str, ModelConfig] = Field(default_factory=dict)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    caching: CachingConfig = Field(default_factory=CachingConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)


class ConfigManager:
    """Manages API configuration loading and validation."""
    
    def __init__(self, config_path: str = "api_config.yaml"):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to the API configuration file
        """
        self.config_path = Path(config_path)
        load_dotenv()  # Load environment variables
        
    def load_config(self) -> APIConfig:
        """Load and validate API configuration.
        
        Returns:
            APIConfig: Validated configuration object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"API configuration file not found: {self.config_path}")
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                raw_config = yaml.safe_load(file)
                
            # Substitute environment variables
            processed_config = self._substitute_env_vars(raw_config)
            
            # Validate and create config object
            config = APIConfig(**processed_config)
            
            return config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in API configuration file: {e}")
        except Exception as e:
            raise Exception(f"Error loading API configuration: {e}")
    
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


# Global configuration instance
config_manager = ConfigManager()
config = config_manager.load_config()