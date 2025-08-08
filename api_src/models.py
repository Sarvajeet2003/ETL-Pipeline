"""Pydantic models for API requests and responses."""

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class PredictionStatus(str, Enum):
    """Prediction status enumeration."""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"


class ModelStatus(str, Enum):
    """Model status enumeration."""
    LOADED = "loaded"
    LOADING = "loading"
    ERROR = "error"
    NOT_FOUND = "not_found"


class TrainingStatus(str, Enum):
    """Training status enumeration."""
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HealthStatus(str, Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


# Request Models

class PredictionRequest(BaseModel):
    """Single prediction request model."""
    data: Dict[str, Any] = Field(..., description="Input data for prediction")
    model_name: Optional[str] = Field(None, description="Name of model to use (optional)")
    include_confidence: bool = Field(False, description="Include prediction confidence")
    include_explanation: bool = Field(False, description="Include model explanation")
    
    class Config:
        schema_extra = {
            "example": {
                "data": {
                    "age": 35,
                    "experience_years": 8,
                    "education_level": 3,
                    "department": "Engineering",
                    "location": "SF"
                },
                "model_name": "salary_prediction",
                "include_confidence": True,
                "include_explanation": False
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""
    data: List[Dict[str, Any]] = Field(..., description="List of input data for predictions")
    model_name: Optional[str] = Field(None, description="Name of model to use (optional)")
    include_confidence: bool = Field(False, description="Include prediction confidence")
    include_explanation: bool = Field(False, description="Include model explanation")
    
    @validator('data')
    def validate_batch_size(cls, v):
        if len(v) == 0:
            raise ValueError("Batch cannot be empty")
        if len(v) > 1000:  # This should come from config
            raise ValueError("Batch size too large (max 1000)")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "data": [
                    {
                        "age": 35,
                        "experience_years": 8,
                        "education_level": 3,
                        "department": "Engineering",
                        "location": "SF"
                    },
                    {
                        "age": 28,
                        "experience_years": 3,
                        "education_level": 2,
                        "department": "Marketing",
                        "location": "NYC"
                    }
                ],
                "model_name": "salary_prediction",
                "include_confidence": True
            }
        }


class RetrainRequest(BaseModel):
    """Model retraining request model."""
    model_name: Optional[str] = Field(None, description="Name of model to retrain")
    config_overrides: Optional[Dict[str, Any]] = Field(None, description="Configuration overrides")
    data_path: Optional[str] = Field(None, description="Path to training data")
    async_training: bool = Field(True, description="Run training asynchronously")
    notification_webhook: Optional[str] = Field(None, description="Webhook URL for completion notification")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "salary_prediction",
                "config_overrides": {
                    "model": {
                        "parameters": {
                            "n_estimators": 200
                        }
                    }
                },
                "async_training": True
            }
        }


class LoginRequest(BaseModel):
    """User login request model."""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")
    
    class Config:
        schema_extra = {
            "example": {
                "username": "admin",
                "password": "admin123"
            }
        }


# Response Models

class PredictionResponse(BaseModel):
    """Single prediction response model."""
    prediction: Union[float, int, str, List] = Field(..., description="Model prediction")
    confidence: Optional[float] = Field(None, description="Prediction confidence score")
    model_name: str = Field(..., description="Name of model used")
    model_version: Optional[str] = Field(None, description="Version of model used")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    explanation: Optional[Dict[str, Any]] = Field(None, description="Model explanation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 85000.0,
                "confidence": 0.92,
                "model_name": "salary_prediction",
                "model_version": "v1.2.0",
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_123456"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    batch_size: int = Field(..., description="Number of predictions in batch")
    processing_time: float = Field(..., description="Total processing time in seconds")
    model_name: str = Field(..., description="Name of model used")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Batch processing timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "prediction": 85000.0,
                        "confidence": 0.92,
                        "model_name": "salary_prediction",
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                ],
                "batch_size": 1,
                "processing_time": 0.15,
                "model_name": "salary_prediction",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class ModelInfo(BaseModel):
    """Model information model."""
    name: str = Field(..., description="Model name")
    version: Optional[str] = Field(None, description="Model version")
    type: str = Field(..., description="Model type")
    status: ModelStatus = Field(..., description="Model status")
    loaded_at: Optional[datetime] = Field(None, description="When model was loaded")
    path: str = Field(..., description="Model file path")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Model metadata")
    performance_metrics: Optional[Dict[str, float]] = Field(None, description="Model performance metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "salary_prediction",
                "version": "v1.2.0",
                "type": "xgboost",
                "status": "loaded",
                "loaded_at": "2024-01-15T09:00:00Z",
                "path": "models/salary_prediction/v1.2.0/",
                "performance_metrics": {
                    "r2_score": 0.89,
                    "rmse": 5420.0
                }
            }
        }


class TrainingJobResponse(BaseModel):
    """Training job response model."""
    job_id: str = Field(..., description="Training job ID")
    status: TrainingStatus = Field(..., description="Training status")
    model_name: str = Field(..., description="Model being trained")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Training start time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    progress: Optional[float] = Field(None, description="Training progress (0-1)")
    message: Optional[str] = Field(None, description="Status message")
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "train_123456",
                "status": "running",
                "model_name": "salary_prediction",
                "started_at": "2024-01-15T10:00:00Z",
                "progress": 0.45,
                "message": "Training in progress..."
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: HealthStatus = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Uptime in seconds")
    checks: Dict[str, Any] = Field(..., description="Individual health checks")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "version": "1.0.0",
                "uptime": 3600.0,
                "checks": {
                    "database": "healthy",
                    "redis": "healthy",
                    "models": "healthy"
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input data",
                "details": {
                    "field": "age",
                    "issue": "must be a positive number"
                },
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_123456"
            }
        }


class APIResponse(BaseModel):
    """Generic API response wrapper."""
    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[Any] = Field(None, description="Response data")
    error: Optional[ErrorResponse] = Field(None, description="Error information if request failed")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "data": {
                    "prediction": 85000.0,
                    "confidence": 0.92
                },
                "metadata": {
                    "processing_time": 0.15,
                    "model_version": "v1.2.0"
                }
            }
        }


# Metrics Models

class PredictionMetrics(BaseModel):
    """Prediction metrics model."""
    total_predictions: int = Field(..., description="Total number of predictions")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    average_response_time: float = Field(..., description="Average response time in seconds")
    predictions_per_minute: float = Field(..., description="Predictions per minute")
    
    class Config:
        schema_extra = {
            "example": {
                "total_predictions": 1000,
                "successful_predictions": 995,
                "failed_predictions": 5,
                "average_response_time": 0.12,
                "predictions_per_minute": 45.2
            }
        }