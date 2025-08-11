"""
Pydantic models for API request/response validation
"""
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator
import pandas as pd

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    features: Dict[str, Union[str, int, float]] = Field(
        ..., 
        description="Feature values for prediction",
        example={
            "age": 35,
            "tenure": 24,
            "monthly_charges": 65.5,
            "total_charges": 1570.0,
            "internet_service": "Fiber optic",
            "contract": "Month-to-month",
            "payment_method": "Electronic check"
        }
    )
    
    @validator('features')
    def validate_features(cls, v):
        """Validate that features is not empty"""
        if not v:
            raise ValueError("Features cannot be empty")
        return v

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: int = Field(..., description="Predicted class (0 or 1)")
    probability: float = Field(..., description="Prediction probability")
    confidence: str = Field(..., description="Confidence level (Low/Medium/High)")
    model_used: str = Field(..., description="Name of the model used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.75,
                "confidence": "High",
                "model_used": "random_forest",
                "timestamp": "2024-01-15T10:30:00"
            }
        }

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    instances: List[Dict[str, Union[str, int, float]]] = Field(
        ...,
        description="List of feature dictionaries for batch prediction",
        min_items=1,
        max_items=1000
    )
    
    @validator('instances')
    def validate_instances(cls, v):
        """Validate batch instances"""
        if not v:
            raise ValueError("Instances cannot be empty")
        if len(v) > 1000:
            raise ValueError("Maximum 1000 instances allowed per batch")
        return v

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    batch_size: int = Field(..., description="Number of instances processed")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class RetrainRequest(BaseModel):
    """Request model for model retraining"""
    data_path: Optional[str] = Field(
        None, 
        description="Optional path to new training data (uses default if not provided)"
    )
    model_types: Optional[List[str]] = Field(
        None,
        description="Optional list of model types to train (trains all if not provided)"
    )
    hyperparameters: Optional[Dict[str, Dict[str, Any]]] = Field(
        None,
        description="Optional custom hyperparameters for models"
    )

class RetrainResponse(BaseModel):
    """Response model for model retraining"""
    status: str = Field(..., description="Retraining status")
    message: str = Field(..., description="Status message")
    job_id: str = Field(..., description="Retraining job ID")
    estimated_completion_time: Optional[str] = Field(None, description="Estimated completion time")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "started",
                "message": "Model retraining initiated successfully",
                "job_id": "retrain_20240115_103000",
                "estimated_completion_time": "2024-01-15T10:45:00"
            }
        }

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    model_status: Dict[str, Any] = Field(..., description="Model loading status")
    system_info: Dict[str, Any] = Field(..., description="System information")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00",
                "version": "1.0.0",
                "model_status": {
                    "model_loaded": True,
                    "model_name": "random_forest",
                    "last_trained": "2024-01-15T09:00:00"
                },
                "system_info": {
                    "cpu_usage": 25.5,
                    "memory_usage": 45.2,
                    "disk_usage": 60.1
                }
            }
        }

class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str = Field(..., description="Name of the current model")
    model_type: str = Field(..., description="Type of the model")
    training_date: str = Field(..., description="When the model was trained")
    performance_metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "random_forest",
                "model_type": "RandomForestClassifier",
                "training_date": "2024-01-15T09:00:00",
                "performance_metrics": {
                    "accuracy": 0.85,
                    "precision": 0.83,
                    "recall": 0.87,
                    "f1": 0.85
                },
                "feature_importance": {
                    "tenure": 0.25,
                    "monthly_charges": 0.20,
                    "contract_Month-to-month": 0.18
                }
            }
        }

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input data provided",
                "timestamp": "2024-01-15T10:30:00",
                "request_id": "req_123456789"
            }
        }