"""Prediction endpoints for ML model serving."""

import time
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
import structlog

from ..config import config
from ..auth import require_read, User
from ..models import (
    PredictionRequest, BatchPredictionRequest,
    PredictionResponse, BatchPredictionResponse,
    APIResponse, ErrorResponse
)
from ..model_manager import model_manager, PredictionError
from ..middleware import PREDICTION_COUNT, PREDICTION_DURATION

logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["predictions"])

# Rate limiter (will be set by main app)
limiter: Optional[Limiter] = None


def set_limiter(app_limiter: Limiter) -> None:
    """Set the rate limiter for this router.
    
    Args:
        app_limiter: Rate limiter instance from main app
    """
    global limiter
    limiter = app_limiter


@router.post(
    "/predict",
    response_model=APIResponse,
    summary="Make a single prediction",
    description="Make a prediction using the specified or default ML model",
    responses={
        200: {"description": "Prediction successful"},
        400: {"description": "Invalid input data"},
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"}
    }
)
async def predict(
    request: Request,
    prediction_request: PredictionRequest,
    current_user: User = Depends(require_read)
) -> APIResponse:
    """Make a single prediction.
    
    Args:
        request: FastAPI request object
        prediction_request: Prediction request data
        current_user: Authenticated user
        
    Returns:
        API response with prediction
    """
    # Apply rate limiting if enabled
    if limiter and config.security.rate_limiting.enabled:
        await limiter.limit(f"{config.security.rate_limiting.requests_per_minute}/minute")(request)
    
    request_id = getattr(request.state, 'request_id', 'unknown')
    model_name = prediction_request.model_name or "default"
    
    logger.info("Prediction request received",
               request_id=request_id,
               model_name=model_name,
               user=current_user.username,
               include_confidence=prediction_request.include_confidence)
    
    try:
        # Start timing
        start_time = time.time()
        
        # Make prediction
        prediction_response = await model_manager.predict(
            data=prediction_request.data,
            model_name=prediction_request.model_name,
            include_confidence=prediction_request.include_confidence
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Update metrics
        PREDICTION_COUNT.labels(model_name=model_name, status="success").inc()
        PREDICTION_DURATION.labels(model_name=model_name).observe(processing_time)
        
        # Add request ID to response
        prediction_response.request_id = request_id
        
        logger.info("Prediction completed successfully",
                   request_id=request_id,
                   model_name=model_name,
                   processing_time=processing_time)
        
        return APIResponse(
            success=True,
            data=prediction_response,
            metadata={
                "processing_time": processing_time,
                "model_name": model_name
            }
        )
        
    except PredictionError as e:
        # Update error metrics
        PREDICTION_COUNT.labels(model_name=model_name, status="error").inc()
        
        logger.error("Prediction failed",
                    request_id=request_id,
                    model_name=model_name,
                    error=str(e))
        
        return APIResponse(
            success=False,
            error=ErrorResponse(
                error="PredictionError",
                message=str(e),
                request_id=request_id
            )
        )
        
    except Exception as e:
        # Update error metrics
        PREDICTION_COUNT.labels(model_name=model_name, status="error").inc()
        
        logger.error("Unexpected error in prediction",
                    request_id=request_id,
                    model_name=model_name,
                    error=str(e),
                    error_type=type(e).__name__)
        
        return APIResponse(
            success=False,
            error=ErrorResponse(
                error="InternalError",
                message="An unexpected error occurred during prediction",
                request_id=request_id
            )
        )


@router.post(
    "/predict/batch",
    response_model=APIResponse,
    summary="Make batch predictions",
    description="Make predictions for multiple inputs using the specified or default ML model",
    responses={
        200: {"description": "Batch prediction successful"},
        400: {"description": "Invalid input data"},
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        413: {"description": "Batch size too large"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"}
    }
)
async def predict_batch(
    request: Request,
    batch_request: BatchPredictionRequest,
    current_user: User = Depends(require_read)
) -> APIResponse:
    """Make batch predictions.
    
    Args:
        request: FastAPI request object
        batch_request: Batch prediction request data
        current_user: Authenticated user
        
    Returns:
        API response with batch predictions
    """
    # Check if batch predictions are enabled
    if not config.features.batch_prediction:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Batch predictions are not enabled"
        )
    
    # Apply rate limiting if enabled
    if limiter and config.security.rate_limiting.enabled:
        await limiter.limit(f"{config.security.rate_limiting.requests_per_minute}/minute")(request)
    
    request_id = getattr(request.state, 'request_id', 'unknown')
    model_name = batch_request.model_name or "default"
    batch_size = len(batch_request.data)
    
    # Check batch size limits
    max_batch_size = 1000  # This should come from model config
    if batch_size > max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Batch size {batch_size} exceeds maximum allowed size {max_batch_size}"
        )
    
    logger.info("Batch prediction request received",
               request_id=request_id,
               model_name=model_name,
               batch_size=batch_size,
               user=current_user.username)
    
    try:
        # Start timing
        start_time = time.time()
        
        # Make batch predictions
        prediction_responses = await model_manager.predict(
            data=batch_request.data,
            model_name=batch_request.model_name,
            include_confidence=batch_request.include_confidence
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Update metrics
        PREDICTION_COUNT.labels(model_name=model_name, status="success").inc(batch_size)
        PREDICTION_DURATION.labels(model_name=model_name).observe(processing_time)
        
        # Add request IDs to responses
        for response in prediction_responses:
            response.request_id = request_id
        
        # Create batch response
        batch_response = BatchPredictionResponse(
            predictions=prediction_responses,
            batch_size=batch_size,
            processing_time=processing_time,
            model_name=model_name,
            request_id=request_id
        )
        
        logger.info("Batch prediction completed successfully",
                   request_id=request_id,
                   model_name=model_name,
                   batch_size=batch_size,
                   processing_time=processing_time,
                   avg_time_per_prediction=processing_time/batch_size)
        
        return APIResponse(
            success=True,
            data=batch_response,
            metadata={
                "processing_time": processing_time,
                "batch_size": batch_size,
                "avg_time_per_prediction": processing_time / batch_size
            }
        )
        
    except PredictionError as e:
        # Update error metrics
        PREDICTION_COUNT.labels(model_name=model_name, status="error").inc(batch_size)
        
        logger.error("Batch prediction failed",
                    request_id=request_id,
                    model_name=model_name,
                    batch_size=batch_size,
                    error=str(e))
        
        return APIResponse(
            success=False,
            error=ErrorResponse(
                error="PredictionError",
                message=str(e),
                request_id=request_id
            )
        )
        
    except Exception as e:
        # Update error metrics
        PREDICTION_COUNT.labels(model_name=model_name, status="error").inc(batch_size)
        
        logger.error("Unexpected error in batch prediction",
                    request_id=request_id,
                    model_name=model_name,
                    batch_size=batch_size,
                    error=str(e),
                    error_type=type(e).__name__)
        
        return APIResponse(
            success=False,
            error=ErrorResponse(
                error="InternalError",
                message="An unexpected error occurred during batch prediction",
                request_id=request_id
            )
        )


@router.get(
    "/models",
    response_model=APIResponse,
    summary="List available models",
    description="Get information about all loaded ML models",
    responses={
        200: {"description": "Models listed successfully"},
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"}
    }
)
async def list_models(
    request: Request,
    current_user: User = Depends(require_read)
) -> APIResponse:
    """List all available models.
    
    Args:
        request: FastAPI request object
        current_user: Authenticated user
        
    Returns:
        API response with model information
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.info("Models list requested",
               request_id=request_id,
               user=current_user.username)
    
    try:
        # Get model information
        models_info = model_manager.get_model_info()
        model_names = model_manager.list_models()
        
        logger.info("Models listed successfully",
                   request_id=request_id,
                   model_count=len(model_names))
        
        return APIResponse(
            success=True,
            data={
                "models": models_info,
                "model_names": model_names,
                "default_model": model_manager.default_model
            },
            metadata={
                "model_count": len(model_names)
            }
        )
        
    except Exception as e:
        logger.error("Failed to list models",
                    request_id=request_id,
                    error=str(e),
                    error_type=type(e).__name__)
        
        return APIResponse(
            success=False,
            error=ErrorResponse(
                error="InternalError",
                message="Failed to retrieve model information",
                request_id=request_id
            )
        )


@router.get(
    "/models/{model_name}",
    response_model=APIResponse,
    summary="Get model information",
    description="Get detailed information about a specific ML model",
    responses={
        200: {"description": "Model information retrieved successfully"},
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Model not found"}
    }
)
async def get_model_info(
    model_name: str,
    request: Request,
    current_user: User = Depends(require_read)
) -> APIResponse:
    """Get information about a specific model.
    
    Args:
        model_name: Name of the model
        request: FastAPI request object
        current_user: Authenticated user
        
    Returns:
        API response with model information
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.info("Model info requested",
               request_id=request_id,
               model_name=model_name,
               user=current_user.username)
    
    try:
        # Get model information
        model_info = model_manager.get_model_info(model_name)
        
        logger.info("Model info retrieved successfully",
                   request_id=request_id,
                   model_name=model_name)
        
        return APIResponse(
            success=True,
            data=model_info
        )
        
    except ValueError as e:
        logger.warning("Model not found",
                      request_id=request_id,
                      model_name=model_name)
        
        return APIResponse(
            success=False,
            error=ErrorResponse(
                error="ModelNotFound",
                message=str(e),
                request_id=request_id
            )
        )
        
    except Exception as e:
        logger.error("Failed to get model info",
                    request_id=request_id,
                    model_name=model_name,
                    error=str(e),
                    error_type=type(e).__name__)
        
        return APIResponse(
            success=False,
            error=ErrorResponse(
                error="InternalError",
                message="Failed to retrieve model information",
                request_id=request_id
            )
        )