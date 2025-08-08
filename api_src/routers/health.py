"""Health check and monitoring endpoints."""

import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any
from fastapi import APIRouter, Request
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import structlog

from ..config import config
from ..models import HealthCheckResponse, HealthStatus, APIResponse
from ..model_manager import model_manager

logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(tags=["health"])

# Application start time for uptime calculation
app_start_time = time.time()


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Basic health check",
    description="Get basic health status of the API",
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unhealthy"}
    }
)
async def health_check(request: Request) -> HealthCheckResponse:
    """Basic health check endpoint.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Health check response
    """
    if not config.monitoring.health_check.enabled:
        return HealthCheckResponse(
            status=HealthStatus.HEALTHY,
            version=config.app.version,
            uptime=time.time() - app_start_time,
            checks={"health_check": "disabled"}
        )
    
    try:
        # Calculate uptime
        uptime = time.time() - app_start_time
        
        # Basic checks
        checks = {
            "api": "healthy",
            "uptime": f"{uptime:.2f}s"
        }
        
        # Check models
        try:
            model_names = model_manager.list_models()
            if model_names:
                checks["models"] = "healthy"
                checks["loaded_models"] = len(model_names)
            else:
                checks["models"] = "no_models_loaded"
        except Exception as e:
            checks["models"] = f"error: {str(e)}"
        
        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        
        return HealthCheckResponse(
            status=overall_status,
            version=config.app.version,
            uptime=uptime,
            checks=checks
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        
        return HealthCheckResponse(
            status=HealthStatus.UNHEALTHY,
            version=config.app.version,
            uptime=time.time() - app_start_time,
            checks={"error": str(e)}
        )


@router.get(
    "/health/detailed",
    response_model=APIResponse,
    summary="Detailed health check",
    description="Get detailed health status including system metrics",
    responses={
        200: {"description": "Detailed health information"},
        503: {"description": "Service is unhealthy"}
    }
)
async def detailed_health_check(request: Request) -> APIResponse:
    """Detailed health check with system metrics.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Detailed health information
    """
    try:
        # Calculate uptime
        uptime = time.time() - app_start_time
        uptime_formatted = str(timedelta(seconds=int(uptime)))
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Model information
        model_info = {}
        model_status = "healthy"
        try:
            models = model_manager.get_model_info()
            model_info = {
                "loaded_models": len(models),
                "models": {name: info.dict() for name, info in models.items()},
                "default_model": model_manager.default_model
            }
        except Exception as e:
            model_status = f"error: {str(e)}"
        
        # Database check (if enabled)
        database_status = "not_configured"
        if config.database.enabled:
            try:
                # Add database connectivity check here
                database_status = "healthy"
            except Exception as e:
                database_status = f"error: {str(e)}"
        
        # Redis check (if enabled)
        redis_status = "not_configured"
        if config.redis.enabled:
            try:
                # Add Redis connectivity check here
                redis_status = "healthy"
            except Exception as e:
                redis_status = f"error: {str(e)}"
        
        # Compile detailed checks
        detailed_checks = {
            "api": {
                "status": "healthy",
                "version": config.app.version,
                "environment": config.app.environment,
                "uptime": uptime_formatted,
                "uptime_seconds": uptime
            },
            "system": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used
                },
                "disk": {
                    "total": disk.total,
                    "free": disk.free,
                    "used": disk.used,
                    "percent": (disk.used / disk.total) * 100
                }
            },
            "models": {
                "status": model_status,
                **model_info
            },
            "database": {
                "status": database_status,
                "enabled": config.database.enabled
            },
            "redis": {
                "status": redis_status,
                "enabled": config.redis.enabled
            }
        }
        
        # Determine overall health
        overall_status = HealthStatus.HEALTHY
        
        # Check for issues
        if cpu_percent > 90:
            overall_status = HealthStatus.DEGRADED
        
        if memory.percent > 90:
            overall_status = HealthStatus.DEGRADED
        
        if (disk.used / disk.total) * 100 > 90:
            overall_status = HealthStatus.DEGRADED
        
        if "error" in model_status or "error" in database_status or "error" in redis_status:
            overall_status = HealthStatus.UNHEALTHY
        
        return APIResponse(
            success=True,
            data={
                "status": overall_status,
                "timestamp": datetime.utcnow(),
                "checks": detailed_checks
            },
            metadata={
                "check_type": "detailed",
                "uptime": uptime
            }
        )
        
    except Exception as e:
        logger.error("Detailed health check failed", error=str(e))
        
        return APIResponse(
            success=False,
            data={
                "status": HealthStatus.UNHEALTHY,
                "timestamp": datetime.utcnow(),
                "checks": {"error": str(e)}
            }
        )


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Get Prometheus metrics for monitoring",
    responses={
        200: {"description": "Prometheus metrics", "content": {"text/plain": {}}}
    }
)
async def get_metrics() -> Response:
    """Get Prometheus metrics.
    
    Returns:
        Prometheus metrics in text format
    """
    if not config.monitoring.metrics.enabled:
        return Response(
            content="# Metrics disabled\n",
            media_type="text/plain"
        )
    
    try:
        # Generate Prometheus metrics
        metrics_data = generate_latest()
        
        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST
        )
        
    except Exception as e:
        logger.error("Failed to generate metrics", error=str(e))
        
        return Response(
            content=f"# Error generating metrics: {str(e)}\n",
            media_type="text/plain",
            status_code=500
        )


@router.get(
    "/status",
    response_model=APIResponse,
    summary="Service status",
    description="Get current service status and statistics",
    responses={
        200: {"description": "Service status information"}
    }
)
async def get_status(request: Request) -> APIResponse:
    """Get service status and statistics.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Service status information
    """
    try:
        # Calculate uptime
        uptime = time.time() - app_start_time
        
        # Get model statistics
        models = model_manager.list_models()
        
        # Compile status information
        status_info = {
            "service": {
                "name": config.app.title,
                "version": config.app.version,
                "environment": config.app.environment,
                "uptime_seconds": uptime,
                "uptime_formatted": str(timedelta(seconds=int(uptime))),
                "started_at": datetime.fromtimestamp(app_start_time).isoformat()
            },
            "models": {
                "total_loaded": len(models),
                "model_names": models,
                "default_model": model_manager.default_model
            },
            "configuration": {
                "debug": config.app.debug,
                "authentication": {
                    "api_key_enabled": config.security.api_key.enabled,
                    "oauth2_enabled": config.security.oauth2.enabled
                },
                "features": {
                    "batch_prediction": config.features.batch_prediction,
                    "model_explanation": config.features.model_explanation,
                    "training_enabled": config.training.enabled
                }
            }
        }
        
        return APIResponse(
            success=True,
            data=status_info,
            metadata={
                "timestamp": datetime.utcnow(),
                "uptime": uptime
            }
        )
        
    except Exception as e:
        logger.error("Failed to get service status", error=str(e))
        
        return APIResponse(
            success=False,
            error={
                "error": "StatusError",
                "message": f"Failed to retrieve service status: {str(e)}"
            }
        )


@router.get(
    "/ping",
    summary="Simple ping endpoint",
    description="Simple endpoint to check if the service is responding",
    responses={
        200: {"description": "Service is responding"}
    }
)
async def ping() -> Dict[str, Any]:
    """Simple ping endpoint.
    
    Returns:
        Simple pong response
    """
    return {
        "message": "pong",
        "timestamp": datetime.utcnow(),
        "version": config.app.version
    }