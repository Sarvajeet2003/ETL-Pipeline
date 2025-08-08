"""Main FastAPI application for ML model serving."""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import structlog
import uvicorn

from .config import config
from .middleware import setup_middleware
from .model_manager import model_manager
from .routers import predictions, training, health, auth

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if config.logging.format == "json" else structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.
    
    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("Starting ML Model Serving API", version=config.app.version)
    
    try:
        # Initialize model manager
        await model_manager.initialize()
        logger.info("Model manager initialized successfully")
        
        # Additional startup tasks can be added here
        # - Database connections
        # - Cache initialization
        # - External service health checks
        
        yield
        
    except Exception as e:
        logger.error("Failed to initialize application", error=str(e))
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down ML Model Serving API")
        
        # Cleanup tasks
        # - Close database connections
        # - Clear caches
        # - Save metrics


# Create FastAPI application
app = FastAPI(
    title=config.app.title,
    description=config.app.description,
    version=config.app.version,
    debug=config.app.debug,
    lifespan=lifespan,
    docs_url="/docs" if config.app.debug else None,
    redoc_url="/redoc" if config.app.debug else None,
    openapi_url="/openapi.json" if config.app.debug else None,
)

# Setup middleware
limiter = setup_middleware(app)

# Set limiter for routers that need it
if limiter:
    predictions.set_limiter(limiter)

# Add trusted host middleware for production
if config.app.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure with actual allowed hosts
    )

# Include routers
app.include_router(predictions.router)
app.include_router(training.router)
app.include_router(health.router)
app.include_router(auth.router)


# Global exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors.
    
    Args:
        request: FastAPI request object
        exc: Validation error exception
        
    Returns:
        JSON error response
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.warning("Request validation error",
                  request_id=request_id,
                  errors=exc.errors(),
                  body=exc.body)
    
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {
                "error": "ValidationError",
                "message": "Request validation failed",
                "details": exc.errors(),
                "request_id": request_id
            }
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions.
    
    Args:
        request: FastAPI request object
        exc: Exception
        
    Returns:
        JSON error response
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.error("Unhandled exception",
                request_id=request_id,
                error=str(exc),
                error_type=type(exc).__name__)
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "error": "InternalServerError",
                "message": "An internal server error occurred",
                "request_id": request_id
            }
        }
    )


# Root endpoint
@app.get(
    "/",
    summary="API Root",
    description="Get basic API information",
    tags=["root"]
)
async def root():
    """Root endpoint with API information.
    
    Returns:
        API information
    """
    return {
        "name": config.app.title,
        "version": config.app.version,
        "description": config.app.description,
        "environment": config.app.environment,
        "docs_url": "/docs" if config.app.debug else None,
        "health_check": "/health",
        "api_prefix": "/api/v1"
    }


# API information endpoint
@app.get(
    "/api/v1/info",
    summary="API Information",
    description="Get detailed API information and capabilities",
    tags=["info"]
)
async def api_info():
    """Get API information and capabilities.
    
    Returns:
        Detailed API information
    """
    return {
        "api": {
            "name": config.app.title,
            "version": config.app.version,
            "description": config.app.description,
            "environment": config.app.environment
        },
        "authentication": {
            "api_key_enabled": config.security.api_key.enabled,
            "oauth2_enabled": config.security.oauth2.enabled,
            "api_key_header": config.security.api_key.header_name if config.security.api_key.enabled else None
        },
        "features": {
            "batch_prediction": config.features.batch_prediction,
            "model_explanation": config.features.model_explanation,
            "a_b_testing": config.features.a_b_testing,
            "model_monitoring": config.features.model_monitoring,
            "training_enabled": config.training.enabled
        },
        "endpoints": {
            "predictions": "/api/v1/predict",
            "batch_predictions": "/api/v1/predict/batch",
            "models": "/api/v1/models",
            "training": "/api/v1/retrain",
            "health": "/health",
            "metrics": "/metrics"
        },
        "limits": {
            "rate_limiting_enabled": config.security.rate_limiting.enabled,
            "requests_per_minute": config.security.rate_limiting.requests_per_minute if config.security.rate_limiting.enabled else None,
            "max_batch_size": 1000  # This should come from model config
        }
    }


def create_app() -> FastAPI:
    """Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    return app


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "api_src.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
        workers=1 if config.server.reload else config.server.workers,
        log_level=config.server.log_level,
        access_log=config.server.access_log
    )