"""Middleware for request processing, logging, and monitoring."""

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import structlog
from prometheus_client import Counter, Histogram, Gauge
import json

from .config import config

logger = structlog.get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_REQUESTS = Gauge(
    'http_requests_active',
    'Number of active HTTP requests'
)

PREDICTION_COUNT = Counter(
    'ml_predictions_total',
    'Total ML predictions',
    ['model_name', 'status']
)

PREDICTION_DURATION = Histogram(
    'ml_prediction_duration_seconds',
    'ML prediction duration in seconds',
    ['model_name']
)

MODEL_LOAD_COUNT = Counter(
    'ml_model_loads_total',
    'Total model loads',
    ['model_name', 'status']
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware/endpoint
            
        Returns:
            Response object
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        if config.logging.request_logging.enabled:
            request_data = {
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": get_remote_address(request),
                "user_agent": request.headers.get("user-agent", ""),
            }
            
            # Include headers if configured
            if config.logging.request_logging.include_headers:
                request_data["headers"] = dict(request.headers)
            
            # Include body for POST/PUT requests if configured
            if config.logging.request_logging.include_body and request.method in ["POST", "PUT"]:
                try:
                    body = await request.body()
                    if body:
                        request_data["body"] = body.decode("utf-8")[:1000]  # Limit body size
                except Exception:
                    pass  # Skip body logging if it fails
            
            logger.info("Request started", **request_data)
        
        # Process request
        ACTIVE_REQUESTS.inc()
        try:
            response = await call_next(request)
        except Exception as e:
            # Log error
            logger.error("Request failed", 
                        request_id=request_id,
                        error=str(e),
                        error_type=type(e).__name__)
            raise
        finally:
            ACTIVE_REQUESTS.dec()
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Add response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log response
        if config.logging.request_logging.enabled:
            response_data = {
                "request_id": request_id,
                "status_code": response.status_code,
                "process_time": process_time,
            }
            
            # Log slow requests
            if (config.logging.performance_logging.enabled and 
                process_time > config.logging.performance_logging.slow_request_threshold):
                logger.warning("Slow request detected", **response_data)
            else:
                logger.info("Request completed", **response_data)
        
        # Update metrics
        endpoint = request.url.path
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=endpoint
        ).observe(process_time)
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware/endpoint
            
        Returns:
            Response with security headers
        """
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Add CSP header for non-API endpoints
        if not request.url.path.startswith("/api/"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "font-src 'self'"
            )
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for global error handling."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle errors globally.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware/endpoint
            
        Returns:
            Response or error response
        """
        try:
            return await call_next(request)
        except Exception as e:
            # Log the error
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            logger.error("Unhandled error in request",
                        request_id=request_id,
                        error=str(e),
                        error_type=type(e).__name__,
                        method=request.method,
                        url=str(request.url))
            
            # Return generic error response
            from fastapi import HTTPException
            from fastapi.responses import JSONResponse
            
            if isinstance(e, HTTPException):
                return JSONResponse(
                    status_code=e.status_code,
                    content={
                        "error": "HTTPException",
                        "message": e.detail,
                        "request_id": request_id
                    }
                )
            else:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "InternalServerError",
                        "message": "An internal server error occurred",
                        "request_id": request_id
                    }
                )


def setup_cors_middleware(app):
    """Setup CORS middleware.
    
    Args:
        app: FastAPI application instance
    """
    if config.security.cors.enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.security.cors.allow_origins,
            allow_credentials=config.security.cors.allow_credentials,
            allow_methods=config.security.cors.allow_methods,
            allow_headers=config.security.cors.allow_headers,
        )
        logger.info("CORS middleware enabled", 
                   allow_origins=config.security.cors.allow_origins)


def setup_rate_limiting(app):
    """Setup rate limiting middleware.
    
    Args:
        app: FastAPI application instance
    """
    if config.security.rate_limiting.enabled:
        # Create limiter
        limiter = Limiter(key_func=get_remote_address)
        
        # Add middleware
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        app.add_middleware(SlowAPIMiddleware)
        
        logger.info("Rate limiting enabled",
                   requests_per_minute=config.security.rate_limiting.requests_per_minute)
        
        return limiter
    
    return None


def setup_middleware(app):
    """Setup all middleware.
    
    Args:
        app: FastAPI application instance
        
    Returns:
        Rate limiter instance if enabled
    """
    # Add custom middleware (order matters - last added is executed first)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    
    # Setup CORS
    setup_cors_middleware(app)
    
    # Setup rate limiting
    limiter = setup_rate_limiting(app)
    
    logger.info("Middleware setup completed")
    
    return limiter