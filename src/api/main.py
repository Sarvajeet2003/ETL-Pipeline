"""
FastAPI main application
"""
import time
import uuid
import psutil
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
import asyncio
import subprocess

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from src.utils.config import config_manager
from src.utils.logging import setup_logger, get_logger
from src.utils.exceptions import MLException, ModelLoadException
from src.api.auth import get_current_user, User
from src.api.models import (
    PredictionRequest, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse,
    RetrainRequest, RetrainResponse, HealthResponse, ModelInfo, ErrorResponse
)
from src.api.predictor import predictor

# Load configuration
config = config_manager.get_api_config()['api']

# Setup logging
setup_logger(
    log_file=config['logging']['log_file'],
    level=config['logging']['level'],
    component="API"
)
logger = get_logger()

# Create FastAPI app
app = FastAPI(
    title="Enterprise Data Science Pipeline API",
    description="REST API for ML model predictions and management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['cors']['allow_origins'],
    allow_credentials=config['cors']['allow_credentials'],
    allow_methods=config['cors']['allow_methods'],
    allow_headers=config['cors']['allow_headers'],
)

# Global variables for background tasks
retraining_jobs = {}

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    logger.warning(f"Validation error for {request.url}: {exc}")
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="ValidationError",
            message=f"Invalid input data: {str(exc)}",
            timestamp=datetime.now().isoformat(),
            request_id=str(uuid.uuid4())
        ).dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP error {exc.status_code} for {request.url}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPException",
            message=exc.detail,
            timestamp=datetime.now().isoformat(),
            request_id=str(uuid.uuid4())
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unexpected error for {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            timestamp=datetime.now().isoformat(),
            request_id=str(uuid.uuid4())
        ).dict()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns system status and model information
    """
    try:
        # Get system information
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get model status
        model_info = predictor.get_model_info()
        
        health_data = HealthResponse(
            status="healthy" if predictor.is_model_loaded() else "degraded",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            model_status={
                "model_loaded": predictor.is_model_loaded(),
                "model_name": model_info.get('model_name', 'unknown'),
                "last_loaded": model_info.get('last_loaded')
            },
            system_info={
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "disk_usage": disk.percent
            }
        )
        
        logger.info("Health check completed successfully")
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """
    Get detailed information about the current model
    """
    try:
        if not predictor.is_model_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        model_info = predictor.get_model_info()
        
        # Load performance metrics if available
        metrics_path = Path("logs/ml_metrics.json")
        performance_metrics = {}
        if metrics_path.exists():
            import json
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
                performance_metrics = metrics_data.get('test_metrics', {})
        
        response = ModelInfo(
            model_name=model_info.get('model_name', 'unknown'),
            model_type=model_info.get('model_type', 'unknown'),
            training_date=model_info.get('last_loaded', datetime.now().isoformat()),
            performance_metrics=performance_metrics,
            feature_importance=model_info.get('feature_importance')
        )
        
        logger.info(f"Model info requested")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get model information")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a single prediction
    """
    try:
        if not predictor.is_model_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        start_time = time.time()
        result = predictor.predict_single(request.features)
        processing_time = (time.time() - start_time) * 1000
        
        response = PredictionResponse(**result)
        
        logger.info(f"Prediction made in {processing_time:.2f}ms")
        return response
        
    except MLException as e:
        logger.error(f"ML error during prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions
    """
    try:
        if not predictor.is_model_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        start_time = time.time()
        results = predictor.predict_batch(request.instances)
        processing_time = (time.time() - start_time) * 1000
        
        predictions = [PredictionResponse(**result) for result in results]
        
        response = BatchPredictionResponse(
            predictions=predictions,
            batch_size=len(request.instances),
            processing_time_ms=processing_time
        )
        
        logger.info(f"Batch prediction completed: {len(request.instances)} instances in {processing_time:.2f}ms")
        return response
        
    except MLException as e:
        logger.error(f"ML error during batch prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")

async def run_retraining_task(job_id: str, data_path: str = None):
    """Background task for model retraining"""
    try:
        logger.info(f"Starting retraining job {job_id}")
        retraining_jobs[job_id] = {"status": "running", "start_time": datetime.now()}
        
        # Run ETL pipeline
        logger.info("Running ETL pipeline...")
        etl_result = subprocess.run(
            ["python", "-m", "src.etl.pipeline"],
            capture_output=True,
            text=True,
            cwd="."
        )
        
        if etl_result.returncode != 0:
            raise Exception(f"ETL pipeline failed: {etl_result.stderr}")
        
        # Run ML pipeline
        logger.info("Running ML pipeline...")
        ml_result = subprocess.run(
            ["python", "-m", "src.ml.pipeline"],
            capture_output=True,
            text=True,
            cwd="."
        )
        
        if ml_result.returncode != 0:
            raise Exception(f"ML pipeline failed: {ml_result.stderr}")
        
        # Reload the model
        logger.info("Reloading model...")
        if predictor.reload_model():
            retraining_jobs[job_id] = {
                "status": "completed",
                "start_time": retraining_jobs[job_id]["start_time"],
                "end_time": datetime.now(),
                "message": "Model retraining completed successfully"
            }
            logger.info(f"Retraining job {job_id} completed successfully")
        else:
            raise Exception("Failed to reload model after retraining")
        
    except Exception as e:
        logger.error(f"Retraining job {job_id} failed: {str(e)}")
        retraining_jobs[job_id] = {
            "status": "failed",
            "start_time": retraining_jobs[job_id]["start_time"],
            "end_time": datetime.now(),
            "error": str(e)
        }

@app.post("/retrain", response_model=RetrainResponse)
async def retrain_model(
    request: RetrainRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger model retraining
    """
    try:
        # Generate job ID
        job_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Add background task
        background_tasks.add_task(
            run_retraining_task,
            job_id,
            request.data_path
        )
        
        response = RetrainResponse(
            status="started",
            message="Model retraining initiated successfully",
            job_id=job_id,
            estimated_completion_time=(datetime.now().replace(minute=datetime.now().minute + 15)).isoformat()
        )
        
        logger.info(f"Retraining job {job_id} started")
        return response
        
    except Exception as e:
        logger.error(f"Error starting retraining: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start retraining")

@app.get("/retrain/status/{job_id}")
async def get_retrain_status(job_id: str):
    """
    Get status of a retraining job
    """
    if job_id not in retraining_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = retraining_jobs[job_id]
    logger.info(f"Retraining status requested for job {job_id}")
    
    return {
        "job_id": job_id,
        "status": job_info["status"],
        "start_time": job_info["start_time"].isoformat(),
        "end_time": job_info.get("end_time", {}).isoformat() if job_info.get("end_time") else None,
        "message": job_info.get("message"),
        "error": job_info.get("error")
    }

@app.post("/model/reload")
async def reload_model():
    """
    Manually reload the model
    """
    try:
        success = predictor.reload_model()
        if success:
            logger.info(f"Model reloaded")
            return {"status": "success", "message": "Model reloaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reload model")
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to reload model")

if __name__ == "__main__":
    import uvicorn
    
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        app,
        host=config['server']['host'],
        port=config['server']['port'],
        log_level=config['logging']['level'].lower()
    )