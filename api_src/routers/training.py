"""Training endpoints for ML model retraining."""

import os
import uuid
import asyncio
import subprocess
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
import structlog

from ..config import config
from ..auth import require_write, User
from ..models import (
    RetrainRequest, TrainingJobResponse, TrainingStatus,
    APIResponse, ErrorResponse
)

logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["training"])

# In-memory storage for training jobs (use database in production)
training_jobs: Dict[str, Dict[str, Any]] = {}


@router.post(
    "/retrain",
    response_model=APIResponse,
    summary="Trigger model retraining",
    description="Start a model retraining job with optional configuration overrides",
    responses={
        200: {"description": "Training job started successfully"},
        400: {"description": "Invalid training request"},
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        501: {"description": "Training not enabled"},
        500: {"description": "Internal server error"}
    }
)
async def retrain_model(
    request: Request,
    retrain_request: RetrainRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_write)
) -> APIResponse:
    """Trigger model retraining.
    
    Args:
        request: FastAPI request object
        retrain_request: Retraining request data
        background_tasks: FastAPI background tasks
        current_user: Authenticated user
        
    Returns:
        API response with training job information
    """
    # Check if training is enabled
    if not config.training.enabled:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Model training is not enabled"
        )
    
    request_id = getattr(request.state, 'request_id', 'unknown')
    job_id = f"train_{uuid.uuid4().hex[:8]}"
    model_name = retrain_request.model_name or "default"
    
    logger.info("Training request received",
               request_id=request_id,
               job_id=job_id,
               model_name=model_name,
               user=current_user.username,
               async_training=retrain_request.async_training)
    
    try:
        # Create training job record
        job_record = {
            "job_id": job_id,
            "model_name": model_name,
            "status": TrainingStatus.STARTED,
            "started_at": datetime.utcnow(),
            "started_by": current_user.username,
            "config_overrides": retrain_request.config_overrides,
            "data_path": retrain_request.data_path,
            "notification_webhook": retrain_request.notification_webhook,
            "request_id": request_id,
            "progress": 0.0,
            "message": "Training job initialized"
        }
        
        training_jobs[job_id] = job_record
        
        # Start training based on configuration
        if retrain_request.async_training and config.training.async_training.enabled:
            # Use Celery for async training
            await _start_async_training(job_id, retrain_request)
        else:
            # Use background tasks for training
            background_tasks.add_task(
                _run_training_job,
                job_id,
                retrain_request
            )
        
        # Create response
        training_response = TrainingJobResponse(
            job_id=job_id,
            status=TrainingStatus.STARTED,
            model_name=model_name,
            started_at=job_record["started_at"],
            message="Training job started successfully"
        )
        
        logger.info("Training job started",
                   request_id=request_id,
                   job_id=job_id,
                   model_name=model_name)
        
        return APIResponse(
            success=True,
            data=training_response,
            metadata={
                "job_id": job_id,
                "async_training": retrain_request.async_training
            }
        )
        
    except Exception as e:
        logger.error("Failed to start training job",
                    request_id=request_id,
                    job_id=job_id,
                    model_name=model_name,
                    error=str(e),
                    error_type=type(e).__name__)
        
        # Update job status if created
        if job_id in training_jobs:
            training_jobs[job_id]["status"] = TrainingStatus.FAILED
            training_jobs[job_id]["message"] = f"Failed to start: {str(e)}"
        
        return APIResponse(
            success=False,
            error=ErrorResponse(
                error="TrainingError",
                message=f"Failed to start training job: {str(e)}",
                request_id=request_id
            )
        )


@router.get(
    "/training/jobs/{job_id}",
    response_model=APIResponse,
    summary="Get training job status",
    description="Get the status and progress of a training job",
    responses={
        200: {"description": "Training job status retrieved successfully"},
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Training job not found"}
    }
)
async def get_training_job_status(
    job_id: str,
    request: Request,
    current_user: User = Depends(require_write)
) -> APIResponse:
    """Get training job status.
    
    Args:
        job_id: Training job ID
        request: FastAPI request object
        current_user: Authenticated user
        
    Returns:
        API response with training job status
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.info("Training job status requested",
               request_id=request_id,
               job_id=job_id,
               user=current_user.username)
    
    # Check if job exists
    if job_id not in training_jobs:
        logger.warning("Training job not found",
                      request_id=request_id,
                      job_id=job_id)
        
        return APIResponse(
            success=False,
            error=ErrorResponse(
                error="JobNotFound",
                message=f"Training job {job_id} not found",
                request_id=request_id
            )
        )
    
    try:
        job_record = training_jobs[job_id]
        
        # Create response
        training_response = TrainingJobResponse(
            job_id=job_id,
            status=job_record["status"],
            model_name=job_record["model_name"],
            started_at=job_record["started_at"],
            estimated_completion=job_record.get("estimated_completion"),
            progress=job_record.get("progress"),
            message=job_record.get("message")
        )
        
        logger.info("Training job status retrieved",
                   request_id=request_id,
                   job_id=job_id,
                   status=job_record["status"])
        
        return APIResponse(
            success=True,
            data=training_response
        )
        
    except Exception as e:
        logger.error("Failed to get training job status",
                    request_id=request_id,
                    job_id=job_id,
                    error=str(e),
                    error_type=type(e).__name__)
        
        return APIResponse(
            success=False,
            error=ErrorResponse(
                error="InternalError",
                message="Failed to retrieve training job status",
                request_id=request_id
            )
        )


@router.get(
    "/training/jobs",
    response_model=APIResponse,
    summary="List training jobs",
    description="Get a list of all training jobs",
    responses={
        200: {"description": "Training jobs listed successfully"},
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"}
    }
)
async def list_training_jobs(
    request: Request,
    current_user: User = Depends(require_write)
) -> APIResponse:
    """List all training jobs.
    
    Args:
        request: FastAPI request object
        current_user: Authenticated user
        
    Returns:
        API response with training jobs list
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.info("Training jobs list requested",
               request_id=request_id,
               user=current_user.username)
    
    try:
        # Convert job records to response format
        jobs_list = []
        for job_id, job_record in training_jobs.items():
            job_response = TrainingJobResponse(
                job_id=job_id,
                status=job_record["status"],
                model_name=job_record["model_name"],
                started_at=job_record["started_at"],
                estimated_completion=job_record.get("estimated_completion"),
                progress=job_record.get("progress"),
                message=job_record.get("message")
            )
            jobs_list.append(job_response)
        
        logger.info("Training jobs listed successfully",
                   request_id=request_id,
                   job_count=len(jobs_list))
        
        return APIResponse(
            success=True,
            data={
                "jobs": jobs_list,
                "total_jobs": len(jobs_list)
            },
            metadata={
                "job_count": len(jobs_list)
            }
        )
        
    except Exception as e:
        logger.error("Failed to list training jobs",
                    request_id=request_id,
                    error=str(e),
                    error_type=type(e).__name__)
        
        return APIResponse(
            success=False,
            error=ErrorResponse(
                error="InternalError",
                message="Failed to retrieve training jobs",
                request_id=request_id
            )
        )


async def _start_async_training(job_id: str, retrain_request: RetrainRequest) -> None:
    """Start async training using Celery.
    
    Args:
        job_id: Training job ID
        retrain_request: Retraining request data
    """
    try:
        # Import Celery here to avoid dependency issues if not configured
        from celery import Celery
        
        # Create Celery app
        celery_app = Celery(
            'ml_training',
            broker=config.training.async_training.broker_url,
            backend=config.training.async_training.result_backend
        )
        
        # Submit training task
        task = celery_app.send_task(
            'ml_training.train_model',
            args=[job_id, retrain_request.dict()],
            queue='training'
        )
        
        # Update job record with task ID
        training_jobs[job_id]["celery_task_id"] = task.id
        training_jobs[job_id]["status"] = TrainingStatus.RUNNING
        training_jobs[job_id]["message"] = "Training submitted to Celery"
        
        logger.info("Training task submitted to Celery",
                   job_id=job_id,
                   task_id=task.id)
        
    except ImportError:
        logger.error("Celery not available for async training", job_id=job_id)
        training_jobs[job_id]["status"] = TrainingStatus.FAILED
        training_jobs[job_id]["message"] = "Celery not available"
        
    except Exception as e:
        logger.error("Failed to start async training",
                    job_id=job_id,
                    error=str(e))
        training_jobs[job_id]["status"] = TrainingStatus.FAILED
        training_jobs[job_id]["message"] = f"Async training failed: {str(e)}"


async def _run_training_job(job_id: str, retrain_request: RetrainRequest) -> None:
    """Run training job in background.
    
    Args:
        job_id: Training job ID
        retrain_request: Retraining request data
    """
    try:
        # Update job status
        training_jobs[job_id]["status"] = TrainingStatus.RUNNING
        training_jobs[job_id]["message"] = "Training in progress"
        training_jobs[job_id]["progress"] = 0.1
        
        logger.info("Starting training job", job_id=job_id)
        
        # Prepare training command
        cmd = ["python", "ml_train.py"]
        
        # Add config path
        config_path = config.training.job_config.config_path
        if retrain_request.data_path:
            # Create temporary config with data path override
            config_path = await _create_temp_config(retrain_request)
        
        cmd.extend(["--config", config_path])
        
        # Add output path
        if config.training.job_config.output_path:
            cmd.extend(["--output", config.training.job_config.output_path])
        
        # Update progress
        training_jobs[job_id]["progress"] = 0.2
        training_jobs[job_id]["message"] = "Starting ML training process"
        
        # Run training process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.getcwd()
        )
        
        # Update progress
        training_jobs[job_id]["progress"] = 0.5
        training_jobs[job_id]["message"] = "Training process running"
        
        # Wait for completion
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            # Training successful
            training_jobs[job_id]["status"] = TrainingStatus.COMPLETED
            training_jobs[job_id]["progress"] = 1.0
            training_jobs[job_id]["message"] = "Training completed successfully"
            training_jobs[job_id]["completed_at"] = datetime.utcnow()
            
            logger.info("Training job completed successfully", job_id=job_id)
            
            # Send notification if webhook provided
            if retrain_request.notification_webhook:
                await _send_training_notification(job_id, retrain_request.notification_webhook)
            
        else:
            # Training failed
            error_message = stderr.decode() if stderr else "Training process failed"
            training_jobs[job_id]["status"] = TrainingStatus.FAILED
            training_jobs[job_id]["message"] = f"Training failed: {error_message}"
            training_jobs[job_id]["error"] = error_message
            
            logger.error("Training job failed",
                        job_id=job_id,
                        return_code=process.returncode,
                        error=error_message)
        
    except Exception as e:
        # Training job failed with exception
        training_jobs[job_id]["status"] = TrainingStatus.FAILED
        training_jobs[job_id]["message"] = f"Training job failed: {str(e)}"
        training_jobs[job_id]["error"] = str(e)
        
        logger.error("Training job failed with exception",
                    job_id=job_id,
                    error=str(e),
                    error_type=type(e).__name__)


async def _create_temp_config(retrain_request: RetrainRequest) -> str:
    """Create temporary config file with overrides.
    
    Args:
        retrain_request: Retraining request data
        
    Returns:
        Path to temporary config file
    """
    import yaml
    import tempfile
    
    # Load base config
    base_config_path = config.training.job_config.config_path
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Apply overrides
    if retrain_request.data_path:
        base_config['data']['source_path'] = retrain_request.data_path
    
    if retrain_request.config_overrides:
        # Deep merge config overrides
        def deep_merge(base: dict, override: dict) -> dict:
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        deep_merge(base_config, retrain_request.config_overrides)
    
    # Create temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.yaml', prefix='ml_config_')
    
    try:
        with os.fdopen(temp_fd, 'w') as f:
            yaml.dump(base_config, f, default_flow_style=False)
        
        return temp_path
        
    except Exception:
        # Clean up on error
        try:
            os.unlink(temp_path)
        except:
            pass
        raise


async def _send_training_notification(job_id: str, webhook_url: str) -> None:
    """Send training completion notification.
    
    Args:
        job_id: Training job ID
        webhook_url: Webhook URL to send notification to
    """
    try:
        import httpx
        
        job_record = training_jobs[job_id]
        
        notification_data = {
            "job_id": job_id,
            "status": job_record["status"],
            "model_name": job_record["model_name"],
            "started_at": job_record["started_at"].isoformat(),
            "completed_at": job_record.get("completed_at", datetime.utcnow()).isoformat(),
            "message": job_record.get("message", "")
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                webhook_url,
                json=notification_data,
                timeout=30.0
            )
            
            if response.status_code == 200:
                logger.info("Training notification sent successfully",
                           job_id=job_id,
                           webhook_url=webhook_url)
            else:
                logger.warning("Training notification failed",
                              job_id=job_id,
                              webhook_url=webhook_url,
                              status_code=response.status_code)
        
    except Exception as e:
        logger.error("Failed to send training notification",
                    job_id=job_id,
                    webhook_url=webhook_url,
                    error=str(e))