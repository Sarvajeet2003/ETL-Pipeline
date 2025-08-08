"""Model management and serving logic."""

import os
import time
import asyncio
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import structlog

from .config import config
from .models import ModelInfo, ModelStatus, PredictionResponse

logger = structlog.get_logger(__name__)


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""
    pass


class PredictionError(Exception):
    """Exception raised when prediction fails."""
    pass


class BaseModelWrapper:
    """Base class for model wrappers."""
    
    def __init__(self, name: str, path: str, model_type: str):
        """Initialize model wrapper.
        
        Args:
            name: Model name
            path: Model path
            model_type: Model type
        """
        self.name = name
        self.path = path
        self.model_type = model_type
        self.model = None
        self.metadata = {}
        self.loaded_at = None
        self.version = None
        
    async def load(self) -> None:
        """Load the model asynchronously."""
        raise NotImplementedError
    
    async def predict(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Any, List[Any]]:
        """Make predictions asynchronously."""
        raise NotImplementedError
    
    def get_info(self) -> ModelInfo:
        """Get model information."""
        return ModelInfo(
            name=self.name,
            version=self.version,
            type=self.model_type,
            status=ModelStatus.LOADED if self.model else ModelStatus.NOT_FOUND,
            loaded_at=self.loaded_at,
            path=self.path,
            metadata=self.metadata,
            performance_metrics=self.metadata.get('metrics', {})
        )


class MLPipelineWrapper(BaseModelWrapper):
    """Wrapper for ML pipeline models."""
    
    async def load(self) -> None:
        """Load ML pipeline model."""
        try:
            model_path = Path(self.path)
            
            if not model_path.exists():
                raise ModelLoadError(f"Model path does not exist: {model_path}")
            
            # Load model using joblib (assuming it's saved with joblib)
            model_file = model_path / "model.joblib"
            if not model_file.exists():
                # Try alternative file names
                for filename in ["model.pkl", "model.pickle"]:
                    alt_file = model_path / filename
                    if alt_file.exists():
                        model_file = alt_file
                        break
                else:
                    raise ModelLoadError(f"No model file found in {model_path}")
            
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(None, joblib.load, str(model_file))
            
            # Load metadata if available
            metadata_file = model_path / "metadata.json"
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                    self.version = self.metadata.get('version')
            
            self.loaded_at = datetime.utcnow()
            
            logger.info("Model loaded successfully", 
                       model_name=self.name, 
                       model_path=str(model_path),
                       model_type=self.model_type)
            
        except Exception as e:
            logger.error("Failed to load model", 
                        model_name=self.name, 
                        error=str(e))
            raise ModelLoadError(f"Failed to load model {self.name}: {str(e)}")
    
    async def predict(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Any, List[Any]]:
        """Make predictions using the loaded model."""
        if not self.model:
            raise PredictionError("Model not loaded")
        
        try:
            # Convert input data to DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame([data])
                single_prediction = True
            else:
                df = pd.DataFrame(data)
                single_prediction = False
            
            # Make predictions in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            predictions = await loop.run_in_executor(None, self.model.predict, df)
            
            # Convert numpy arrays to Python types
            if isinstance(predictions, np.ndarray):
                predictions = predictions.tolist()
            
            # Return single prediction or list based on input
            if single_prediction:
                return predictions[0] if isinstance(predictions, list) else predictions
            else:
                return predictions
                
        except Exception as e:
            logger.error("Prediction failed", 
                        model_name=self.name, 
                        error=str(e))
            raise PredictionError(f"Prediction failed: {str(e)}")
    
    async def predict_proba(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Optional[Union[Any, List[Any]]]:
        """Get prediction probabilities if supported."""
        if not self.model or not hasattr(self.model, 'predict_proba'):
            return None
        
        try:
            # Convert input data to DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame([data])
                single_prediction = True
            else:
                df = pd.DataFrame(data)
                single_prediction = False
            
            # Make predictions in thread pool
            loop = asyncio.get_event_loop()
            probabilities = await loop.run_in_executor(None, self.model.predict_proba, df)
            
            # Convert numpy arrays to Python types
            if isinstance(probabilities, np.ndarray):
                probabilities = probabilities.tolist()
            
            # Return single prediction or list based on input
            if single_prediction:
                return probabilities[0] if isinstance(probabilities, list) else probabilities
            else:
                return probabilities
                
        except Exception as e:
            logger.warning("Probability prediction failed", 
                          model_name=self.name, 
                          error=str(e))
            return None


class SklearnWrapper(BaseModelWrapper):
    """Wrapper for scikit-learn models."""
    
    async def load(self) -> None:
        """Load scikit-learn model."""
        try:
            model_path = Path(self.path)
            
            # Load model
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(None, joblib.load, str(model_path))
            
            self.loaded_at = datetime.utcnow()
            
            logger.info("Scikit-learn model loaded", model_name=self.name)
            
        except Exception as e:
            logger.error("Failed to load scikit-learn model", 
                        model_name=self.name, 
                        error=str(e))
            raise ModelLoadError(f"Failed to load model {self.name}: {str(e)}")
    
    async def predict(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Any, List[Any]]:
        """Make predictions using scikit-learn model."""
        if not self.model:
            raise PredictionError("Model not loaded")
        
        try:
            # Convert to appropriate format for sklearn
            if isinstance(data, dict):
                # Convert dict to array (assumes features are in correct order)
                X = np.array(list(data.values())).reshape(1, -1)
                single_prediction = True
            else:
                # Convert list of dicts to 2D array
                X = np.array([list(item.values()) for item in data])
                single_prediction = False
            
            # Make predictions
            loop = asyncio.get_event_loop()
            predictions = await loop.run_in_executor(None, self.model.predict, X)
            
            # Convert to Python types
            if isinstance(predictions, np.ndarray):
                predictions = predictions.tolist()
            
            if single_prediction:
                return predictions[0] if isinstance(predictions, list) else predictions
            else:
                return predictions
                
        except Exception as e:
            logger.error("Scikit-learn prediction failed", 
                        model_name=self.name, 
                        error=str(e))
            raise PredictionError(f"Prediction failed: {str(e)}")


class ModelManager:
    """Manages multiple ML models."""
    
    def __init__(self):
        """Initialize model manager."""
        self.models: Dict[str, BaseModelWrapper] = {}
        self.default_model: Optional[str] = None
        
    async def initialize(self) -> None:
        """Initialize all configured models."""
        logger.info("Initializing model manager")
        
        for model_name, model_config in config.models.items():
            try:
                await self.load_model(
                    name=model_config.name,
                    path=model_config.path,
                    model_type=model_config.type,
                    warm_up=model_config.warm_up
                )
                
                # Set first model as default if none specified
                if self.default_model is None:
                    self.default_model = model_config.name
                    
            except Exception as e:
                logger.error("Failed to initialize model", 
                           model_name=model_name, 
                           error=str(e))
        
        logger.info("Model manager initialized", 
                   loaded_models=list(self.models.keys()),
                   default_model=self.default_model)
    
    async def load_model(self, name: str, path: str, model_type: str, warm_up: bool = True) -> None:
        """Load a model.
        
        Args:
            name: Model name
            path: Model path
            model_type: Model type
            warm_up: Whether to warm up the model
        """
        logger.info("Loading model", model_name=name, model_type=model_type)
        
        # Create appropriate wrapper based on model type
        if model_type == "ml_pipeline":
            wrapper = MLPipelineWrapper(name, path, model_type)
        elif model_type == "sklearn":
            wrapper = SklearnWrapper(name, path, model_type)
        else:
            # Default to ML pipeline wrapper
            wrapper = MLPipelineWrapper(name, path, model_type)
        
        # Load the model
        await wrapper.load()
        
        # Warm up the model if requested
        if warm_up:
            await self._warm_up_model(wrapper)
        
        # Store the model
        self.models[name] = wrapper
        
        logger.info("Model loaded and ready", model_name=name)
    
    async def _warm_up_model(self, wrapper: BaseModelWrapper) -> None:
        """Warm up a model with dummy data.
        
        Args:
            wrapper: Model wrapper to warm up
        """
        try:
            # Create dummy data for warm-up
            dummy_data = {"feature_1": 1.0, "feature_2": "test"}
            
            # Make a dummy prediction to warm up the model
            await wrapper.predict(dummy_data)
            
            logger.info("Model warmed up successfully", model_name=wrapper.name)
            
        except Exception as e:
            logger.warning("Model warm-up failed", 
                          model_name=wrapper.name, 
                          error=str(e))
    
    async def predict(self, 
                     data: Union[Dict[str, Any], List[Dict[str, Any]]], 
                     model_name: Optional[str] = None,
                     include_confidence: bool = False) -> Union[PredictionResponse, List[PredictionResponse]]:
        """Make predictions using specified or default model.
        
        Args:
            data: Input data for prediction
            model_name: Name of model to use (uses default if None)
            include_confidence: Whether to include confidence scores
            
        Returns:
            Prediction response(s)
        """
        # Determine which model to use
        if model_name is None:
            model_name = self.default_model
        
        if model_name is None:
            raise PredictionError("No model specified and no default model available")
        
        if model_name not in self.models:
            raise PredictionError(f"Model '{model_name}' not found")
        
        wrapper = self.models[model_name]
        
        # Make predictions
        start_time = time.time()
        predictions = await wrapper.predict(data)
        prediction_time = time.time() - start_time
        
        # Get confidence scores if requested
        confidence_scores = None
        if include_confidence:
            confidence_scores = await wrapper.predict_proba(data)
        
        # Create response(s)
        if isinstance(data, dict):
            # Single prediction
            confidence = None
            if confidence_scores is not None:
                if isinstance(confidence_scores, list) and len(confidence_scores) > 0:
                    # For classification, use max probability as confidence
                    confidence = max(confidence_scores) if isinstance(confidence_scores[0], (int, float)) else max(confidence_scores[0])
                elif isinstance(confidence_scores, (int, float)):
                    confidence = confidence_scores
            
            return PredictionResponse(
                prediction=predictions,
                confidence=confidence,
                model_name=model_name,
                model_version=wrapper.version,
                timestamp=datetime.utcnow()
            )
        else:
            # Batch predictions
            responses = []
            for i, prediction in enumerate(predictions):
                confidence = None
                if confidence_scores is not None and i < len(confidence_scores):
                    if isinstance(confidence_scores[i], list):
                        confidence = max(confidence_scores[i])
                    else:
                        confidence = confidence_scores[i]
                
                responses.append(PredictionResponse(
                    prediction=prediction,
                    confidence=confidence,
                    model_name=model_name,
                    model_version=wrapper.version,
                    timestamp=datetime.utcnow()
                ))
            
            return responses
    
    def get_model_info(self, model_name: Optional[str] = None) -> Union[ModelInfo, Dict[str, ModelInfo]]:
        """Get information about model(s).
        
        Args:
            model_name: Name of specific model (returns all if None)
            
        Returns:
            Model information
        """
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found")
            return self.models[model_name].get_info()
        else:
            return {name: wrapper.get_info() for name, wrapper in self.models.items()}
    
    def list_models(self) -> List[str]:
        """List all loaded model names.
        
        Returns:
            List of model names
        """
        return list(self.models.keys())
    
    async def reload_model(self, model_name: str) -> None:
        """Reload a specific model.
        
        Args:
            model_name: Name of model to reload
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        wrapper = self.models[model_name]
        
        # Reload the model
        await wrapper.load()
        
        logger.info("Model reloaded", model_name=model_name)
    
    async def unload_model(self, model_name: str) -> None:
        """Unload a specific model.
        
        Args:
            model_name: Name of model to unload
        """
        if model_name in self.models:
            del self.models[model_name]
            
            # Update default model if necessary
            if self.default_model == model_name:
                self.default_model = next(iter(self.models.keys())) if self.models else None
            
            logger.info("Model unloaded", model_name=model_name)


# Global model manager instance
model_manager = ModelManager()