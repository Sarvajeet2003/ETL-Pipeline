"""Model persistence and versioning module."""

import os
import json
import joblib
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib
from loguru import logger

from .config import ModelPersistenceConfig
from .models.base import BaseModel


class ModelPersistence:
    """Handles model saving, loading, and versioning."""
    
    def __init__(self, config: ModelPersistenceConfig):
        """Initialize model persistence manager.
        
        Args:
            config: Model persistence configuration
        """
        self.config = config
        self.logger = logger.bind(component="ModelPersistence")
        
        # Create save directory
        self.save_path = Path(self.config.save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
    def save_model(self, model: BaseModel, experiment_name: str, 
                  metrics: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save model with versioning and metadata.
        
        Args:
            model: Trained model to save
            experiment_name: Name of the experiment
            metrics: Model performance metrics
            metadata: Additional metadata to save
            
        Returns:
            Path to saved model directory
        """
        if not model.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        # Generate version
        version = self._generate_version()
        
        # Create model directory
        model_dir = self.save_path / experiment_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model
            model_file = model_dir / f"model.{self._get_file_extension()}"
            self._save_model_file(model, model_file)
            
            # Save metadata
            metadata_dict = {
                'experiment_name': experiment_name,
                'version': version,
                'model_type': model.model_type,
                'is_classifier': model.is_classifier(),
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'model_params': model.get_model_params(),
                'file_format': self.config.model_format
            }
            
            if metadata:
                metadata_dict.update(metadata)
            
            metadata_file = model_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2, default=str)
            
            # Save model configuration
            config_file = model_dir / "model_config.json"
            with open(config_file, 'w') as f:
                json.dump(model.config, f, indent=2, default=str)
            
            # Create model registry entry
            if self.config.versioning.get('enabled', True):
                self._update_model_registry(experiment_name, version, model_dir, metrics)
            
            self.logger.info(f"Model saved successfully: {model_dir}")
            return str(model_dir)
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, experiment_name: str, version: Optional[str] = None) -> tuple[BaseModel, Dict[str, Any]]:
        """Load model and metadata.
        
        Args:
            experiment_name: Name of the experiment
            version: Model version (if None, loads latest)
            
        Returns:
            Tuple of (loaded_model, metadata)
        """
        if version is None:
            version = self._get_latest_version(experiment_name)
            if version is None:
                raise ValueError(f"No models found for experiment: {experiment_name}")
        
        model_dir = self.save_path / experiment_name / version
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {model_dir}")
        
        try:
            # Load metadata
            metadata_file = model_dir / "metadata.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load model configuration
            config_file = model_dir / "model_config.json"
            with open(config_file, 'r') as f:
                model_config = json.load(f)
            
            # Create model instance
            model_type = metadata['model_type']
            model = self._create_model_instance(model_type, model_config)
            
            # Load model file
            model_file = model_dir / f"model.{self._get_file_extension()}"
            model.load_model(str(model_file))
            
            self.logger.info(f"Model loaded successfully: {model_dir}")
            return model, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def list_models(self, experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """List available models.
        
        Args:
            experiment_name: Filter by experiment name (optional)
            
        Returns:
            Dictionary of available models
        """
        models_info = {}
        
        if experiment_name:
            experiment_dirs = [self.save_path / experiment_name] if (self.save_path / experiment_name).exists() else []
        else:
            experiment_dirs = [d for d in self.save_path.iterdir() if d.is_dir()]
        
        for exp_dir in experiment_dirs:
            exp_name = exp_dir.name
            models_info[exp_name] = {}
            
            version_dirs = [d for d in exp_dir.iterdir() if d.is_dir()]
            
            for version_dir in version_dirs:
                version = version_dir.name
                metadata_file = version_dir / "metadata.json"
                
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        models_info[exp_name][version] = {
                            'model_type': metadata.get('model_type'),
                            'timestamp': metadata.get('timestamp'),
                            'metrics': metadata.get('metrics', {}),
                            'path': str(version_dir)
                        }
                    except Exception as e:
                        self.logger.warning(f"Failed to read metadata for {version_dir}: {e}")
        
        return models_info
    
    def delete_model(self, experiment_name: str, version: str) -> bool:
        """Delete a specific model version.
        
        Args:
            experiment_name: Name of the experiment
            version: Model version to delete
            
        Returns:
            True if successful, False otherwise
        """
        model_dir = self.save_path / experiment_name / version
        
        if not model_dir.exists():
            self.logger.warning(f"Model not found: {model_dir}")
            return False
        
        try:
            import shutil
            shutil.rmtree(model_dir)
            
            # Update registry
            self._remove_from_registry(experiment_name, version)
            
            self.logger.info(f"Model deleted: {model_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete model: {e}")
            return False
    
    def get_best_model(self, experiment_name: str, metric: str = 'r2') -> Optional[tuple[str, Dict[str, Any]]]:
        """Get the best model version based on a metric.
        
        Args:
            experiment_name: Name of the experiment
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (version, metadata) for best model or None
        """
        models = self.list_models(experiment_name)
        
        if experiment_name not in models or not models[experiment_name]:
            return None
        
        best_version = None
        best_score = float('-inf')
        best_metadata = None
        
        for version, info in models[experiment_name].items():
            metrics = info.get('metrics', {})
            if metric in metrics:
                score = metrics[metric]
                if score > best_score:
                    best_score = score
                    best_version = version
                    best_metadata = info
        
        return (best_version, best_metadata) if best_version else None
    
    def _generate_version(self) -> str:
        """Generate version string based on configuration.
        
        Returns:
            Version string
        """
        strategy = self.config.versioning.get('strategy', 'timestamp')
        
        if strategy == 'timestamp':
            return datetime.now().strftime('%Y%m%d_%H%M%S')
        elif strategy == 'semantic':
            # Simple semantic versioning (would need more logic for real use)
            return "v1.0.0"
        elif strategy == 'hash':
            # Generate hash based on current time
            return hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
        else:
            return datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def _get_file_extension(self) -> str:
        """Get file extension based on format configuration.
        
        Returns:
            File extension string
        """
        format_map = {
            'joblib': 'joblib',
            'pickle': 'pkl',
            'mlflow': 'mlflow'
        }
        return format_map.get(self.config.model_format, 'joblib')
    
    def _save_model_file(self, model: BaseModel, filepath: Path) -> None:
        """Save model to file using specified format.
        
        Args:
            model: Model to save
            filepath: Path to save the model
        """
        if self.config.model_format == 'joblib':
            joblib.dump(model.model, filepath)
        elif self.config.model_format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(model.model, f)
        else:
            # Default to joblib
            joblib.dump(model.model, filepath)
    
    def _get_latest_version(self, experiment_name: str) -> Optional[str]:
        """Get the latest version for an experiment.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Latest version string or None
        """
        exp_dir = self.save_path / experiment_name
        
        if not exp_dir.exists():
            return None
        
        version_dirs = [d for d in exp_dir.iterdir() if d.is_dir()]
        
        if not version_dirs:
            return None
        
        # Sort by modification time (most recent first)
        version_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        return version_dirs[0].name
    
    def _create_model_instance(self, model_type: str, config: Dict[str, Any]) -> BaseModel:
        """Create model instance from type and configuration.
        
        Args:
            model_type: Type of model to create
            config: Model configuration
            
        Returns:
            Model instance
        """
        # Import model classes
        from .models import (
            SklearnRandomForest, SklearnLinearRegression, SklearnLogisticRegression,
            XGBoostModel, LightGBMModel, CatBoostModel
        )
        
        model_classes = {
            'SklearnRandomForest': SklearnRandomForest,
            'SklearnLinearRegression': SklearnLinearRegression,
            'SklearnLogisticRegression': SklearnLogisticRegression,
            'XGBoostModel': XGBoostModel,
            'LightGBMModel': LightGBMModel,
            'CatBoostModel': CatBoostModel
        }
        
        if model_type not in model_classes:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model_classes[model_type](config)
    
    def _update_model_registry(self, experiment_name: str, version: str, 
                             model_dir: Path, metrics: Dict[str, Any]) -> None:
        """Update model registry with new model.
        
        Args:
            experiment_name: Name of the experiment
            version: Model version
            model_dir: Path to model directory
            metrics: Model metrics
        """
        registry_file = self.save_path / "model_registry.json"
        
        # Load existing registry
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                registry = json.load(f)
        else:
            registry = {}
        
        # Add new entry
        if experiment_name not in registry:
            registry[experiment_name] = {}
        
        registry[experiment_name][version] = {
            'path': str(model_dir),
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        # Save updated registry
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2, default=str)
    
    def _remove_from_registry(self, experiment_name: str, version: str) -> None:
        """Remove model from registry.
        
        Args:
            experiment_name: Name of the experiment
            version: Model version to remove
        """
        registry_file = self.save_path / "model_registry.json"
        
        if not registry_file.exists():
            return
        
        try:
            with open(registry_file, 'r') as f:
                registry = json.load(f)
            
            if experiment_name in registry and version in registry[experiment_name]:
                del registry[experiment_name][version]
                
                # Remove experiment if no versions left
                if not registry[experiment_name]:
                    del registry[experiment_name]
                
                # Save updated registry
                with open(registry_file, 'w') as f:
                    json.dump(registry, f, indent=2, default=str)
                    
        except Exception as e:
            self.logger.error(f"Failed to update registry: {e}")