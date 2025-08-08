"""Main ML training pipeline orchestrator."""

import os
import time
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

from .config import MLConfigManager, MLConfig
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer
from .models import (
    SklearnRandomForest, SklearnLinearRegression, SklearnLogisticRegression,
    XGBoostModel, LightGBMModel, CatBoostModel
)
from .hyperparameter_tuning import HyperparameterTuner
from .evaluation import ModelEvaluator
from .visualization import MLVisualizer
from .model_persistence import ModelPersistence

# MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class MLPipeline:
    """Main ML training pipeline orchestrator."""
    
    def __init__(self, config_path: str = "ml_config.yaml"):
        """Initialize ML pipeline.
        
        Args:
            config_path: Path to ML configuration file
        """
        self.config_manager = MLConfigManager(config_path)
        self.config = self.config_manager.load_config()
        self.logger = logger.bind(component="MLPipeline")
        
        # Initialize components
        self.data_loader = DataLoader(self.config.data)
        self.feature_engineer = FeatureEngineer(self.config.feature_engineering)
        self.evaluator = ModelEvaluator(self.config.evaluation)
        self.visualizer = MLVisualizer(self.config.visualization)
        self.model_persistence = ModelPersistence(self.config.model_persistence)
        
        # Setup logging
        self._setup_logging()
        
        # Setup MLflow if enabled
        if self.config.logging.mlflow.get('enabled', False):
            self._setup_mlflow()
        
        # Initialize model
        self.model = None
        self.feature_names = None
        
    def run(self) -> Dict[str, Any]:
        """Run the complete ML training pipeline.
        
        Returns:
            Dictionary containing pipeline execution results
        """
        self.logger.info("Starting ML training pipeline")
        
        pipeline_stats = {
            'experiment_name': self.config.experiment.name,
            'model_type': self.config.model.type,
            'data_shape': None,
            'feature_count': None,
            'training_time': 0,
            'evaluation_metrics': {},
            'model_path': None,
            'plots_created': [],
            'success': False
        }
        
        start_time = time.time()
        
        try:
            # Start MLflow run if enabled
            if MLFLOW_AVAILABLE and self.config.logging.mlflow.get('enabled', False):
                mlflow.start_run(run_name=f"{self.config.experiment.name}_{int(time.time())}")
                
                # Log experiment info
                mlflow.set_tags({
                    'experiment_name': self.config.experiment.name,
                    'model_type': self.config.model.type,
                    'description': self.config.experiment.description
                })
                
                for tag in self.config.experiment.tags:
                    mlflow.set_tag(f'tag_{tag}', True)
            
            # Load and prepare data
            self.logger.info("Loading and preparing data")
            X, y, data_summary = self._load_and_prepare_data()
            pipeline_stats['data_shape'] = X.shape
            
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.split_data(X, y)
            
            # Feature engineering
            self.logger.info("Applying feature engineering")
            X_train_processed, X_val_processed, X_test_processed = self.feature_engineer.fit_transform(
                X_train, y_train, X_val, X_test
            )
            
            # Store feature names
            self.feature_names = self.feature_engineer.get_feature_names()
            pipeline_stats['feature_count'] = X_train_processed.shape[1]
            
            # Initialize and train model
            self.logger.info("Training model")
            self.model = self._initialize_model()
            
            # Hyperparameter tuning if enabled
            tuning_results = None
            if self.config.model.hyperparameter_tuning.get('enabled', False):
                self.logger.info("Performing hyperparameter tuning")
                tuner = HyperparameterTuner(self.config.model.hyperparameter_tuning)
                tuning_results = tuner.tune_hyperparameters(
                    self.model, X_train_processed, y_train, X_val_processed, y_val
                )
                
                # Update model with best parameters
                if tuning_results.get('best_params'):
                    self.model.config['parameters'].update(tuning_results['best_params'])
                    self.model = self._initialize_model()  # Reinitialize with best params
            
            # Train final model
            training_start = time.time()
            self.model.fit(X_train_processed, y_train, X_val_processed, y_val)
            training_time = time.time() - training_start
            pipeline_stats['training_time'] = training_time
            
            # Evaluate model
            self.logger.info("Evaluating model")
            evaluation_metrics = self.evaluator.evaluate_model(
                self.model, X_test_processed, y_test, X_train_processed, y_train
            )
            pipeline_stats['evaluation_metrics'] = evaluation_metrics
            
            # Cross-validation if enabled
            if self.config.training.cross_validation.get('enabled', False):
                cv_results = self.evaluator.cross_validate_model(
                    self.model, X_train_processed, y_train,
                    cv_folds=self.config.training.cross_validation.get('cv_folds', 5),
                    scoring=self.config.training.cross_validation.get('scoring')
                )
                evaluation_metrics.update(cv_results)
            
            # Feature importance
            feature_importance = self.evaluator.calculate_feature_importance(
                self.model, self.feature_names
            )
            
            # Create visualizations
            if self.config.visualization.enabled:
                self.logger.info("Creating visualizations")
                y_pred = self.model.predict(X_test_processed)
                plot_files = self.visualizer.create_all_plots(
                    self.model, X_test_processed, y_test, y_pred,
                    evaluation_metrics, feature_importance, self.feature_names
                )
                pipeline_stats['plots_created'] = list(plot_files.values())
            
            # Save model
            self.logger.info("Saving model")
            model_metadata = {
                'data_summary': data_summary,
                'feature_names': self.feature_names,
                'training_time': training_time,
                'tuning_results': tuning_results
            }
            
            model_path = self.model_persistence.save_model(
                self.model, self.config.experiment.name,
                evaluation_metrics, model_metadata
            )
            pipeline_stats['model_path'] = model_path
            
            # Log to MLflow
            if MLFLOW_AVAILABLE and self.config.logging.mlflow.get('enabled', False):
                self._log_to_mlflow(evaluation_metrics, self.model, feature_importance)
            
            # Calculate total execution time
            total_time = time.time() - start_time
            pipeline_stats['total_execution_time'] = total_time
            pipeline_stats['success'] = True
            
            # Print summary
            self._print_pipeline_summary(pipeline_stats, evaluation_metrics)
            
            self.logger.info(f"ML pipeline completed successfully in {total_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"ML pipeline failed: {e}")
            pipeline_stats['error'] = str(e)
            raise
        
        finally:
            # End MLflow run
            if MLFLOW_AVAILABLE and self.config.logging.mlflow.get('enabled', False):
                mlflow.end_run()
        
        return pipeline_stats
    
    def _load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """Load and prepare data for training.
        
        Returns:
            Tuple of (features, target, data_summary)
        """
        # Load data
        df = self.data_loader.load_data()
        
        # Prepare features and target
        X, y = self.data_loader.prepare_features_and_target(df)
        
        # Get data summary
        data_summary = self.data_loader.get_data_summary(X, y)
        
        return X, y, data_summary
    
    def _initialize_model(self):
        """Initialize model based on configuration.
        
        Returns:
            Initialized model instance
        """
        model_type = self.config.model.type
        model_config = self.config.model.dict()
        
        model_classes = {
            'sklearn_rf': SklearnRandomForest,
            'sklearn_lr': SklearnLinearRegression,
            'sklearn_logistic': SklearnLogisticRegression,
            'xgboost': XGBoostModel,
            'lightgbm': LightGBMModel,
            'catboost': CatBoostModel
        }
        
        if model_type not in model_classes:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = model_classes[model_type]
        model = model_class(model_config)
        model.build_model()
        
        return model
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = self.config.logging
        
        # Create logs directory
        log_file = Path(log_config.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logger
        logger.remove()  # Remove default handler
        logger.add(
            log_file,
            level=log_config.level,
            format=log_config.format,
            rotation="10 MB",
            retention="30 days"
        )
        logger.add(
            lambda msg: print(msg, end=""),
            level=log_config.level,
            format=log_config.format
        )
        
        self.logger.info("ML Pipeline initialized")
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        if not MLFLOW_AVAILABLE:
            self.logger.warning("MLflow not available, skipping MLflow setup")
            return
        
        registry_config = self.config.model_persistence.registry
        
        if registry_config.get('enabled', False):
            tracking_uri = registry_config.get('tracking_uri', 'sqlite:///mlflow.db')
            mlflow.set_tracking_uri(tracking_uri)
            
            experiment_name = registry_config.get('experiment_name', 'ml_experiments')
            try:
                mlflow.create_experiment(experiment_name)
            except:
                pass  # Experiment already exists
            
            mlflow.set_experiment(experiment_name)
            
            self.logger.info(f"MLflow tracking configured: {tracking_uri}")
    
    def _log_to_mlflow(self, metrics: Dict[str, Any], model, feature_importance: Optional[pd.DataFrame]) -> None:
        """Log results to MLflow.
        
        Args:
            metrics: Evaluation metrics
            model: Trained model
            feature_importance: Feature importance DataFrame
        """
        if not MLFLOW_AVAILABLE:
            return
        
        mlflow_config = self.config.logging.mlflow
        
        # Log parameters
        if mlflow_config.get('log_params', True):
            mlflow.log_params(model.get_model_params())
        
        # Log metrics
        if mlflow_config.get('log_metrics', True):
            # Log main metrics
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)
        
        # Log model
        if mlflow_config.get('log_model', True):
            mlflow.sklearn.log_model(model.model, "model")
        
        # Log artifacts
        if mlflow_config.get('log_artifacts', True):
            # Log feature importance
            if feature_importance is not None:
                importance_file = "feature_importance.csv"
                feature_importance.to_csv(importance_file, index=False)
                mlflow.log_artifact(importance_file)
                os.remove(importance_file)  # Clean up
            
            # Log configuration
            config_file = "ml_config.yaml"
            self.config_manager.save_config(self.config, config_file)
            mlflow.log_artifact(config_file)
            os.remove(config_file)  # Clean up
    
    def _print_pipeline_summary(self, stats: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """Print pipeline execution summary.
        
        Args:
            stats: Pipeline statistics
            metrics: Evaluation metrics
        """
        print("\n" + "="*80)
        print("ML TRAINING PIPELINE SUMMARY")
        print("="*80)
        print(f"Experiment: {stats['experiment_name']}")
        print(f"Model Type: {stats['model_type']}")
        print(f"Data Shape: {stats['data_shape']}")
        print(f"Features: {stats['feature_count']}")
        print(f"Training Time: {stats['training_time']:.2f} seconds")
        print(f"Total Time: {stats.get('total_execution_time', 0):.2f} seconds")
        
        print("\nPERFORMANCE METRICS:")
        print("-" * 40)
        
        if metrics.get('is_classification', False):
            # Classification metrics
            if 'accuracy' in metrics:
                print(f"Accuracy:     {metrics['accuracy']:.4f}")
            if 'precision' in metrics:
                print(f"Precision:    {metrics['precision']:.4f}")
            if 'recall' in metrics:
                print(f"Recall:       {metrics['recall']:.4f}")
            if 'f1' in metrics:
                print(f"F1 Score:     {metrics['f1']:.4f}")
        else:
            # Regression metrics
            if 'r2' in metrics:
                print(f"R² Score:     {metrics['r2']:.4f}")
            if 'rmse' in metrics:
                print(f"RMSE:         {metrics['rmse']:.4f}")
            if 'mae' in metrics:
                print(f"MAE:          {metrics['mae']:.4f}")
        
        # Cross-validation results
        if 'cv_mean' in metrics:
            print(f"\nCross-Validation: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
        
        print(f"\nModel saved to: {stats['model_path']}")
        
        if stats['plots_created']:
            print(f"Plots created: {len(stats['plots_created'])}")
        
        print("="*80)
    
    def predict(self, data_path: str, model_version: Optional[str] = None) -> np.ndarray:
        """Make predictions using trained model.
        
        Args:
            data_path: Path to data for prediction
            model_version: Model version to use (if None, uses current model)
            
        Returns:
            Predictions array
        """
        # Load model if not already loaded or different version requested
        if self.model is None or model_version is not None:
            self.model, _ = self.model_persistence.load_model(
                self.config.experiment.name, model_version
            )
        
        # Load and prepare data
        if isinstance(data_path, str):
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.parquet'):
                df = pd.read_parquet(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
        else:
            df = data_path  # Assume it's already a DataFrame
        
        # Apply feature engineering
        X_processed = self.feature_engineer.transform(df)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        return predictions
    
    def get_pipeline_metadata(self) -> Dict[str, Any]:
        """Get metadata about the pipeline configuration.
        
        Returns:
            Dictionary containing pipeline metadata
        """
        return {
            'experiment_name': self.config.experiment.name,
            'model_type': self.config.model.type,
            'data_source': self.config.data.source_path,
            'target_column': self.config.data.target_column,
            'feature_engineering_enabled': self.config.feature_engineering.enabled,
            'hyperparameter_tuning_enabled': self.config.model.hyperparameter_tuning.get('enabled', False),
            'cross_validation_enabled': self.config.training.cross_validation.get('enabled', False),
            'visualization_enabled': self.config.visualization.enabled,
            'mlflow_enabled': self.config.logging.mlflow.get('enabled', False)
        }