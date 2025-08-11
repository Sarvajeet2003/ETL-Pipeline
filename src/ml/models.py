"""
Machine learning models module
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
from pathlib import Path
import json

from src.utils.logging import get_logger
from src.utils.exceptions import MLException, ModelTrainingException

logger = get_logger()

class ModelTrainer:
    """Train and evaluate machine learning models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.training_results = {}
    
    def get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get enabled model configurations
        
        Returns:
            Dictionary of enabled model configurations
        """
        enabled_models = {}
        
        for model_name, model_config in self.config['models'].items():
            if model_config.get('enabled', False):
                enabled_models[model_name] = model_config
        
        logger.info(f"Enabled models: {list(enabled_models.keys())}")
        return enabled_models
    
    def create_model(self, model_name: str, params: Dict[str, Any]) -> Any:
        """
        Create a model instance with given parameters
        
        Args:
            model_name: Name of the model
            params: Model parameters
            
        Returns:
            Model instance
        """
        try:
            if model_name == 'logistic_regression':
                return LogisticRegression(**params)
            elif model_name == 'random_forest':
                return RandomForestClassifier(**params)
            elif model_name == 'xgboost':
                return xgb.XGBClassifier(**params)
            else:
                raise ModelTrainingException(f"Unknown model type: {model_name}")
                
        except Exception as e:
            logger.error(f"Error creating model {model_name}: {str(e)}")
            raise ModelTrainingException(f"Failed to create model {model_name}: {str(e)}")
    
    def train_model_with_cv(
        self, 
        model_name: str, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """
        Train a model with cross-validation and hyperparameter tuning
        
        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Dictionary with training results
        """
        try:
            logger.info(f"Training {model_name} with cross-validation")
            
            model_config = self.config['models'][model_name]
            param_grid = model_config['params']
            
            # Create base model
            base_model = self.create_model(model_name, {})
            
            # Setup cross-validation
            cv_folds = self.config['model_selection']['cv_folds']
            scoring = self.config['model_selection']['scoring']
            n_jobs = self.config['model_selection']['n_jobs']
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_cv_score = grid_search.best_score_
            
            # Evaluate on validation set
            val_predictions = best_model.predict(X_val)
            val_probabilities = best_model.predict_proba(X_val)[:, 1] if hasattr(best_model, 'predict_proba') else None
            
            # Calculate metrics
            val_metrics = self.calculate_metrics(y_val, val_predictions, val_probabilities)
            
            # Store model and results
            self.models[model_name] = best_model
            
            training_result = {
                'model_name': model_name,
                'best_params': best_params,
                'cv_score': best_cv_score,
                'validation_metrics': val_metrics,
                'training_samples': len(X_train),
                'validation_samples': len(X_val)
            }
            
            self.training_results[model_name] = training_result
            
            logger.info(f"{model_name} training completed:")
            logger.info(f"  Best CV score: {best_cv_score:.4f}")
            logger.info(f"  Validation accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"  Best params: {best_params}")
            
            return training_result
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            raise ModelTrainingException(f"Failed to train {model_name}: {str(e)}")
    
    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add AUC if probabilities are available
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            except ValueError:
                # Handle case where only one class is present
                metrics['roc_auc'] = 0.5
        
        return metrics
    
    def train_all_models(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """
        Train all enabled models
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Dictionary with all training results
        """
        try:
            logger.info("Starting training for all enabled models")
            
            enabled_models = self.get_model_configs()
            
            for model_name in enabled_models:
                self.train_model_with_cv(model_name, X_train, y_train, X_val, y_val)
            
            # Select best model based on validation performance
            self.select_best_model()
            
            logger.info("All models training completed")
            return self.training_results
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise ModelTrainingException(f"Failed to train models: {str(e)}")
    
    def select_best_model(self) -> None:
        """Select the best model based on validation performance"""
        try:
            if not self.training_results:
                raise ModelTrainingException("No training results available")
            
            scoring_metric = self.config['model_selection']['scoring']
            
            best_score = -np.inf
            best_model_name = None
            
            for model_name, results in self.training_results.items():
                # Use validation accuracy as the primary metric for selection
                score = results['validation_metrics']['accuracy']
                
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
            
            self.best_model_name = best_model_name
            self.best_model = self.models[best_model_name]
            
            logger.info(f"Best model selected: {best_model_name} with score: {best_score:.4f}")
            
        except Exception as e:
            logger.error(f"Error selecting best model: {str(e)}")
            raise ModelTrainingException(f"Failed to select best model: {str(e)}")
    
    def evaluate_on_test(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate the best model on test set
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with test results
        """
        try:
            if self.best_model is None:
                raise ModelTrainingException("No best model available")
            
            logger.info(f"Evaluating {self.best_model_name} on test set")
            
            # Make predictions
            test_predictions = self.best_model.predict(X_test)
            test_probabilities = self.best_model.predict_proba(X_test)[:, 1] if hasattr(self.best_model, 'predict_proba') else None
            
            # Calculate metrics
            test_metrics = self.calculate_metrics(y_test, test_predictions, test_probabilities)
            
            test_results = {
                'model_name': self.best_model_name,
                'test_metrics': test_metrics,
                'test_samples': len(X_test)
            }
            
            logger.info(f"Test evaluation completed:")
            logger.info(f"  Test accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"  Test F1 score: {test_metrics['f1']:.4f}")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Error evaluating on test set: {str(e)}")
            raise ModelTrainingException(f"Failed to evaluate on test set: {str(e)}")
    
    def save_models(self, model_path: str) -> None:
        """
        Save trained models and results
        
        Args:
            model_path: Directory to save models
        """
        try:
            model_dir = Path(model_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save best model
            if self.best_model is not None:
                best_model_path = model_dir / "best_model.joblib"
                joblib.dump(self.best_model, best_model_path)
                logger.info(f"Saved best model ({self.best_model_name}) to {best_model_path}")
            
            # Save all models
            for model_name, model in self.models.items():
                model_path_file = model_dir / f"{model_name}_model.joblib"
                joblib.dump(model, model_path_file)
                logger.info(f"Saved {model_name} to {model_path_file}")
            
            # Save training results
            results_path = model_dir / "training_results.json"
            with open(results_path, 'w') as f:
                json.dump(self.training_results, f, indent=2)
            logger.info(f"Saved training results to {results_path}")
            
            # Save model metadata
            metadata = {
                'best_model_name': self.best_model_name,
                'models_trained': list(self.models.keys()),
                'config_used': self.config
            }
            metadata_path = model_dir / "model_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved model metadata to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise ModelTrainingException(f"Failed to save models: {str(e)}")
    
    def load_best_model(self, model_path: str) -> Any:
        """
        Load the best trained model
        
        Args:
            model_path: Directory containing models
            
        Returns:
            Loaded model
        """
        try:
            model_dir = Path(model_path)
            best_model_path = model_dir / "best_model.joblib"
            
            if not best_model_path.exists():
                raise ModelTrainingException(f"Best model not found at {best_model_path}")
            
            model = joblib.load(best_model_path)
            logger.info(f"Loaded best model from {best_model_path}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading best model: {str(e)}")
            raise ModelTrainingException(f"Failed to load best model: {str(e)}")