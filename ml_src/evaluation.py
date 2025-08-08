"""Model evaluation and metrics module."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    # Regression metrics
    mean_squared_error, mean_absolute_error, r2_score,
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_validate, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from .config import EvaluationConfig
from .models.base import BaseModel


class ModelEvaluator:
    """Handles model evaluation and metrics calculation."""
    
    def __init__(self, config: EvaluationConfig):
        """Initialize evaluator with configuration.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.logger = logger.bind(component="ModelEvaluator")
        
    def evaluate_model(self, model: BaseModel, X_test: np.ndarray, y_test: np.ndarray,
                      X_train: Optional[np.ndarray] = None, y_train: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Evaluate model performance on test data.
        
        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test targets
            X_train: Training features (optional, for comparison)
            y_train: Training targets (optional, for comparison)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info("Starting model evaluation")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Determine task type
        is_classification = model.is_classifier()
        
        # Calculate metrics
        if is_classification:
            metrics = self._calculate_classification_metrics(y_test, y_pred, y_pred_proba)
        else:
            metrics = self._calculate_regression_metrics(y_test, y_pred)
        
        # Add training metrics for comparison if available
        if X_train is not None and y_train is not None:
            train_pred = model.predict(X_train)
            if is_classification:
                train_metrics = self._calculate_classification_metrics(y_train, train_pred, None)
                metrics['train_metrics'] = train_metrics
            else:
                train_metrics = self._calculate_regression_metrics(y_train, train_pred)
                metrics['train_metrics'] = train_metrics
        
        # Add model-specific metrics
        metrics['model_type'] = model.model_type
        metrics['is_classification'] = is_classification
        
        self.logger.info("Model evaluation completed")
        return metrics
    
    def cross_validate_model(self, model: BaseModel, X: np.ndarray, y: np.ndarray,
                           cv_folds: int = 5, scoring: Optional[str] = None) -> Dict[str, Any]:
        """Perform cross-validation evaluation.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Targets
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric to use
            
        Returns:
            Cross-validation results
        """
        self.logger.info(f"Starting {cv_folds}-fold cross-validation")
        
        # Determine scoring metric if not provided
        if scoring is None:
            scoring = 'accuracy' if model.is_classifier() else 'neg_mean_squared_error'
        
        try:
            # Perform cross-validation
            cv_results = cross_validate(
                model.model, X, y,
                cv=cv_folds,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1
            )
            
            # Calculate summary statistics
            results = {
                'cv_scores': cv_results['test_score'].tolist(),
                'cv_mean': cv_results['test_score'].mean(),
                'cv_std': cv_results['test_score'].std(),
                'train_scores': cv_results['train_score'].tolist(),
                'train_mean': cv_results['train_score'].mean(),
                'train_std': cv_results['train_score'].std(),
                'scoring_metric': scoring,
                'cv_folds': cv_folds
            }
            
            self.logger.info(f"Cross-validation completed: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Cross-validation failed: {e}")
            return {}
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of regression metrics
        """
        metrics = {}
        
        # Get configured metrics
        regression_metrics = self.config.metrics.get('regression', [])
        
        for metric_name in regression_metrics:
            try:
                if metric_name == 'mse':
                    metrics['mse'] = mean_squared_error(y_true, y_pred)
                elif metric_name == 'rmse':
                    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
                elif metric_name == 'mae':
                    metrics['mae'] = mean_absolute_error(y_true, y_pred)
                elif metric_name == 'r2':
                    metrics['r2'] = r2_score(y_true, y_pred)
                elif metric_name == 'mape':
                    # Mean Absolute Percentage Error
                    metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                else:
                    self.logger.warning(f"Unknown regression metric: {metric_name}")
            except Exception as e:
                self.logger.error(f"Failed to calculate {metric_name}: {e}")
        
        return metrics
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                        y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary of classification metrics
        """
        metrics = {}
        
        # Get configured metrics
        classification_metrics = self.config.metrics.get('classification', [])
        
        for metric_name in classification_metrics:
            try:
                if metric_name == 'accuracy':
                    metrics['accuracy'] = accuracy_score(y_true, y_pred)
                elif metric_name == 'precision':
                    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
                elif metric_name == 'recall':
                    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
                elif metric_name == 'f1':
                    metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
                elif metric_name == 'roc_auc':
                    if y_pred_proba is not None:
                        if len(np.unique(y_true)) == 2:
                            # Binary classification
                            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                        else:
                            # Multi-class classification
                            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                    else:
                        self.logger.warning("Cannot calculate ROC AUC without prediction probabilities")
                else:
                    self.logger.warning(f"Unknown classification metric: {metric_name}")
            except Exception as e:
                self.logger.error(f"Failed to calculate {metric_name}: {e}")
        
        # Add confusion matrix
        try:
            metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        except Exception as e:
            self.logger.error(f"Failed to calculate confusion matrix: {e}")
        
        return metrics
    
    def generate_evaluation_report(self, metrics: Dict[str, Any], model_name: str = "Model") -> str:
        """Generate a formatted evaluation report.
        
        Args:
            metrics: Evaluation metrics dictionary
            model_name: Name of the model
            
        Returns:
            Formatted evaluation report string
        """
        report_lines = [
            f"{'='*60}",
            f"MODEL EVALUATION REPORT: {model_name}",
            f"{'='*60}",
            f"Model Type: {metrics.get('model_type', 'Unknown')}",
            f"Task Type: {'Classification' if metrics.get('is_classification', False) else 'Regression'}",
            ""
        ]
        
        # Test metrics
        report_lines.append("TEST SET PERFORMANCE:")
        report_lines.append("-" * 30)
        
        if metrics.get('is_classification', False):
            # Classification metrics
            if 'accuracy' in metrics:
                report_lines.append(f"Accuracy:     {metrics['accuracy']:.4f}")
            if 'precision' in metrics:
                report_lines.append(f"Precision:    {metrics['precision']:.4f}")
            if 'recall' in metrics:
                report_lines.append(f"Recall:       {metrics['recall']:.4f}")
            if 'f1' in metrics:
                report_lines.append(f"F1 Score:     {metrics['f1']:.4f}")
            if 'roc_auc' in metrics:
                report_lines.append(f"ROC AUC:      {metrics['roc_auc']:.4f}")
        else:
            # Regression metrics
            if 'r2' in metrics:
                report_lines.append(f"R² Score:     {metrics['r2']:.4f}")
            if 'rmse' in metrics:
                report_lines.append(f"RMSE:         {metrics['rmse']:.4f}")
            if 'mae' in metrics:
                report_lines.append(f"MAE:          {metrics['mae']:.4f}")
            if 'mape' in metrics:
                report_lines.append(f"MAPE:         {metrics['mape']:.2f}%")
        
        # Training metrics comparison
        if 'train_metrics' in metrics:
            report_lines.extend(["", "TRAINING SET COMPARISON:", "-" * 30])
            train_metrics = metrics['train_metrics']
            
            if metrics.get('is_classification', False):
                if 'accuracy' in train_metrics:
                    report_lines.append(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
            else:
                if 'r2' in train_metrics:
                    report_lines.append(f"Train R²:       {train_metrics['r2']:.4f}")
                if 'rmse' in train_metrics:
                    report_lines.append(f"Train RMSE:     {train_metrics['rmse']:.4f}")
        
        # Cross-validation results
        if 'cv_mean' in metrics:
            report_lines.extend(["", "CROSS-VALIDATION RESULTS:", "-" * 30])
            report_lines.append(f"CV Score:     {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
            report_lines.append(f"CV Folds:     {metrics.get('cv_folds', 'Unknown')}")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def calculate_feature_importance(self, model: BaseModel, feature_names: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """Calculate and return feature importance.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance or None if not available
        """
        importance_scores = model.get_feature_importance()
        
        if importance_scores is None:
            self.logger.warning("Feature importance not available for this model")
            return None
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance_scores))]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        self.logger.info(f"Feature importance calculated for {len(importance_df)} features")
        return importance_df