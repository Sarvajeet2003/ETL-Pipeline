"""Visualization module for ML training results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc
from loguru import logger

from .config import VisualizationConfig
from .models.base import BaseModel


class MLVisualizer:
    """Handles visualization of ML training results and model performance."""
    
    def __init__(self, config: VisualizationConfig):
        """Initialize visualizer with configuration.
        
        Args:
            config: Visualization configuration
        """
        self.config = config
        self.logger = logger.bind(component="MLVisualizer")
        
        # Create plots directory
        if self.config.save_plots:
            Path(self.config.plots_path).mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_all_plots(self, model: BaseModel, X_test: np.ndarray, y_test: np.ndarray,
                        y_pred: np.ndarray, metrics: Dict[str, Any],
                        feature_importance: Optional[pd.DataFrame] = None,
                        feature_names: Optional[List[str]] = None) -> Dict[str, str]:
        """Create all configured plots.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            y_pred: Predictions
            metrics: Evaluation metrics
            feature_importance: Feature importance DataFrame
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping plot types to file paths
        """
        if not self.config.enabled:
            return {}
        
        self.logger.info("Creating visualization plots")
        plot_files = {}
        
        for plot_type in self.config.plots:
            try:
                if plot_type == 'feature_importance' and feature_importance is not None:
                    plot_files[plot_type] = self.plot_feature_importance(feature_importance)
                
                elif plot_type == 'learning_curves':
                    training_history = model.get_training_history()
                    if training_history:
                        plot_files[plot_type] = self.plot_learning_curves(training_history)
                
                elif plot_type == 'residuals' and not model.is_classifier():
                    plot_files[plot_type] = self.plot_residuals(y_test, y_pred)
                
                elif plot_type == 'confusion_matrix' and model.is_classifier():
                    plot_files[plot_type] = self.plot_confusion_matrix(y_test, y_pred)
                
                elif plot_type == 'roc_curve' and model.is_classifier():
                    y_pred_proba = model.predict_proba(X_test)
                    if y_pred_proba is not None:
                        plot_files[plot_type] = self.plot_roc_curve(y_test, y_pred_proba)
                
                elif plot_type == 'prediction_vs_actual':
                    plot_files[plot_type] = self.plot_prediction_vs_actual(y_test, y_pred, model.is_classifier())
                
            except Exception as e:
                self.logger.error(f"Failed to create {plot_type} plot: {e}")
        
        self.logger.info(f"Created {len(plot_files)} visualization plots")
        return plot_files
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, top_n: int = 20) -> str:
        """Plot feature importance.
        
        Args:
            feature_importance: DataFrame with feature importance
            top_n: Number of top features to show
            
        Returns:
            Path to saved plot file
        """
        # Take top N features
        top_features = feature_importance.head(top_n)
        
        # Create plot
        plt.figure(figsize=(10, max(6, len(top_features) * 0.3)))
        
        # Horizontal bar plot
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score')
        plt.title(f'Top {len(top_features)} Feature Importance')
        plt.gca().invert_yaxis()  # Highest importance at top
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            plt.text(bar.get_width() + max(top_features['importance']) * 0.01, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{importance:.3f}', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"feature_importance.{self.config.plot_format}"
        filepath = Path(self.config.plots_path) / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_learning_curves(self, training_history: Dict[str, Any]) -> str:
        """Plot learning curves from training history.
        
        Args:
            training_history: Training history dictionary
            
        Returns:
            Path to saved plot file
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Extract metrics from training history
        # This will vary depending on the model type
        if 'validation_0' in training_history:
            # XGBoost/LightGBM format
            train_metric = list(training_history['validation_0'].values())[0]
            val_metric = list(training_history['validation_1'].values())[0] if 'validation_1' in training_history else None
            metric_name = list(training_history['validation_0'].keys())[0]
        else:
            # Generic format
            train_metric = training_history.get('train', [])
            val_metric = training_history.get('validation', [])
            metric_name = 'metric'
        
        # Plot training curve
        if train_metric:
            axes[0].plot(train_metric, label='Training', marker='o', markersize=3)
            if val_metric:
                axes[0].plot(val_metric, label='Validation', marker='s', markersize=3)
            
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel(metric_name.replace('_', ' ').title())
            axes[0].set_title('Learning Curves')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Plot loss/error curve (if different from main metric)
        if len(training_history) > 1:
            # Try to find a loss metric
            loss_key = None
            for key in training_history.keys():
                if 'loss' in key.lower() or 'error' in key.lower():
                    loss_key = key
                    break
            
            if loss_key and loss_key != 'validation_0':
                loss_values = training_history[loss_key]
                if isinstance(loss_values, dict):
                    loss_values = list(loss_values.values())[0]
                
                axes[1].plot(loss_values, label='Loss', color='red', marker='o', markersize=3)
                axes[1].set_xlabel('Iteration')
                axes[1].set_ylabel('Loss')
                axes[1].set_title('Training Loss')
                axes[1].grid(True, alpha=0.3)
        
        # If no second plot, remove the axis
        if not axes[1].has_data():
            fig.delaxes(axes[1])
            fig.set_size_inches(8, 5)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"learning_curves.{self.config.plot_format}"
        filepath = Path(self.config.plots_path) / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Plot residuals for regression models.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Path to saved plot file
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Residuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals vs Order
        axes[1, 1].plot(residuals, marker='o', markersize=3, alpha=0.6)
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Observation Order')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals vs Order')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"residuals.{self.config.plot_format}"
        filepath = Path(self.config.plots_path) / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Plot confusion matrix for classification models.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Path to saved plot file
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(y_true), 
                   yticklabels=np.unique(y_true))
        
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        
        # Save plot
        filename = f"confusion_matrix.{self.config.plot_format}"
        filepath = Path(self.config.plots_path) / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> str:
        """Plot ROC curve for binary classification.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Path to saved plot file
        """
        plt.figure(figsize=(8, 6))
        
        # Handle binary vs multiclass
        if len(np.unique(y_true)) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
        else:
            # Multiclass - plot ROC for each class
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_curve, auc
            
            # Binarize the output
            y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
            n_classes = y_true_bin.shape[1]
            
            # Compute ROC curve for each class
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, 
                        label=f'Class {i} (AUC = {roc_auc:.2f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        filename = f"roc_curve.{self.config.plot_format}"
        filepath = Path(self.config.plots_path) / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_prediction_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 is_classification: bool = False) -> str:
        """Plot predictions vs actual values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            is_classification: Whether this is a classification task
            
        Returns:
            Path to saved plot file
        """
        plt.figure(figsize=(8, 8))
        
        if is_classification:
            # For classification, create a confusion-style scatter plot
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            
            # Create scatter plot with jitter
            jitter = 0.1
            y_true_jitter = y_true + np.random.normal(0, jitter, len(y_true))
            y_pred_jitter = y_pred + np.random.normal(0, jitter, len(y_pred))
            
            plt.scatter(y_true_jitter, y_pred_jitter, alpha=0.6)
            
            # Perfect prediction line
            plt.plot(unique_labels, unique_labels, 'r--', lw=2, label='Perfect Prediction')
            
            plt.xlabel('True Labels')
            plt.ylabel('Predicted Labels')
            plt.title('Predictions vs Actual (Classification)')
            
        else:
            # For regression, create scatter plot
            plt.scatter(y_true, y_pred, alpha=0.6)
            
            # Perfect prediction line
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            # Calculate R²
            from sklearn.metrics import r2_score
            r2 = r2_score(y_true, y_pred)
            plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title('Predictions vs Actual (Regression)')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Make plot square
        plt.gca().set_aspect('equal', adjustable='box')
        
        # Save plot
        filename = f"prediction_vs_actual.{self.config.plot_format}"
        filepath = Path(self.config.plots_path) / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def create_interactive_dashboard(self, metrics: Dict[str, Any], 
                                   feature_importance: Optional[pd.DataFrame] = None) -> str:
        """Create an interactive dashboard using Plotly.
        
        Args:
            metrics: Evaluation metrics
            feature_importance: Feature importance DataFrame
            
        Returns:
            Path to saved HTML dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Metrics Overview', 'Feature Importance', 
                          'Performance Comparison', 'Model Info'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        # Metrics overview
        if metrics.get('is_classification', False):
            metric_names = ['accuracy', 'precision', 'recall', 'f1']
            metric_values = [metrics.get(name, 0) for name in metric_names]
        else:
            metric_names = ['r2', 'rmse', 'mae']
            metric_values = [metrics.get(name, 0) for name in metric_names]
        
        fig.add_trace(
            go.Bar(x=metric_names, y=metric_values, name="Test Metrics"),
            row=1, col=1
        )
        
        # Feature importance
        if feature_importance is not None:
            top_features = feature_importance.head(10)
            fig.add_trace(
                go.Bar(x=top_features['importance'], y=top_features['feature'], 
                      orientation='h', name="Feature Importance"),
                row=1, col=2
            )
        
        # Performance comparison (train vs test if available)
        if 'train_metrics' in metrics:
            train_metrics = metrics['train_metrics']
            comparison_metrics = []
            train_values = []
            test_values = []
            
            for metric in metric_names:
                if metric in metrics and metric in train_metrics:
                    comparison_metrics.append(metric)
                    train_values.append(train_metrics[metric])
                    test_values.append(metrics[metric])
            
            fig.add_trace(
                go.Bar(x=comparison_metrics, y=train_values, name="Train"),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(x=comparison_metrics, y=test_values, name="Test"),
                row=2, col=1
            )
        
        # Model info table
        model_info = [
            ['Model Type', metrics.get('model_type', 'Unknown')],
            ['Task Type', 'Classification' if metrics.get('is_classification', False) else 'Regression'],
            ['Best Metric', f"{max(metric_values):.4f}" if metric_values else 'N/A']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Property', 'Value']),
                cells=dict(values=list(zip(*model_info)))
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="ML Model Performance Dashboard",
            height=800,
            showlegend=True
        )
        
        # Save dashboard
        filename = f"dashboard.html"
        filepath = Path(self.config.plots_path) / filename
        fig.write_html(str(filepath))
        
        return str(filepath)