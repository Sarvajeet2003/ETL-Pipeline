"""Hyperparameter tuning module."""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
import optuna
from loguru import logger

from .config import ModelConfig
from .models.base import BaseModel


class HyperparameterTuner:
    """Handles hyperparameter tuning for ML models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize hyperparameter tuner.
        
        Args:
            config: Hyperparameter tuning configuration
        """
        self.config = config
        self.logger = logger.bind(component="HyperparameterTuner")
        self.best_params = None
        self.best_score = None
        
    def tune_hyperparameters(self, model: BaseModel, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Tune hyperparameters for the given model.
        
        Args:
            model: Model to tune
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary containing best parameters and tuning results
        """
        if not self.config.get('enabled', False):
            self.logger.info("Hyperparameter tuning disabled")
            return {'best_params': model.get_model_params(), 'best_score': None}
        
        method = self.config.get('method', 'optuna')
        self.logger.info(f"Starting hyperparameter tuning using {method}")
        
        try:
            if method == 'optuna':
                return self._tune_with_optuna(model, X_train, y_train, X_val, y_val)
            elif method == 'grid_search':
                return self._tune_with_grid_search(model, X_train, y_train)
            elif method == 'random_search':
                return self._tune_with_random_search(model, X_train, y_train)
            else:
                self.logger.error(f"Unknown tuning method: {method}")
                return {'best_params': model.get_model_params(), 'best_score': None}
                
        except Exception as e:
            self.logger.error(f"Hyperparameter tuning failed: {e}")
            return {'best_params': model.get_model_params(), 'best_score': None}
    
    def _tune_with_optuna(self, model: BaseModel, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Tune hyperparameters using Optuna.
        
        Args:
            model: Model to tune
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Tuning results dictionary
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce output
        except ImportError:
            self.logger.error("Optuna not installed. Install with: pip install optuna")
            return {'best_params': model.get_model_params(), 'best_score': None}
        
        search_space = self.config.get('search_space', {})
        n_trials = self.config.get('n_trials', 50)
        
        def objective(trial):
            # Suggest parameters based on search space
            params = {}
            for param_name, param_values in search_space.items():
                if isinstance(param_values, list):
                    if all(isinstance(v, (int, float)) for v in param_values):
                        # Numeric range
                        if all(isinstance(v, int) for v in param_values):
                            params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                        else:
                            params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                    else:
                        # Categorical
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                elif isinstance(param_values, dict):
                    # Range specification
                    if param_values.get('type') == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name, 
                            param_values['min'], 
                            param_values['max'],
                            step=param_values.get('step', 1)
                        )
                    elif param_values.get('type') == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_values['min'],
                            param_values['max'],
                            step=param_values.get('step')
                        )
            
            # Create model with suggested parameters
            model_config = model.config.copy()
            model_config['parameters'].update(params)
            
            # Create new model instance
            model_class = model.__class__
            temp_model = model_class(model_config)
            temp_model.build_model()
            
            # Train and evaluate
            temp_model.fit(X_train, y_train, X_val, y_val)
            
            # Calculate score
            if X_val is not None and y_val is not None:
                y_pred = temp_model.predict(X_val)
                if temp_model.is_classifier():
                    from sklearn.metrics import accuracy_score
                    score = accuracy_score(y_val, y_pred)
                else:
                    from sklearn.metrics import r2_score
                    score = r2_score(y_val, y_pred)
            else:
                # Use cross-validation
                from sklearn.model_selection import cross_val_score
                cv_scores = cross_val_score(temp_model.model, X_train, y_train, cv=3)
                score = cv_scores.mean()
            
            return score
        
        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        self.logger.info(f"Optuna tuning completed. Best score: {self.best_score:.4f}")
        self.logger.info(f"Best parameters: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(study.trials),
            'study': study
        }
    
    def _tune_with_grid_search(self, model: BaseModel, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Tune hyperparameters using Grid Search.
        
        Args:
            model: Model to tune
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Tuning results dictionary
        """
        search_space = self.config.get('search_space', {})
        cv_folds = self.config.get('cv_folds', 5)
        
        # Prepare parameter grid
        param_grid = {}
        for param_name, param_values in search_space.items():
            if isinstance(param_values, list):
                param_grid[param_name] = param_values
            elif isinstance(param_values, dict) and 'values' in param_values:
                param_grid[param_name] = param_values['values']
        
        # Determine scoring metric
        if model.is_classifier():
            scoring = 'accuracy'
        else:
            scoring = 'r2'
        
        # Perform grid search
        grid_search = GridSearchCV(
            model.model,
            param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        self.logger.info(f"Grid search completed. Best score: {self.best_score:.4f}")
        self.logger.info(f"Best parameters: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'cv_results': grid_search.cv_results_,
            'grid_search': grid_search
        }
    
    def _tune_with_random_search(self, model: BaseModel, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Tune hyperparameters using Random Search.
        
        Args:
            model: Model to tune
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Tuning results dictionary
        """
        search_space = self.config.get('search_space', {})
        cv_folds = self.config.get('cv_folds', 5)
        n_iter = self.config.get('n_trials', 50)
        
        # Prepare parameter distributions
        param_distributions = {}
        for param_name, param_values in search_space.items():
            if isinstance(param_values, list):
                param_distributions[param_name] = param_values
            elif isinstance(param_values, dict):
                if param_values.get('type') == 'int':
                    from scipy.stats import randint
                    param_distributions[param_name] = randint(
                        param_values['min'], 
                        param_values['max'] + 1
                    )
                elif param_values.get('type') == 'float':
                    from scipy.stats import uniform
                    param_distributions[param_name] = uniform(
                        param_values['min'],
                        param_values['max'] - param_values['min']
                    )
        
        # Determine scoring metric
        if model.is_classifier():
            scoring = 'accuracy'
        else:
            scoring = 'r2'
        
        # Perform random search
        random_search = RandomizedSearchCV(
            model.model,
            param_distributions,
            n_iter=n_iter,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=0,
            random_state=42
        )
        
        random_search.fit(X_train, y_train)
        
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        
        self.logger.info(f"Random search completed. Best score: {self.best_score:.4f}")
        self.logger.info(f"Best parameters: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'cv_results': random_search.cv_results_,
            'random_search': random_search
        }
    
    def get_best_parameters(self) -> Optional[Dict[str, Any]]:
        """Get the best parameters found during tuning.
        
        Returns:
            Best parameters dictionary or None if tuning hasn't been performed
        """
        return self.best_params
    
    def get_best_score(self) -> Optional[float]:
        """Get the best score achieved during tuning.
        
        Returns:
            Best score or None if tuning hasn't been performed
        """
        return self.best_score