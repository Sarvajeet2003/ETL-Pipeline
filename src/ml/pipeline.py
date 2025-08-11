"""
Main ML pipeline orchestrator
"""
import pandas as pd
from pathlib import Path
import json
from typing import Dict, Any

from src.utils.config import config_manager
from src.utils.logging import setup_logger, get_logger
from src.utils.exceptions import MLException
from src.ml.preprocessor import MLPreprocessor
from src.ml.models import ModelTrainer

class MLPipeline:
    """Main ML pipeline orchestrator"""
    
    def __init__(self):
        # Load configuration
        self.config = config_manager.get_ml_config()['ml']
        
        # Setup logging
        setup_logger(
            log_file=self.config['logging']['log_file'],
            level=self.config['logging']['level'],
            component="ML"
        )
        self.logger = get_logger()
        
        # Initialize components
        self.preprocessor = MLPreprocessor(self.config['data'])
        self.trainer = ModelTrainer(self.config)
        
        # Create output directories
        Path(self.config['output']['model_path']).mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete ML pipeline
        
        Returns:
            Dictionary with pipeline execution results
        """
        try:
            self.logger.info("Starting ML pipeline execution")
            
            # Step 1: Load and preprocess data
            self.logger.info("Step 1: Loading and preprocessing data")
            data_path = self.config['data']['input_path']
            df = self.preprocessor.load_data(data_path)
            
            # Split features and target
            X, y = self.preprocessor.split_features_target(df)
            
            # Split data into train/val/test
            X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data(X, y)
            
            # Feature engineering and preprocessing
            X_train_processed = self.preprocessor.encode_categorical_features(X_train, fit=True)
            X_train_processed = self.preprocessor.scale_numerical_features(X_train_processed, fit=True)
            
            X_val_processed = self.preprocessor.encode_categorical_features(X_val, fit=False)
            X_val_processed = self.preprocessor.scale_numerical_features(X_val_processed, fit=False)
            
            X_test_processed = self.preprocessor.encode_categorical_features(X_test, fit=False)
            X_test_processed = self.preprocessor.scale_numerical_features(X_test_processed, fit=False)
            
            # Step 2: Train models
            self.logger.info("Step 2: Training models")
            training_results = self.trainer.train_all_models(
                X_train_processed, y_train, X_val_processed, y_val
            )
            
            # Step 3: Evaluate on test set
            self.logger.info("Step 3: Evaluating on test set")
            test_results = self.trainer.evaluate_on_test(X_test_processed, y_test)
            
            # Step 4: Save models and preprocessors
            self.logger.info("Step 4: Saving models and preprocessors")
            model_path = self.config['output']['model_path']
            self.trainer.save_models(model_path)
            self.preprocessor.save_preprocessors(model_path)
            
            # Generate pipeline report (convert numpy types to native Python types)
            def convert_numpy_types(obj):
                """Convert numpy types to native Python types for JSON serialization"""
                if isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif hasattr(obj, 'tolist'):  # numpy array
                    return obj.tolist()
                else:
                    return obj
            
            pipeline_report = {
                'status': 'success',
                'timestamp': pd.Timestamp.now().isoformat(),
                'data_info': {
                    'total_samples': int(len(df)),
                    'features': int(len(X.columns)),
                    'target_distribution': {str(k): int(v) for k, v in y.value_counts().to_dict().items()},
                    'train_samples': int(len(X_train)),
                    'val_samples': int(len(X_val)),
                    'test_samples': int(len(X_test))
                },
                'training_results': convert_numpy_types(training_results),
                'test_results': convert_numpy_types(test_results),
                'best_model': str(self.trainer.best_model_name),
                'model_path': str(model_path)
            }
            
            # Save pipeline report
            report_path = Path("logs") / "ml_pipeline_report.json"
            with open(report_path, 'w') as f:
                json.dump(pipeline_report, f, indent=2)
            
            # Save metrics for API
            metrics_path = self.config['output']['metrics_path']
            Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, 'w') as f:
                json.dump({
                    'best_model': self.trainer.best_model_name,
                    'test_metrics': test_results['test_metrics'],
                    'training_results': training_results
                }, f, indent=2)
            
            self.logger.info(f"ML pipeline completed successfully")
            self.logger.info(f"Best model: {self.trainer.best_model_name}")
            self.logger.info(f"Test accuracy: {test_results['test_metrics']['accuracy']:.4f}")
            self.logger.info(f"Models saved to: {model_path}")
            self.logger.info(f"Pipeline report saved to: {report_path}")
            
            return pipeline_report
            
        except Exception as e:
            error_msg = f"ML pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            
            error_report = {
                'status': 'failed',
                'timestamp': pd.Timestamp.now().isoformat(),
                'error': str(error_msg)
            }
            
            # Save error report
            report_path = Path("logs") / "ml_pipeline_error.json"
            with open(report_path, 'w') as f:
                json.dump(error_report, f, indent=2)
            
            raise MLException(error_msg)

def main():
    """Main entry point for ML pipeline"""
    try:
        pipeline = MLPipeline()
        result = pipeline.run()
        print(f"ML Pipeline completed successfully!")
        print(f"Best model: {result['best_model']}")
        print(f"Test accuracy: {result['test_results']['test_metrics']['accuracy']:.4f}")
        print(f"Models saved to: {result['model_path']}")
        
    except Exception as e:
        print(f"ML Pipeline failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())