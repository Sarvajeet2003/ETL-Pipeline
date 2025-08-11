"""
Main ETL pipeline orchestrator
"""
import pandas as pd
from pathlib import Path
import json
from typing import Dict, Any

from src.utils.config import config_manager
from src.utils.logging import setup_logger, get_logger
from src.utils.exceptions import ETLException
from src.etl.extractor import DataExtractor
from src.etl.validator import DataValidator
from src.etl.cleaner import DataCleaner

class ETLPipeline:
    """Main ETL pipeline orchestrator"""
    
    def __init__(self):
        # Load configuration
        self.config = config_manager.get_etl_config()['etl']
        
        # Setup logging
        setup_logger(
            log_file=self.config['logging']['log_file'],
            level=self.config['logging']['level'],
            component="ETL"
        )
        self.logger = get_logger()
        
        # Initialize components
        self.extractor = DataExtractor(
            source_path=self.config['input']['source_path'],
            file_pattern=self.config['input']['file_pattern']
        )
        self.validator = DataValidator(self.config['validation'])
        self.cleaner = DataCleaner(self.config['cleaning'])
        
        # Create output directory
        Path(self.config['output']['processed_path']).mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete ETL pipeline
        
        Returns:
            Dictionary with pipeline execution results
        """
        try:
            self.logger.info("Starting ETL pipeline execution")
            
            # Step 1: Extract data
            self.logger.info("Step 1: Extracting data")
            raw_data = self.extractor.extract_csv_files()
            data_info = self.extractor.get_data_info(raw_data)
            
            # Step 2: Validate data
            self.logger.info("Step 2: Validating data")
            self.validator.validate_structure(raw_data)
            self.validator.validate_data_quality(raw_data)
            self.validator.validate_business_rules(raw_data)
            validation_report = self.validator.get_validation_report()
            
            # Step 3: Clean data
            self.logger.info("Step 3: Cleaning data")
            cleaned_data = self.cleaner.remove_duplicates(raw_data)
            cleaned_data = self.cleaner.handle_missing_values(cleaned_data)
            cleaned_data = self.cleaner.remove_outliers(cleaned_data)
            cleaned_data = self.cleaner.standardize_data_types(cleaned_data)
            cleaning_report = self.cleaner.get_cleaning_report()
            
            # Step 4: Save processed data
            self.logger.info("Step 4: Saving processed data")
            output_path = Path(self.config['output']['processed_path']) / self.config['output']['output_filename']
            cleaned_data.to_csv(output_path, index=False)
            
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
                'input_data': convert_numpy_types(data_info),
                'validation': convert_numpy_types(validation_report),
                'cleaning': convert_numpy_types(cleaning_report),
                'output': {
                    'path': str(output_path),
                    'rows': int(len(cleaned_data)),
                    'columns': int(len(cleaned_data.columns)),
                    'file_size_mb': float(output_path.stat().st_size / (1024 * 1024)) if output_path.exists() else 0.0
                }
            }
            
            # Save pipeline report
            report_path = Path("logs") / "etl_pipeline_report.json"
            with open(report_path, 'w') as f:
                json.dump(pipeline_report, f, indent=2)
            
            self.logger.info(f"ETL pipeline completed successfully. Output saved to: {output_path}")
            self.logger.info(f"Pipeline report saved to: {report_path}")
            
            return pipeline_report
            
        except Exception as e:
            error_msg = f"ETL pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            
            error_report = {
                'status': 'failed',
                'timestamp': pd.Timestamp.now().isoformat(),
                'error': str(error_msg)
            }
            
            # Save error report
            report_path = Path("logs") / "etl_pipeline_error.json"
            with open(report_path, 'w') as f:
                json.dump(error_report, f, indent=2)
            
            raise ETLException(error_msg)

def main():
    """Main entry point for ETL pipeline"""
    try:
        pipeline = ETLPipeline()
        result = pipeline.run()
        print(f"ETL Pipeline completed successfully!")
        print(f"Processed {result['output']['rows']} rows")
        print(f"Output saved to: {result['output']['path']}")
        
    except Exception as e:
        print(f"ETL Pipeline failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())