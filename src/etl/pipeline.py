"""
Main ETL pipeline orchestrator
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from pathlib import Path

from .config import ConfigManager
from .logger import ETLLogger
from .extractors import (
    CSVExtractor, SQLExtractor, PDFExtractor, 
    DOCXExtractor, SharePointExtractor
)
from .transformers import DataCleaner, SchemaValidator
from .loaders import CSVLoader, ParquetLoader, DatabaseLoader


class ETLPipeline:
    """Main ETL pipeline orchestrator"""
    
    # Extractor registry
    EXTRACTORS = {
        'csv': CSVExtractor,
        'sql': SQLExtractor,
        'pdf': PDFExtractor,
        'docx': DOCXExtractor,
        'sharepoint': SharePointExtractor
    }
    
    # Loader registry
    LOADERS = {
        'csv': CSVLoader,
        'parquet': ParquetLoader,
        'database': DatabaseLoader
    }
    
    def __init__(self, config_path: str = "config/etl_config.yaml"):
        """
        Initialize ETL pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config_manager = ConfigManager(config_path)
        
        # Setup logging
        log_config = self.config_manager.get('logging', {})
        ETLLogger(log_config)
        self.logger = ETLLogger.get_logger(self.__class__.__name__)
        
        # Initialize components
        self.data_cleaner = DataCleaner(self.config_manager.get('data_cleaning', {}))
        self.schema_validator = SchemaValidator(self.config_manager.get('schema_validation', {}))
        
        self.logger.info("ETL Pipeline initialized")
    
    def run(self) -> bool:
        """
        Execute the complete ETL pipeline
        
        Returns:
            True if pipeline completed successfully
        """
        self.logger.info("Starting ETL pipeline execution")
        
        try:
            # Get enabled data sources
            data_sources = self.config_manager.get_data_sources()
            
            if not data_sources:
                self.logger.warning("No enabled data sources found")
                return False
            
            # Process each data source
            all_dataframes = []
            
            for source_name, source_config in data_sources.items():
                self.logger.info(f"Processing data source: {source_name}")
                
                try:
                    # Extract data
                    dataframes = self._extract_data(source_name, source_config)
                    
                    # Transform each DataFrame
                    for df in dataframes:
                        if df is not None and not df.empty:
                            transformed_df = self._transform_data(df, source_name)
                            if transformed_df is not None:
                                all_dataframes.append((transformed_df, source_name))
                
                except Exception as e:
                    self.logger.error(f"Error processing data source {source_name}: {e}")
                    continue
            
            if not all_dataframes:
                self.logger.error("No data was successfully extracted and transformed")
                return False
            
            # Combine all DataFrames if multiple sources
            if len(all_dataframes) > 1:
                self.logger.info(f"Combining {len(all_dataframes)} DataFrames")
                combined_df = pd.concat([df for df, _ in all_dataframes], ignore_index=True)
                source_name = "combined"
            else:
                combined_df, source_name = all_dataframes[0]
            
            # Load data
            success = self._load_data(combined_df, source_name)
            
            if success:
                self.logger.info("ETL pipeline completed successfully")
                return True
            else:
                self.logger.error("ETL pipeline failed during loading")
                return False
                
        except Exception as e:
            self.logger.error(f"ETL pipeline failed: {e}")
            return False
    
    def _extract_data(self, source_name: str, source_config: Dict[str, Any]) -> List[pd.DataFrame]:
        """
        Extract data from a specific source
        
        Args:
            source_name: Name of the data source
            source_config: Source configuration
            
        Returns:
            List of extracted DataFrames
        """
        source_type = source_config.get('type')
        extractor_config = source_config.get('config', {})
        
        if source_type not in self.EXTRACTORS:
            raise ValueError(f"Unknown extractor type: {source_type}")
        
        # Create extractor instance
        extractor_class = self.EXTRACTORS[source_type]
        extractor = extractor_class(extractor_config)
        
        # Extract data
        dataframes = []
        for df in extractor.extract():
            if df is not None and not df.empty:
                dataframes.append(df)
                self.logger.info(f"Extracted DataFrame with {len(df)} rows from {source_name}")
        
        return dataframes
    
    def _transform_data(self, df: pd.DataFrame, source_name: str) -> Optional[pd.DataFrame]:
        """
        Transform a single DataFrame
        
        Args:
            df: Input DataFrame
            source_name: Name of the data source
            
        Returns:
            Transformed DataFrame or None if transformation failed
        """
        try:
            self.logger.info(f"Transforming data from {source_name}")
            
            # Validate schema (before cleaning)
            is_valid, errors = self.schema_validator.validate(df)
            if not is_valid and self.schema_validator.strict_mode:
                self.logger.error(f"Schema validation failed for {source_name}")
                return None
            
            # Clean data
            cleaned_df = self.data_cleaner.clean_dataframe(df)
            
            # Standardize column names
            cleaned_df = self.data_cleaner.standardize_column_names(cleaned_df)
            
            # Final validation
            is_valid, errors = self.schema_validator.validate(cleaned_df)
            if not is_valid:
                self.logger.warning(f"Final validation issues for {source_name}: {errors}")
            
            # Generate data profile
            profile = self.schema_validator.get_data_profile(cleaned_df)
            self.logger.info(f"Data profile for {source_name}: {profile['row_count']} rows, "
                           f"{profile['column_count']} columns, "
                           f"{profile['memory_usage_mb']:.2f} MB")
            
            return cleaned_df
            
        except Exception as e:
            self.logger.error(f"Error transforming data from {source_name}: {e}")
            return None
    
    def _load_data(self, df: pd.DataFrame, source_name: str) -> bool:
        """
        Load transformed data to output destination
        
        Args:
            df: DataFrame to load
            source_name: Name of the data source
            
        Returns:
            True if loading was successful
        """
        output_config = self.config_manager.get('output', {})
        output_format = output_config.get('format', 'parquet')
        
        if output_format not in self.LOADERS:
            self.logger.error(f"Unknown output format: {output_format}")
            return False
        
        # Create loader instance
        loader_class = self.LOADERS[output_format]
        loader_config = output_config.copy()
        
        # Use format-specific config if available
        if output_format in output_config:
            loader_config.update(output_config[output_format])
        
        loader = loader_class(loader_config)
        
        # Load data
        metadata = {'source_name': source_name}
        return loader.load(df, metadata)
    
    def run_single_source(self, source_name: str) -> bool:
        """
        Run ETL pipeline for a single data source
        
        Args:
            source_name: Name of the data source to process
            
        Returns:
            True if processing was successful
        """
        data_sources = self.config_manager.get_data_sources()
        
        if source_name not in data_sources:
            self.logger.error(f"Data source '{source_name}' not found or not enabled")
            return False
        
        self.logger.info(f"Processing single data source: {source_name}")
        
        try:
            source_config = data_sources[source_name]
            
            # Extract data
            dataframes = self._extract_data(source_name, source_config)
            
            if not dataframes:
                self.logger.error(f"No data extracted from {source_name}")
                return False
            
            # Process each DataFrame
            for i, df in enumerate(dataframes):
                transformed_df = self._transform_data(df, f"{source_name}_{i}")
                if transformed_df is not None:
                    success = self._load_data(transformed_df, f"{source_name}_{i}")
                    if not success:
                        return False
            
            self.logger.info(f"Successfully processed data source: {source_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing data source {source_name}: {e}")
            return False