#!/usr/bin/env python3
"""
Main entry point for ETL pipeline
"""

import argparse
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from etl.pipeline import ETLPipeline


def main():
    """Main function to run ETL pipeline"""
    parser = argparse.ArgumentParser(description="Production ETL Pipeline")
    parser.add_argument(
        "--config", 
        default="config/etl_config.yaml",
        help="Path to configuration file (default: config/etl_config.yaml)"
    )
    parser.add_argument(
        "--source",
        help="Process only a specific data source"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running pipeline"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = ETLPipeline(args.config)
        
        if args.dry_run:
            print("Configuration validation successful")
            return 0
        
        # Run pipeline
        if args.source:
            success = pipeline.run_single_source(args.source)
        else:
            success = pipeline.run()
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())