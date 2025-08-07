#!/usr/bin/env python3
"""
Example script demonstrating ETL pipeline usage
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from etl.pipeline import ETLPipeline


def main():
    """Run example ETL pipeline"""
    
    # Create example configuration
    example_config = {
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'logs/example_etl.log'
        },
        'output': {
            'format': 'csv',
            'path': 'examples/output/'
        },
        'data_sources': {
            'sample_csv': {
                'type': 'csv',
                'enabled': True,
                'config': {
                    'paths': ['examples/sample_data.csv'],
                    'encoding': 'utf-8',
                    'delimiter': ','
                }
            }
        },
        'data_cleaning': {
            'remove_duplicates': True,
            'handle_missing_values': 'fill',
            'fill_value': 'N/A',
            'text_cleaning': {
                'remove_special_chars': False,
                'lowercase': False,
                'remove_extra_whitespace': True
            }
        },
        'schema_validation': {
            'enabled': True,
            'strict_mode': False,
            'required_columns': ['id', 'name'],
            'column_types': {
                'id': 'int',
                'name': 'string',
                'age': 'int'
            }
        }
    }
    
    # Save example config
    import yaml
    config_path = Path('examples/example_config.yaml')
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(example_config, f, default_flow_style=False)
    
    print("Created example configuration at:", config_path)
    
    # Run pipeline
    try:
        pipeline = ETLPipeline(str(config_path))
        success = pipeline.run()
        
        if success:
            print("Example ETL pipeline completed successfully!")
            print("Check examples/output/ for processed data")
        else:
            print("Example ETL pipeline failed")
            
    except Exception as e:
        print(f"Error running example: {e}")


if __name__ == "__main__":
    main()