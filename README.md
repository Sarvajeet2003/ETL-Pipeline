# Production ETL Pipeline

A fully generic, production-ready ETL (Extract, Transform, Load) pipeline built in Python that supports multiple data sources and output formats.

## Features

- **Multiple Data Sources**: CSV files, SQL databases, PDF documents, DOCX files, SharePoint folders
- **Configurable**: All settings managed through YAML configuration files
- **Data Quality**: Schema validation, data cleaning, and profiling
- **Extensible**: Modular architecture for easy addition of new connectors
- **Production Ready**: Comprehensive logging, error handling, and monitoring
- **ML Ready**: Outputs cleaned data in formats suitable for machine learning (CSV, Parquet, Database)

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. **Configure Data Sources**
   ```bash
   # Edit config/etl_config.yaml to enable/configure your data sources
   ```

4. **Run Pipeline**
   ```bash
   python main.py
   ```

## Configuration

### Main Configuration (`config/etl_config.yaml`)

The pipeline is configured through a YAML file with the following sections:

- **logging**: Log level, format, and output file
- **output**: Output format (csv, parquet, database) and destination
- **data_sources**: Configuration for each data source type
- **data_cleaning**: Data cleaning and transformation rules
- **schema_validation**: Data quality validation rules

### Environment Variables

Sensitive information like credentials should be stored in environment variables:

```bash
# Database connections
DATABASE_URL=postgresql://user:password@localhost:5432/etl_db
SQL_CONNECTION_STRING=postgresql://user:password@localhost:5432/source_db

# SharePoint credentials
SHAREPOINT_SITE_URL=https://yourcompany.sharepoint.com/sites/yoursite
SHAREPOINT_USERNAME=your.email@company.com
SHAREPOINT_PASSWORD=your_password
```

## Data Sources

### CSV Files
```yaml
csv_files:
  type: "csv"
  enabled: true
  config:
    paths:
      - "data/input/*.csv"
    encoding: "utf-8"
    delimiter: ","
```

### SQL Databases
```yaml
sql_database:
  type: "sql"
  enabled: true
  config:
    connection_string: "${SQL_CONNECTION_STRING}"
    query: "SELECT * FROM source_table"
```

### PDF Documents
```yaml
pdf_documents:
  type: "pdf"
  enabled: true
  config:
    paths:
      - "data/documents/*.pdf"
```

### DOCX Documents
```yaml
docx_documents:
  type: "docx"
  enabled: true
  config:
    paths:
      - "data/documents/*.docx"
```

### SharePoint
```yaml
sharepoint:
  type: "sharepoint"
  enabled: true
  config:
    site_url: "${SHAREPOINT_SITE_URL}"
    username: "${SHAREPOINT_USERNAME}"
    password: "${SHAREPOINT_PASSWORD}"
    folder_path: "/Shared Documents"
    file_patterns:
      - "*.pdf"
      - "*.docx"
      - "*.xlsx"
```

## Usage Examples

### Run Complete Pipeline
```bash
python main.py
```

### Process Single Data Source
```bash
python main.py --source csv_files
```

### Use Custom Configuration
```bash
python main.py --config my_config.yaml
```

### Validate Configuration
```bash
python main.py --dry-run
```

## Architecture

```
src/etl/
├── __init__.py
├── config.py          # Configuration management
├── logger.py           # Logging setup
├── pipeline.py         # Main pipeline orchestrator
├── extractors/         # Data extraction modules
│   ├── base.py
│   ├── csv_extractor.py
│   ├── sql_extractor.py
│   ├── pdf_extractor.py
│   ├── docx_extractor.py
│   └── sharepoint_extractor.py
├── transformers/       # Data transformation modules
│   ├── data_cleaner.py
│   └── schema_validator.py
└── loaders/           # Data loading modules
    ├── base.py
    ├── csv_loader.py
    ├── parquet_loader.py
    └── database_loader.py
```

## Extending the Pipeline

### Adding New Extractors

1. Create a new extractor class inheriting from `BaseExtractor`
2. Implement required methods: `extract()` and `validate_config()`
3. Register the extractor in `pipeline.py`

```python
from .extractors.base import BaseExtractor

class MyExtractor(BaseExtractor):
    def validate_config(self) -> bool:
        # Validate configuration
        return True
    
    def extract(self) -> Iterator[pd.DataFrame]:
        # Extract data and yield DataFrames
        yield df
```

### Adding New Loaders

1. Create a new loader class inheriting from `BaseLoader`
2. Implement required methods: `load()` and `validate_config()`
3. Register the loader in `pipeline.py`

## Data Quality Features

- **Schema Validation**: Validate column presence, types, and data quality metrics
- **Data Cleaning**: Handle missing values, remove duplicates, clean text
- **Data Profiling**: Generate comprehensive data quality reports
- **Error Handling**: Robust error handling with detailed logging

## Monitoring and Logging

- Structured logging with configurable levels
- Log rotation to prevent disk space issues
- Detailed error messages and stack traces
- Data processing metrics and statistics

## Requirements

- Python 3.8+
- pandas >= 2.0.0
- SQLAlchemy >= 2.0.0
- PyPDF2 >= 3.0.0
- python-docx >= 0.8.11
- pyarrow >= 12.0.0 (for Parquet support)
- office365-rest-python-client >= 2.4.0 (for SharePoint)

## License

This project is licensed under the MIT License.