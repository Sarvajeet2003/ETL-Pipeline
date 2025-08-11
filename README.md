# Enterprise Data Science Pipeline

A complete, enterprise-grade data science pipeline with ETL, ML, and API layers. This modular solution is designed for production use with configurable components, comprehensive logging, authentication, and automated model management.

## 🏗️ Architecture

The pipeline consists of three main layers:

1. **ETL Layer**: Data extraction, validation, cleaning, and preprocessing
2. **ML Layer**: Model training, validation, and versioning with configurable algorithms  
3. **API Layer**: FastAPI REST endpoints for predictions and model management

## 📁 Project Structure

```
├── config/                 # Configuration files (YAML)
│   ├── etl_config.yaml    # ETL pipeline settings
│   ├── ml_config.yaml     # ML model configurations
│   └── api_config.yaml    # API server settings
├── data/                   # Data storage
│   ├── raw/               # Raw input data
│   ├── processed/         # Cleaned data ready for ML
│   └── sample/            # Sample dataset generator
├── src/                   # Source code
│   ├── etl/               # ETL pipeline components
│   │   ├── extractor.py   # Data extraction
│   │   ├── validator.py   # Data validation
│   │   ├── cleaner.py     # Data cleaning
│   │   └── pipeline.py    # ETL orchestrator
│   ├── ml/                # ML pipeline components
│   │   ├── preprocessor.py # Feature engineering
│   │   ├── models.py      # Model training & evaluation
│   │   └── pipeline.py    # ML orchestrator
│   ├── api/               # API layer
│   │   ├── main.py        # FastAPI application
│   │   ├── auth.py        # Authentication & security
│   │   ├── models.py      # Pydantic models
│   │   └── predictor.py   # Prediction service
│   └── utils/             # Shared utilities
│       ├── config.py      # Configuration management
│       ├── logging.py     # Logging setup
│       └── exceptions.py  # Custom exceptions
├── models/                # Trained models & artifacts
├── logs/                  # Application logs & reports
├── tests/                 # Unit tests
├── run_pipeline.py        # Complete pipeline runner
├── test_api.py           # API testing script
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
└── docker-compose.yml    # Multi-service setup
```

## 🚀 Quick Start

### Option 1: Automated Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline (generates data, runs ETL & ML, starts API)
python run_pipeline.py --start-api
```

### Option 2: Step-by-Step
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate sample data
python data/sample/generate_sample_data.py

# 3. Run ETL pipeline
python -m src.etl.pipeline

# 4. Train ML models
python -m src.ml.pipeline

# 5. Start API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Option 3: Docker
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t ds-pipeline .
docker run -p 8000:8000 ds-pipeline
```

## 🔧 Configuration

All components are configurable via YAML files in the `config/` directory:

### ETL Configuration (`config/etl_config.yaml`)
- Input/output paths
- Data validation rules
- Cleaning parameters
- Logging settings

### ML Configuration (`config/ml_config.yaml`)
- Model types and hyperparameters
- Feature engineering settings
- Cross-validation parameters
- Model selection criteria

### API Configuration (`config/api_config.yaml`)
- Server settings
- Authentication (API keys, JWT)
- CORS configuration
- Model loading settings

## 📊 Sample Dataset

The pipeline includes a customer churn prediction dataset with:
- **Demographics**: Age, tenure
- **Services**: Internet service type, contract type
- **Billing**: Monthly charges, total charges, payment method
- **Target**: Customer churn (0=retained, 1=churned)

Features include missing values, duplicates, and outliers for ETL demonstration.

## 🤖 Supported ML Models

- **Logistic Regression**: Fast, interpretable baseline
- **Random Forest**: Robust ensemble method
- **XGBoost**: High-performance gradient boosting

All models support:
- Hyperparameter tuning via grid search
- Cross-validation
- Feature importance analysis
- Performance metrics tracking

## 🌐 API Endpoints

### Authentication
- **API Key**: Include `X-API-Key` header
- **JWT Token**: Include `Authorization: Bearer <token>` header

### Core Endpoints

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/health` | GET | System health check | No |
| `/model/info` | GET | Model information & metrics | Yes |
| `/predict` | POST | Single prediction | Yes |
| `/predict/batch` | POST | Batch predictions | Yes |
| `/retrain` | POST | Trigger model retraining | Yes |
| `/retrain/status/{job_id}` | GET | Check retraining status | Yes |
| `/model/reload` | POST | Reload model manually | Yes |

### API Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### API Testing
```bash
# Test all API endpoints
python test_api.py

# Manual testing with curl
curl -X GET "http://localhost:8000/health"
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: demo-api-key-123" \
  -H "Content-Type: application/json" \
  -d '{"features": {"age": 35, "tenure": 24, "monthly_charges": 65.5}}'
```

## 📈 Monitoring & Logging

### Structured Logging
- **ETL**: Data processing steps, validation results, cleaning statistics
- **ML**: Training progress, model performance, hyperparameter tuning
- **API**: Request/response logging, authentication events, errors

### Log Files
- `logs/etl.log`: ETL pipeline execution
- `logs/ml.log`: ML training and evaluation
- `logs/api.log`: API server requests and responses
- `logs/*_report.json`: Detailed pipeline execution reports

### Metrics Tracking
- Model performance metrics
- Data quality statistics
- API response times
- System resource usage

## 🔒 Security Features

- **API Key Authentication**: Simple key-based access control
- **JWT Token Support**: Stateless authentication with expiration
- **Input Validation**: Pydantic models for request/response validation
- **Error Handling**: Secure error messages without sensitive data exposure
- **CORS Configuration**: Configurable cross-origin resource sharing

## 🚀 Production Deployment

### Environment Variables
```bash
export PYTHONPATH=/path/to/project
export API_SECRET_KEY=your-production-secret-key
export LOG_LEVEL=INFO
```

### Docker Production
```bash
# Build production image
docker build -t ds-pipeline:prod .

# Run with production settings
docker run -d \
  -p 8000:8000 \
  -v /data:/app/data \
  -v /models:/app/models \
  -e LOG_LEVEL=WARNING \
  ds-pipeline:prod
```

### Scaling Considerations
- **Load Balancing**: Multiple API instances behind a load balancer
- **Model Versioning**: Automated model deployment and rollback
- **Data Pipeline**: Scheduled ETL runs with Apache Airflow or similar
- **Monitoring**: Integration with Prometheus, Grafana, or similar tools

## 🔄 Model Retraining

### Automated Retraining
```python
# Trigger via API
curl -X POST "http://localhost:8000/retrain" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"data_path": "data/raw/new_data.csv"}'
```

### Manual Retraining
```bash
# Update data and retrain
python -m src.etl.pipeline
python -m src.ml.pipeline
# API will auto-reload the new model
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**Model not loading**
- Check if models exist in `models/` directory
- Verify model file permissions
- Check logs for detailed error messages

**API authentication errors**
- Verify API key in `config/api_config.yaml`
- Check request headers format
- Ensure proper Content-Type header

**Pipeline failures**
- Check input data format and location
- Verify configuration file syntax
- Review logs for specific error details

### Support
For issues and questions:
1. Check the logs in `logs/` directory
2. Review configuration files
3. Run tests to verify setup
4. Check API documentation at `/docs`


# Health check
curl -X GET "http://localhost:8000/health"

# Model information
curl -X GET "http://localhost:8000/model/info"

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age": 45,
      "tenure": 36,
      "monthly_charges": 45.0,
      "total_charges": 1620.0,
      "internet_service": "DSL",
      "contract": "Two year",
      "payment_method": "Bank transfer"
    }
  }'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "age": 35,
        "tenure": 24,
        "monthly_charges": 65.5,
        "total_charges": 1570.0,
        "internet_service": "Fiber optic",
        "contract": "Month-to-month",
        "payment_method": "Electronic check"
      },
      {
        "age": 45,
        "tenure": 36,
        "monthly_charges": 45.0,
        "total_charges": 1620.0,
        "internet_service": "DSL",
        "contract": "Two year",
        "payment_method": "Bank transfer"
      }
    ]
  }'
