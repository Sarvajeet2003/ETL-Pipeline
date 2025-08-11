# Getting Started Guide

## 🎯 What You Have

You now have a complete enterprise-grade data science pipeline with:

✅ **Sample Dataset**: Customer churn data (1030 rows) with realistic features and target distribution  
✅ **ETL Pipeline**: Data extraction, validation, cleaning, and preprocessing  
✅ **ML Pipeline**: Multiple models (Logistic Regression, Random Forest, XGBoost) with hyperparameter tuning  
✅ **API Layer**: FastAPI REST endpoints with authentication and comprehensive documentation  
✅ **Configuration**: YAML-based configuration for all components  
✅ **Testing**: Unit tests and API testing scripts  
✅ **Docker**: Containerization support for easy deployment  

## 🚀 Next Steps

### 1. Run the Complete Pipeline
```bash
# Install dependencies (if not already done)
pip install -r requirements.txt

# Run the complete pipeline
python run_pipeline.py --start-api
```

This will:
- Generate sample data (already done ✅)
- Run ETL pipeline to clean and prepare data
- Train ML models and select the best one
- Start the API server at http://localhost:8000

### 2. Test the API
Once the server is running, test it:
```bash
# In a new terminal
python test_api.py
```

Or visit the interactive documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 3. Make Your First Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: demo-api-key-123" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age": 35,
      "tenure": 24,
      "monthly_charges": 65.5,
      "total_charges": 1570.0,
      "internet_service": "Fiber optic",
      "contract": "Month-to-month",
      "payment_method": "Electronic check"
    }
  }'
```

## 🔧 Customization for Your Use Case

### 1. Replace Sample Data
- Put your CSV files in `data/raw/`
- Update `config/etl_config.yaml` with your column requirements
- Modify `config/ml_config.yaml` to set your target column

### 2. Configure Models
Edit `config/ml_config.yaml` to:
- Enable/disable specific models
- Adjust hyperparameters
- Change cross-validation settings
- Set feature engineering options

### 3. Customize API Security
Edit `config/api_config.yaml` to:
- Change API keys
- Update JWT secret key
- Configure CORS settings
- Modify server settings

### 4. Add New Models
To add a new model type:
1. Update `src/ml/models.py` in the `create_model()` method
2. Add configuration in `config/ml_config.yaml`
3. The pipeline will automatically include it

## 📊 Understanding the Output

### ETL Pipeline Results
- **Input**: Raw CSV files from `data/raw/`
- **Output**: Cleaned data in `data/processed/cleaned_data.csv`
- **Report**: `logs/etl_pipeline_report.json`

### ML Pipeline Results
- **Models**: Saved in `models/` directory
- **Best Model**: `models/best_model.joblib`
- **Metrics**: `logs/ml_metrics.json`
- **Report**: `logs/ml_pipeline_report.json`

### API Server
- **Health Check**: GET `/health`
- **Predictions**: POST `/predict` and `/predict/batch`
- **Model Info**: GET `/model/info`
- **Retraining**: POST `/retrain`

## 🐛 Troubleshooting

### Common Issues

**"Module not found" errors**
```bash
export PYTHONPATH=$(pwd)
```

**API authentication errors**
- Use API key: `demo-api-key-123` (from config)
- Include header: `X-API-Key: demo-api-key-123`

**Model not loading**
- Ensure ML pipeline completed successfully
- Check `models/` directory for saved files
- Review `logs/ml.log` for errors

**Port already in use**
```bash
# Use a different port
uvicorn src.api.main:app --port 8001
```

### Getting Help
1. Check logs in `logs/` directory
2. Run tests: `pytest tests/`
3. Review configuration files in `config/`
4. Check API docs at `/docs` when server is running

## 🎯 Production Checklist

Before deploying to production:

- [ ] Change API keys and JWT secret in `config/api_config.yaml`
- [ ] Set up proper logging and monitoring
- [ ] Configure database for persistent storage (optional)
- [ ] Set up automated model retraining schedule
- [ ] Implement proper backup and recovery procedures
- [ ] Add rate limiting and additional security measures
- [ ] Set up CI/CD pipeline for automated testing and deployment

## 🚀 Scaling Up

For larger deployments:
- Use Docker Compose for multi-service setup
- Implement horizontal scaling with load balancers
- Add message queues for async processing
- Use container orchestration (Kubernetes)
- Implement model versioning and A/B testing
- Add comprehensive monitoring and alerting

---

**You're all set!** 🎉 

Your enterprise data science pipeline is ready to use. Start with the sample data to understand the workflow, then customize it for your specific use case.