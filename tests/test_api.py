"""
Unit tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
import json
from unittest.mock import patch, MagicMock

from src.api.main import app

client = TestClient(app)

# Test API key from config
TEST_API_KEY = "demo-api-key-123"

class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check_success(self):
        """Test successful health check"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "model_status" in data
        assert "system_info" in data

class TestAuthenticatedEndpoints:
    """Test endpoints that require authentication"""
    
    def test_model_info_without_auth(self):
        """Test model info endpoint without authentication"""
        response = client.get("/model/info")
        assert response.status_code == 401
    
    def test_model_info_with_api_key(self):
        """Test model info endpoint with API key"""
        headers = {"X-API-Key": TEST_API_KEY}
        response = client.get("/model/info", headers=headers)
        # May return 503 if model not loaded, but should not be 401
        assert response.status_code != 401
    
    def test_predict_without_auth(self):
        """Test prediction endpoint without authentication"""
        sample_data = {
            "features": {
                "age": 35,
                "tenure": 24,
                "monthly_charges": 65.5,
                "total_charges": 1570.0,
                "internet_service": "Fiber optic",
                "contract": "Month-to-month",
                "payment_method": "Electronic check"
            }
        }
        response = client.post("/predict", json=sample_data)
        assert response.status_code == 401
    
    def test_predict_with_invalid_api_key(self):
        """Test prediction endpoint with invalid API key"""
        headers = {"X-API-Key": "invalid-key"}
        sample_data = {
            "features": {
                "age": 35,
                "tenure": 24,
                "monthly_charges": 65.5
            }
        }
        response = client.post("/predict", json=sample_data, headers=headers)
        assert response.status_code == 401
    
    @patch('src.api.predictor.predictor')
    def test_predict_with_valid_auth(self, mock_predictor):
        """Test prediction endpoint with valid authentication"""
        # Mock predictor
        mock_predictor.is_model_loaded.return_value = True
        mock_predictor.predict_single.return_value = {
            'prediction': 1,
            'probability': 0.75,
            'confidence': 'High',
            'model_used': 'test_model',
            'timestamp': '2024-01-15T10:30:00'
        }
        
        headers = {"X-API-Key": TEST_API_KEY}
        sample_data = {
            "features": {
                "age": 35,
                "tenure": 24,
                "monthly_charges": 65.5,
                "total_charges": 1570.0,
                "internet_service": "Fiber optic",
                "contract": "Month-to-month",
                "payment_method": "Electronic check"
            }
        }
        
        response = client.post("/predict", json=sample_data, headers=headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "confidence" in data

class TestValidation:
    """Test input validation"""
    
    def test_predict_empty_features(self):
        """Test prediction with empty features"""
        headers = {"X-API-Key": TEST_API_KEY}
        sample_data = {"features": {}}
        
        response = client.post("/predict", json=sample_data, headers=headers)
        assert response.status_code == 422  # Validation error
    
    def test_predict_missing_features_field(self):
        """Test prediction without features field"""
        headers = {"X-API-Key": TEST_API_KEY}
        sample_data = {"data": {"age": 35}}
        
        response = client.post("/predict", json=sample_data, headers=headers)
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict_empty_instances(self):
        """Test batch prediction with empty instances"""
        headers = {"X-API-Key": TEST_API_KEY}
        sample_data = {"instances": []}
        
        response = client.post("/predict/batch", json=sample_data, headers=headers)
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict_too_many_instances(self):
        """Test batch prediction with too many instances"""
        headers = {"X-API-Key": TEST_API_KEY}
        # Create more than 1000 instances
        instances = [{"age": 35, "tenure": 24}] * 1001
        sample_data = {"instances": instances}
        
        response = client.post("/predict/batch", json=sample_data, headers=headers)
        assert response.status_code == 422  # Validation error

class TestErrorHandling:
    """Test error handling"""
    
    @patch('src.api.predictor.predictor')
    def test_predict_model_not_loaded(self, mock_predictor):
        """Test prediction when model is not loaded"""
        mock_predictor.is_model_loaded.return_value = False
        
        headers = {"X-API-Key": TEST_API_KEY}
        sample_data = {
            "features": {
                "age": 35,
                "tenure": 24,
                "monthly_charges": 65.5
            }
        }
        
        response = client.post("/predict", json=sample_data, headers=headers)
        assert response.status_code == 503  # Service unavailable
    
    @patch('src.api.predictor.predictor')
    def test_predict_ml_exception(self, mock_predictor):
        """Test prediction with ML exception"""
        from src.utils.exceptions import MLException
        
        mock_predictor.is_model_loaded.return_value = True
        mock_predictor.predict_single.side_effect = MLException("Test ML error")
        
        headers = {"X-API-Key": TEST_API_KEY}
        sample_data = {
            "features": {
                "age": 35,
                "tenure": 24,
                "monthly_charges": 65.5
            }
        }
        
        response = client.post("/predict", json=sample_data, headers=headers)
        assert response.status_code == 400  # Bad request