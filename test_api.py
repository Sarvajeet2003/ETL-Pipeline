#!/usr/bin/env python3
"""
API testing script
"""
import requests
import json
import time
from typing import Dict, Any

# API configuration
BASE_URL = "http://localhost:8000"
API_KEY = "demo-api-key-123"  # From config/api_config.yaml

def make_request(method: str, endpoint: str, data: Dict[Any, Any] = None, headers: Dict[str, str] = None) -> Dict[Any, Any]:
    """Make an API request"""
    url = f"{BASE_URL}{endpoint}"
    
    # Add API key to headers
    if headers is None:
        headers = {}
    headers["X-API-Key"] = API_KEY
    headers["Content-Type"] = "application/json"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, headers=headers)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        print(f"{method.upper()} {endpoint} - Status: {response.status_code}")
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.text}")
            return {"error": response.text, "status_code": response.status_code}
    
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection error. Make sure the API server is running at {BASE_URL}")
        return {"error": "Connection error"}
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return {"error": str(e)}

def test_health_check():
    """Test health check endpoint"""
    print("\n🏥 Testing health check...")
    result = make_request("GET", "/health")
    if "status" in result:
        print(f"✅ Health check passed - Status: {result['status']}")
        print(f"   Model loaded: {result.get('model_status', {}).get('model_loaded', False)}")
    else:
        print("❌ Health check failed")
    return result

def test_model_info():
    """Test model info endpoint"""
    print("\n📊 Testing model info...")
    result = make_request("GET", "/model/info")
    if "model_name" in result:
        print(f"✅ Model info retrieved")
        print(f"   Model: {result['model_name']}")
        print(f"   Type: {result['model_type']}")
        if "performance_metrics" in result:
            metrics = result["performance_metrics"]
            print(f"   Accuracy: {metrics.get('accuracy', 'N/A')}")
    else:
        print("❌ Model info failed")
    return result

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n🔮 Testing single prediction...")
    
    # Sample customer data
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
    
    result = make_request("POST", "/predict", sample_data)
    if "prediction" in result:
        print(f"✅ Prediction successful")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Probability: {result['probability']:.3f}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Model: {result['model_used']}")
    else:
        print("❌ Prediction failed")
    return result

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n📦 Testing batch prediction...")
    
    # Sample batch data
    batch_data = {
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
            },
            {
                "age": 28,
                "tenure": 6,
                "monthly_charges": 85.0,
                "total_charges": 510.0,
                "internet_service": "Fiber optic",
                "contract": "Month-to-month",
                "payment_method": "Electronic check"
            }
        ]
    }
    
    result = make_request("POST", "/predict/batch", batch_data)
    if "predictions" in result:
        print(f"✅ Batch prediction successful")
        print(f"   Batch size: {result['batch_size']}")
        print(f"   Processing time: {result['processing_time_ms']:.2f}ms")
        for i, pred in enumerate(result["predictions"]):
            print(f"   Instance {i+1}: Prediction={pred['prediction']}, Probability={pred['probability']:.3f}")
    else:
        print("❌ Batch prediction failed")
    return result

def test_retrain():
    """Test model retraining endpoint"""
    print("\n🔄 Testing model retraining...")
    
    retrain_data = {
        "data_path": None,  # Use default
        "model_types": None  # Train all models
    }
    
    result = make_request("POST", "/retrain", retrain_data)
    if "job_id" in result:
        print(f"✅ Retraining job started")
        print(f"   Job ID: {result['job_id']}")
        print(f"   Status: {result['status']}")
        
        # Check job status
        job_id = result['job_id']
        print(f"\n⏳ Checking job status...")
        
        for i in range(5):  # Check status 5 times
            time.sleep(2)
            status_result = make_request("GET", f"/retrain/status/{job_id}")
            if "status" in status_result:
                print(f"   Status check {i+1}: {status_result['status']}")
                if status_result['status'] in ['completed', 'failed']:
                    break
            else:
                print(f"   Status check {i+1}: Error getting status")
    else:
        print("❌ Retraining failed")
    return result

def main():
    """Run all API tests"""
    print("🧪 Starting API Tests")
    print(f"🌐 API Base URL: {BASE_URL}")
    print(f"🔑 Using API Key: {API_KEY[:10]}...")
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Model Info", test_model_info),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Model Retraining", test_retrain)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {str(e)}")
            results[test_name] = {"error": str(e)}
    
    # Summary
    print(f"\n📋 Test Summary")
    print(f"{'='*50}")
    for test_name, result in results.items():
        status = "✅ PASS" if "error" not in result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\n💡 API Documentation available at: {BASE_URL}/docs")
    print(f"📚 ReDoc documentation available at: {BASE_URL}/redoc")

if __name__ == "__main__":
    main()