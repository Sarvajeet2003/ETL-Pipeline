#!/usr/bin/env python3
"""
Complete pipeline runner script
"""
import sys
import subprocess
from pathlib import Path
import argparse

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*50}")
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print("Output:", result.stdout)
    else:
        print(f"❌ {description} failed")
        if result.stderr:
            print("Error:", result.stderr)
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run the complete data science pipeline")
    parser.add_argument("--skip-data", action="store_true", help="Skip sample data generation")
    parser.add_argument("--skip-etl", action="store_true", help="Skip ETL pipeline")
    parser.add_argument("--skip-ml", action="store_true", help="Skip ML pipeline")
    parser.add_argument("--start-api", action="store_true", help="Start API server after pipeline")
    
    args = parser.parse_args()
    
    print("🚀 Starting Enterprise Data Science Pipeline")
    
    # Create necessary directories
    directories = ["data/raw", "data/processed", "models", "logs"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Step 1: Generate sample data
    if not args.skip_data:
        if not run_command(
            ["python", "data/sample/generate_sample_data.py"],
            "Sample data generation"
        ):
            return 1
    
    # Step 2: Run ETL pipeline
    if not args.skip_etl:
        if not run_command(
            ["python", "-m", "src.etl.pipeline"],
            "ETL pipeline"
        ):
            return 1
    
    # Step 3: Run ML pipeline
    if not args.skip_ml:
        if not run_command(
            ["python", "-m", "src.ml.pipeline"],
            "ML pipeline"
        ):
            return 1
    
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"📊 Check the logs/ directory for detailed reports")
    print(f"🤖 Models saved in the models/ directory")
    
    # Step 4: Start API server if requested
    if args.start_api:
        print(f"\n🌐 Starting API server...")
        try:
            subprocess.run(["uvicorn", "src.api.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"])
        except KeyboardInterrupt:
            print(f"\n👋 API server stopped")
    else:
        print(f"\n💡 To start the API server, run:")
        print(f"   uvicorn src.api.main:app --reload")
        print(f"   or")
        print(f"   python run_pipeline.py --start-api")
    
    return 0

if __name__ == "__main__":
    exit(main())