#!/usr/bin/env python3
"""
API server startup script with proper Python path setup
"""
import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Set environment variable for subprocess
os.environ['PYTHONPATH'] = str(current_dir)

if __name__ == "__main__":
    import uvicorn
    from src.api.main import app
    
    print(f"🚀 Starting API server...")
    print(f"📁 Working directory: {current_dir}")
    print(f"🐍 Python path: {sys.path[0]}")
    print(f"🌐 Server will be available at: http://localhost:8000")
    print(f"📚 API docs at: http://localhost:8000/docs")
    
    try:
        uvicorn.run(
            "src.api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print(f"\n👋 API server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)