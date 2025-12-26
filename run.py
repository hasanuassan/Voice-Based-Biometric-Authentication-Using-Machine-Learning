"""
Main entry point for the Voice Authentication System
Provides easy startup for both backend and frontend
"""

import subprocess
import sys
import os
import time

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import fastapi
        import streamlit
        import librosa
        import sklearn
        import tensorflow
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting FastAPI backend server...")
    print("📍 API will be available at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    print("-" * 50)
    
    try:
        import uvicorn
        from api import app
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        sys.exit(1)

def start_frontend():
    """Start the Streamlit frontend"""
    print("🚀 Starting Streamlit frontend...")
    print("📍 UI will be available at: http://localhost:8501")
    print("-" * 50)
    
    try:
        os.system("streamlit run app.py")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        sys.exit(1)

def main():
    """Main function"""
    print("=" * 50)
    print("🎤 Voice-Based Biometric Authentication System")
    print("=" * 50)
    print()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print()
    print("Select mode:")
    print("1. Start Backend API only")
    print("2. Start Frontend UI only")
    print("3. Start Both (Backend + Frontend)")
    print()
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        start_backend()
    elif choice == "2":
        start_frontend()
    elif choice == "3":
        print("⚠️  Note: Starting both requires two terminal windows")
        print("Starting backend in this window...")
        print("Please open another terminal and run: streamlit run app.py")
        time.sleep(2)
        start_backend()
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main()

