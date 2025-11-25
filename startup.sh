#!/bin/bash

# ============================================================================
# Vozi Backend - Azure App Service Startup Script
# ============================================================================

echo "🚀 Starting Vozi Backend on Azure App Service..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Start Gunicorn server (production WSGI server)
echo "🌐 Starting Gunicorn server..."
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000 --timeout 600 --log-level info
