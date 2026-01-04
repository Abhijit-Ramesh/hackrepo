#!/bin/bash
# Quick start script for Delhi Flood Risk System (Linux/Mac)

echo "========================================"
echo "Delhi Flood Risk Prediction System"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

echo "[1/3] Installing dependencies..."
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi
echo ""

echo "[2/3] Training machine learning model..."
python3 train_model.py
if [ $? -ne 0 ]; then
    echo "ERROR: Model training failed"
    exit 1
fi
echo ""

echo "[3/3] Starting FastAPI server (upload + map)..."
echo ""
echo "Starting local server on http://localhost:8000"
echo "Open your browser and navigate to:"
echo ""
echo "    Home (Upload CSV): http://localhost:8000/"
echo "    Map View:          http://localhost:8000/map.html"
echo ""
echo "Press Ctrl+C to stop the server when done."
echo ""

python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
