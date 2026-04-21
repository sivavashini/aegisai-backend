# AegisAI Trinity — Backend

Three-layer AI cybersecurity system backend.

## Stack
- FastAPI + Uvicorn
- MongoDB Atlas
- LightGBM + ONNX + Scikit-learn
- AlienVault OTX threat intel
- CEF logging for SIEM integration

## Setup

### 1. Clone and enter folder
cd backend

### 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / Pi

### 3. Install dependencies
pip install -r requirements.txt

### 4. Configure environment
Copy .env.example to .env and fill in real values:
- MONGO_URI — MongoDB Atlas connection string
- OTX_API_KEY — AlienVault OTX key

### 5. Add models
Copy all .pkl and .onnx files into backend/models/

### 6. Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

## Endpoints
- POST /ingest
- POST /predict
- POST /respond
- GET  /status
- GET  /incidents
- GET  /file-isolation/report
- GET  /trainiq/risk-score
- POST /trainiq/complete-module
- GET  /trainiq/all-users
- WS   /live

## Deployment
Railway — set all .env variables in Railway dashboard environment variables.