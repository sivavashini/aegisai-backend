"""
config.py — AegisAI Trinity Central Configuration
All constants and environment variables loaded here.
Every other file imports from this file only.
"""

import os
from dotenv import load_dotenv

# Load .env file first before reading anything
load_dotenv()


# ─── MongoDB ────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB_NAME = "aegisai"
MONGO_COLLECTION_INCIDENTS = "incidents"
MONGO_COLLECTION_FILES = "file_isolation"
MONGO_COLLECTION_TRAINIQ = "trainiq"


# ─── OTX Threat Intel ───────────────────────────────────────
OTX_API_KEY = os.getenv("OTX_API_KEY", "")
OTX_REFRESH_INTERVAL = 900  # seconds — 15 minutes


# ─── Threat Score Thresholds ────────────────────────────────
THREAT_BLOCK_THRESHOLD = 71
THREAT_SANDBOX_THRESHOLD = 41
PHANTOMNET_CONFIDENCE_THRESHOLD = 0.60


# ─── Model Weights for AWT Scoring ──────────────────────────
RF_WEIGHT = 0.65
IF_WEIGHT = 0.35


# ─── File Paths ─────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
CEF_LOG_PATH = os.getenv("CEF_LOG_PATH", "alerts.cef")
BACKUP_PARTITION = os.getenv("BACKUP_PARTITION", "./backup")
QUARANTINE_DIR = os.getenv("QUARANTINE_DIR", "./quarantine")
PROTECTED_DIRS = os.getenv("PROTECTED_DIRS", "./protected").split(",")
BASELINE_PATH = os.path.join(os.path.dirname(__file__), "baseline.json")


# ─── Network ────────────────────────────────────────────────
PI1_IP = "192.168.1.10"
PI2_IP = "192.168.1.20"
PI3_IP = "192.168.1.30"
LAPTOP_IP = "192.168.1.40"
BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = 8000


# ─── GPIO Pin Numbers ───────────────────────────────────────
GPIO_RELAY = 17
GPIO_RED_LED = 27
GPIO_GREEN_LED = 22
GPIO_ORANGE_LED = 23


# ─── Railway Deployment ─────────────────────────────────────
IS_PRODUCTION = os.getenv("IS_PRODUCTION", "false").lower() == "true"


# ─── Validation — crash early if critical values missing ────
if not MONGO_URI:
    raise ValueError("MONGO_URI is not set in .env file")

if not OTX_API_KEY:
    raise ValueError("OTX_API_KEY is not set in .env file")