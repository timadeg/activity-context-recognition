from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Data
RAW_DATA_FILE = RAW_DATA_DIR / "activity_context_tracking_data.csv"
TARGET_COLUMN = "activity"

# Training
RANDOM_STATE = 42
TEST_SIZE = 0.2
DESIRED_SAMPLES_PER_CLASS = 2000

# Feature columns
SENSOR_COLUMNS = [
    "orX", "orY", "orZ",
    "rX", "rY", "rZ",
    "accX", "accY", "accZ",
    "gX", "gY", "gZ",
    "mX", "mY", "mZ",
    "lux", "soundLevel"
]
