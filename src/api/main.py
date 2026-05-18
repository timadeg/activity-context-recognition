"""FastAPI inference service for activity recognition."""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd

from src.config import MODELS_DIR, SENSOR_COLUMNS
from src.models.predict import load_artifacts
from src.api.schemas import SensorReading, PredictionResponse, HealthResponse

MODEL_PATH = MODELS_DIR / "random_forest_v1.pkl"
SCALER_PATH = MODELS_DIR / "scaler_v1.pkl"
MODEL_VERSION = "random_forest_v1"
STATIC_DIR = Path(__file__).parent / "static"

artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading model from {MODEL_PATH}")
    model, scaler = load_artifacts(MODEL_PATH, SCALER_PATH)
    artifacts["model"] = model
    artifacts["scaler"] = scaler
    print("Model and scaler loaded successfully")
    yield
    artifacts.clear()


app = FastAPI(
    title="Activity Recognition API",
    description="Classify human activities from smartphone sensor data",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def root():
    """Serve the demo frontend."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/info", tags=["meta"])
def info():
    """JSON service metadata (was previously at /)."""
    return {
        "service": "Activity Recognition API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded="model" in artifacts and "scaler" in artifacts,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(reading: SensorReading):
    if "model" not in artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        row = {col: getattr(reading, col) for col in SENSOR_COLUMNS}
        df = pd.DataFrame([row], columns=SENSOR_COLUMNS)
        scaled = artifacts["scaler"].transform(df)
        prediction = artifacts["model"].predict(scaled)[0]
        return PredictionResponse(activity=str(prediction), model_version=MODEL_VERSION)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
