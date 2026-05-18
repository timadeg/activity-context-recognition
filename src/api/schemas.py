"""Pydantic schemas for the prediction API."""
from pydantic import BaseModel, Field
from typing import List


class SensorReading(BaseModel):
    """A single sensor reading with all 17 features."""
    orX: float = Field(..., description="Orientation X")
    orY: float = Field(..., description="Orientation Y")
    orZ: float = Field(..., description="Orientation Z")
    rX: float = Field(..., description="Rotation X")
    rY: float = Field(..., description="Rotation Y")
    rZ: float = Field(..., description="Rotation Z")
    accX: float = Field(..., description="Acceleration X")
    accY: float = Field(..., description="Acceleration Y")
    accZ: float = Field(..., description="Acceleration Z")
    gX: float = Field(..., description="Gravity X")
    gY: float = Field(..., description="Gravity Y")
    gZ: float = Field(..., description="Gravity Z")
    mX: float = Field(..., description="Magnetometer X")
    mY: float = Field(..., description="Magnetometer Y")
    mZ: float = Field(..., description="Magnetometer Z")
    lux: float = Field(..., description="Ambient light")
    soundLevel: float = Field(..., description="Ambient sound")

    model_config = {
        "json_schema_extra": {
            "example": {
                "orX": 127, "orY": -17, "orZ": 2,
                "rX": 0.071, "rY": -0.132, "rZ": -0.878,
                "accX": -0.038, "accY": 2.682, "accZ": 8.657,
                "gX": -0.041, "gY": 2.677, "gZ": 8.643,
                "mX": -31.2, "mY": -35.6, "mZ": -37.6,
                "lux": 5000, "soundLevel": 49.56,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response from /predict."""
    activity: str
    model_version: str


class HealthResponse(BaseModel):
    """Response from /health."""
    status: str
    model_loaded: bool
