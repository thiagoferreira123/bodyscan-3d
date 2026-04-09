from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """Metadata sent alongside images."""
    age: int = Field(..., ge=10, le=120, description="Age in years")
    height_cm: float = Field(..., ge=50, le=250, description="Height in centimeters")
    weight_kg: float = Field(..., ge=30, le=300, description="Weight in kilograms")
    gender: str = Field(..., pattern="^[MF]$", description="M or F")


class MeasurementsResponse(BaseModel):
    neck: float
    shoulder: float
    chest: float
    waist: float
    hip: float
    bicep_left: float
    bicep_right: float
    forearm_left: float
    forearm_right: float
    thigh_left: float
    thigh_right: float
    calf_left: float
    calf_right: float
    wrist: float


class CalculatedMetrics(BaseModel):
    bmi: float
    whr: float
    bsi: float
    bai: float
    ci: float
    ponderal_index: float


class ModelPredictions(BaseModel):
    xgboost: float
    lightgbm: float
    ridge: float


class EnsembleWeights(BaseModel):
    xgboost: float
    lightgbm: float
    ridge: float


class ModelVersions(BaseModel):
    stage_a: str
    stage_b: str


class AnalysisResponse(BaseModel):
    body_fat_pct: float
    lean_mass_kg: float
    fat_mass_kg: float
    muscle_mass_kg: float
    bone_mass_kg: float
    water_pct: float
    bmr: float
    tdee: float
    waist_cm: float
    measurements: MeasurementsResponse
    calculated_metrics: CalculatedMetrics
    model_predictions: ModelPredictions
    ensemble_weights: EnsembleWeights
    model_versions: ModelVersions
