"""
Body Scan ML Inference — FastAPI Microservice
Receives 1-2 images + metadata, returns body composition metrics.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .dependencies import LoadedModels, set_models, get_models
from .inference import load_stage_a, load_stage_b, run_inference
from .models import AnalysisResponse

logger = logging.getLogger("bodyscan")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models at startup, release at shutdown."""
    models_dir = Path(settings.models_dir)
    logger.info(f"Loading models from {models_dir.resolve()}")

    stage_a_model, stage_a_scaler = load_stage_a(models_dir)
    logger.info("Stage A (CNN) loaded")

    stage_b_male, stage_b_female = load_stage_b(models_dir)
    logger.info("Stage B (Ensemble) loaded")

    set_models(LoadedModels(
        stage_a_model=stage_a_model,
        stage_a_scaler=stage_a_scaler,
        stage_b_male=stage_b_male,
        stage_b_female=stage_b_female,
    ))
    logger.info("All models ready")

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="Body Scan ML Service",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


ALLOWED_MIMETYPES = {"image/jpeg", "image/png", "image/webp"}


@app.get("/health")
async def health():
    try:
        get_models()
        return {"status": "ok", "models_loaded": True}
    except RuntimeError:
        return {"status": "starting", "models_loaded": False}


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze(
    front_image: UploadFile = File(..., description="Front silhouette image"),
    side_image: UploadFile = File(..., description="Side silhouette image"),
    age: int = Form(..., ge=10, le=120),
    height_cm: float = Form(..., ge=50, le=250),
    weight_kg: float = Form(..., ge=30, le=300),
    gender: str = Form(..., pattern="^[MF]$"),
):
    """
    Analyze body composition from front + side silhouette images.
    Returns body fat %, lean mass, muscle mass, bone mass, water %, BMR, TDEE,
    and 14 body measurements.
    """
    # Validate MIME types
    if front_image.content_type not in ALLOWED_MIMETYPES:
        raise HTTPException(400, f"front_image must be JPEG, PNG, or WebP (got {front_image.content_type})")
    if side_image.content_type not in ALLOWED_MIMETYPES:
        raise HTTPException(400, f"side_image must be JPEG, PNG, or WebP (got {side_image.content_type})")

    models = get_models()

    front_bytes = await front_image.read()
    side_bytes = await side_image.read()

    try:
        result = run_inference(
            stage_a_model=models.stage_a_model,
            stage_a_scaler=models.stage_a_scaler,
            stage_b_male=models.stage_b_male,
            stage_b_female=models.stage_b_female,
            front_image_bytes=front_bytes,
            side_image_bytes=side_bytes,
            age=age,
            height_cm=height_cm,
            weight_kg=weight_kg,
            gender=gender,
        )
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(500, f"Inference error: {str(e)}")

    return AnalysisResponse(**result)
