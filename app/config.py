from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    models_dir: str = "./models"
    cors_origins: str = "http://localhost:3000,http://localhost:3001"
    log_level: str = "info"

    # --- Stage A preprocessing -------------------------------------------
    # "legacy"  = original squash to 224x224 (DEFAULT: production unchanged).
    # "letterbox" = aspect-preserving + pad. ONLY enable after confirming the
    # training convention in the bodyscan-3d repo (see app/preprocess.py).
    preprocess_mode: str = "legacy"
    # Run person segmentation before resizing so the model sees a silhouette.
    # Requires `rembg` + `onnxruntime` and a mask convention matched to training.
    enable_segmentation: bool = False
    # Padding fill value for letterbox (0=black). Must match training.
    letterbox_pad: int = 0

    # --- Waist method ----------------------------------------------------
    # "model"    = Stage A CNN waist (DEFAULT: production behavior unchanged).
    # "geometry" = height-anchored geometric measurement from the uploaded
    #              silhouettes (requires the frontend silhouette + capture
    #              validation to be live). See app/geometry.py.
    waist_method: str = "model"
    # Ellipse shape factor for the geometric waist (calibrated on tape data;
    # 1.04 centers the one known real point on MediaPipe silhouettes).
    waist_geometry_k: float = 1.04

    # --- Post-hoc calibration --------------------------------------------
    # Optional JSON with per-sex, per-measurement affine correction:
    #   {"M": {"waist": [slope, intercept], ...}, "F": {...}}
    # corrected = slope * predicted + intercept. Absent -> no-op (identity).
    # This is the statistically correct way to remove systematic bias once you
    # have paired tape-measure ground truth. See README / inference.apply_calibration.
    calibration_path: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
