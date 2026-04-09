"""
Body Scan ML Inference Pipeline
Stage A: EfficientNet-B0 (CNN) → 14 body measurements
Stage B: Ensemble (XGBoost + LightGBM + Ridge) → body fat %
"""

import io
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import timm
import xgboost as xgb
import lightgbm as lgb
from PIL import Image

warnings.filterwarnings("ignore")

MEASUREMENTS = [
    "neck", "shoulder", "chest", "waist", "hip",
    "bicep_left", "bicep_right", "forearm_left", "forearm_right",
    "thigh_left", "thigh_right", "calf_left", "calf_right", "wrist",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Stage A: CNN Model Definition
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(feat_dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)

    def forward(self, front_feat: torch.Tensor, side_feat: torch.Tensor) -> torch.Tensor:
        front = front_feat.unsqueeze(1)
        side = side_feat.unsqueeze(1)

        attn_front, _ = self.attention(front, side, side)
        attn_front = self.norm1(attn_front + front)

        attn_side, _ = self.attention(side, front, front)
        attn_side = self.norm2(attn_side + side)

        return torch.cat([attn_front.squeeze(1), attn_side.squeeze(1)], dim=1)


class SilhouetteToMeasurements(nn.Module):
    def __init__(self, num_measurements: int = 14, backbone: str = "efficientnet_b0", use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention

        self.front_branch = timm.create_model(backbone, pretrained=False, in_chans=1, num_classes=0, global_pool="avg")
        self.side_branch = timm.create_model(backbone, pretrained=False, in_chans=1, num_classes=0, global_pool="avg")

        dummy_input = torch.randn(1, 1, 224, 224)
        with torch.no_grad():
            feat_dim = self.front_branch(dummy_input).shape[1]

        if self.use_attention:
            self.cross_attention = CrossAttentionFusion(feat_dim)

        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_measurements),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front = x[:, 0:1, :, :]
        side = x[:, 1:2, :, :]

        front_feat = self.front_branch(front)
        side_feat = self.side_branch(side)

        if self.use_attention:
            combined = self.cross_attention(front_feat, side_feat)
        else:
            combined = torch.cat([front_feat, side_feat], dim=1)

        return self.fusion(combined)


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_stage_a(models_dir: Path) -> tuple:
    """Load Stage A CNN model and scaler. Returns (model, scaler)."""
    model_path = models_dir / "stage_a_v2_best.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Stage A model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = SilhouetteToMeasurements(num_measurements=len(MEASUREMENTS), backbone="efficientnet_b0")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    return model, checkpoint["scaler"]


def _load_hybrid_model(path: Path) -> dict[str, Any]:
    """Load hybrid format model (XGBoost UBJSON + LightGBM string + Ridge/Scaler pickle)."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    # XGBoost from UBJSON bytes
    xgb_booster = xgb.Booster()
    xgb_booster.load_model(bytearray(data["model_xgb_ubj"]))

    class XGBWrapper:
        def __init__(self, booster):
            self.booster = booster

        def predict(self, X):
            return self.booster.predict(xgb.DMatrix(X))

    # LightGBM from string
    lgb_booster = lgb.Booster(model_str=data["model_lgb_str"])

    class LGBMWrapper:
        def __init__(self, booster):
            self.booster_ = booster

        def predict(self, X):
            return self.booster_.predict(X)

    return {
        "model_xgb": XGBWrapper(xgb_booster),
        "model_lgb": LGBMWrapper(lgb_booster),
        "model_ridge": data["model_ridge"],
        "scaler": data["scaler"],
        "weights": data["weights"],
        "feature_names": data["feature_names"],
    }


def load_stage_b(models_dir: Path) -> tuple[dict, dict]:
    """Load Stage B ensemble models. Returns (male_model, female_model)."""
    male_path = models_dir / "stage_b__male.pkl"
    female_path = models_dir / "stage_b__female.pkl"

    if not male_path.exists():
        raise FileNotFoundError(f"Stage B male model not found: {male_path}")
    if not female_path.exists():
        raise FileNotFoundError(f"Stage B female model not found: {female_path}")

    return _load_hybrid_model(male_path), _load_hybrid_model(female_path)


# ---------------------------------------------------------------------------
# Image Preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Convert raw image bytes to a normalized grayscale 224x224 tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return torch.FloatTensor(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, 224, 224]


# ---------------------------------------------------------------------------
# Stage A: Predict Measurements
# ---------------------------------------------------------------------------

def predict_measurements(
    model: nn.Module,
    scaler: Any,
    front_bytes: bytes,
    side_bytes: bytes,
) -> dict[str, float]:
    """Predict 14 body measurements from front+side silhouette images."""
    front_tensor = preprocess_image(front_bytes).to(DEVICE)
    side_tensor = preprocess_image(side_bytes).to(DEVICE)

    # Concatenate front and side along channel dim → [1, 2, 224, 224]
    silhouettes = torch.cat([front_tensor, side_tensor], dim=1)

    with torch.no_grad():
        predictions = model(silhouettes)
        predictions_scaled = predictions.cpu().numpy()

    # Denormalize with scaler
    predictions_cm = scaler.inverse_transform(predictions_scaled)[0]

    return {name: float(pred) for name, pred in zip(MEASUREMENTS, predictions_cm)}


# ---------------------------------------------------------------------------
# Feature Engineering for Stage B
# ---------------------------------------------------------------------------

def engineer_features(
    waist_cm: float, hip_cm: float, neck_cm: float,
    age: float, height_cm: float, weight_kg: float,
) -> pd.DataFrame:
    """Create 55+ engineered features for Stage B prediction."""
    df = pd.DataFrame({
        "waist_cm": [waist_cm], "hip_cm": [hip_cm], "neck_cm": [neck_cm],
        "age": [age], "height_cm": [height_cm], "weight_kg": [weight_kg],
    })

    df["height_m"] = df["height_cm"] / 100
    df["bmi"] = df["weight_kg"] / (df["height_m"] ** 2)
    df["whr"] = df["waist_cm"] / df["height_cm"]
    df["waist_weight_ratio"] = df["waist_cm"] / df["weight_kg"]
    df["waist_squared"] = df["waist_cm"] ** 2
    df["waist_cubed"] = df["waist_cm"] ** 3
    df["sqrt_waist"] = np.sqrt(np.maximum(df["waist_cm"], 0))
    df["log_waist"] = np.log1p(np.maximum(df["waist_cm"], 0))
    df["age_squared"] = df["age"] ** 2
    df["age_cubed"] = df["age"] ** 3
    df["sqrt_age"] = np.sqrt(np.maximum(df["age"], 0))
    df["log_age"] = np.log1p(np.maximum(df["age"], 0))
    df["age_over_50"] = (df["age"] > 50).astype(float)
    df["bmi_squared"] = df["bmi"] ** 2
    df["bmi_cubed"] = df["bmi"] ** 3
    df["sqrt_bmi"] = np.sqrt(np.maximum(df["bmi"], 0))
    df["log_bmi"] = np.log1p(np.maximum(df["bmi"], 0))
    df["bmi_category"] = pd.cut(df["bmi"], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(float)
    df["weight_squared"] = df["weight_kg"] ** 2
    df["sqrt_weight"] = np.sqrt(np.maximum(df["weight_kg"], 0))
    df["log_weight"] = np.log1p(np.maximum(df["weight_kg"], 0))
    df["height_squared"] = df["height_cm"] ** 2
    df["sqrt_height"] = np.sqrt(np.maximum(df["height_cm"], 0))
    df["bmi_waist"] = df["bmi"] * df["waist_cm"]
    df["bmi_age"] = df["bmi"] * df["age"]
    df["waist_age"] = df["waist_cm"] * df["age"]
    df["height_weight_ratio"] = df["height_cm"] / df["weight_kg"]
    df["weight_height_product"] = df["weight_kg"] * df["height_cm"]
    df["bmi_whr"] = df["bmi"] * df["whr"]
    df["bsi"] = df["waist_cm"] / (df["bmi"] ** (2 / 3) * np.sqrt(np.maximum(df["height_cm"], 1)))
    df["ci"] = df["waist_cm"] / (0.109 * np.sqrt(np.maximum(df["weight_kg"] / np.maximum(df["height_cm"], 1), 0)))
    df["bai"] = (df["height_cm"] / np.sqrt(np.maximum(df["weight_kg"], 1))) - 18
    df["volume_approx"] = df["height_cm"] * df["weight_kg"]
    df["body_cylinder_volume"] = np.pi * ((df["waist_cm"] / (2 * np.pi)) ** 2) * df["height_cm"]
    df["bmi_age_waist"] = df["bmi"] * df["age"] * df["waist_cm"]
    df["bmi_height_weight"] = df["bmi"] * df["height_cm"] * df["weight_kg"]
    df["age_height_waist"] = df["age"] * df["height_cm"] * df["waist_cm"]
    df["bmi2_waist"] = (df["bmi"] ** 2) * df["waist_cm"]
    df["bmi_waist2"] = df["bmi"] * (df["waist_cm"] ** 2)
    df["age_bmi2"] = df["age"] * (df["bmi"] ** 2)
    df["age2_bmi"] = (df["age"] ** 2) * df["bmi"]
    df["waist2_age"] = (df["waist_cm"] ** 2) * df["age"]
    df["waist_per_height"] = df["waist_cm"] / df["height_cm"]
    df["weight_per_height"] = df["weight_kg"] / df["height_cm"]
    df["weight_per_height2"] = df["weight_kg"] / (df["height_cm"] ** 2)
    df["waist_per_weight"] = df["waist_cm"] / df["weight_kg"]
    df["ponderal_index"] = df["weight_kg"] / (df["height_m"] ** 3)
    df["corpulence_index"] = df["weight_kg"] / df["height_m"]
    df["quetelet_index"] = df["bmi"]

    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    return df


# ---------------------------------------------------------------------------
# Stage B: Predict Body Fat
# ---------------------------------------------------------------------------

def predict_bodyfat(
    male_model: dict, female_model: dict,
    measurements: dict[str, float],
    age: float, gender: str, height_cm: float, weight_kg: float,
) -> dict[str, Any]:
    """Predict body fat % using ensemble of XGBoost + LightGBM + Ridge."""
    model = male_model if gender == "M" else female_model

    features_df = engineer_features(
        measurements["waist"], measurements["hip"], measurements["neck"],
        age, height_cm, weight_kg,
    )

    feature_cols = model["feature_names"]
    X = features_df[feature_cols].values

    # Scale features
    X_scaled = model["scaler"].transform(X)

    # Get predictions from each model
    try:
        y_pred_xgb = float(model["model_xgb"].predict(X_scaled)[0])
    except Exception:
        y_pred_xgb = 0.0

    try:
        y_pred_lgb = float(model["model_lgb"].predict(X_scaled)[0])
    except Exception:
        y_pred_lgb = 0.0

    try:
        y_pred_ridge = float(model["model_ridge"].predict(X_scaled)[0])
    except Exception:
        y_pred_ridge = 0.0

    # Weighted ensemble
    weights = model["weights"]
    bodyfat_pct = (
        weights["xgb"] * y_pred_xgb
        + weights["lgb"] * y_pred_lgb
        + weights["ridge"] * y_pred_ridge
    )

    return {
        "bodyfat_pct": float(bodyfat_pct),
        "calculated_metrics": {
            "bmi": float(features_df["bmi"].values[0]),
            "whr": float(features_df["whr"].values[0]),
            "bsi": float(features_df["bsi"].values[0]),
            "bai": float(features_df["bai"].values[0]),
            "ci": float(features_df["ci"].values[0]),
            "ponderal_index": float(features_df["ponderal_index"].values[0]),
        },
        "model_predictions": {
            "xgboost": y_pred_xgb,
            "lightgbm": y_pred_lgb,
            "ridge": y_pred_ridge,
        },
        "ensemble_weights": {
            "xgboost": float(weights["xgb"]),
            "lightgbm": float(weights["lgb"]),
            "ridge": float(weights["ridge"]),
        },
    }


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def run_inference(
    stage_a_model: nn.Module,
    stage_a_scaler: Any,
    stage_b_male: dict,
    stage_b_female: dict,
    front_image_bytes: bytes,
    side_image_bytes: bytes,
    age: int,
    height_cm: float,
    weight_kg: float,
    gender: str,
) -> dict[str, Any]:
    """
    Run the full 2-stage inference pipeline.
    Returns a dict with all metrics ready for the API response.
    """
    # Stage A: CNN → 14 measurements
    all_measurements = predict_measurements(
        stage_a_model, stage_a_scaler, front_image_bytes, side_image_bytes,
    )

    # Stage B: Ensemble → body fat %
    stage_b_result = predict_bodyfat(
        stage_b_male, stage_b_female,
        all_measurements, age, gender, height_cm, weight_kg,
    )

    bodyfat_pct = stage_b_result["bodyfat_pct"]
    fat_mass_kg = (bodyfat_pct / 100.0) * weight_kg
    lean_mass_kg = weight_kg - fat_mass_kg

    # Derived body composition estimates
    muscle_mass_kg = lean_mass_kg * 0.56  # ~56% of lean mass is skeletal muscle
    bone_mass_kg = lean_mass_kg * 0.12    # ~12% of lean mass is bone
    water_pct = lean_mass_kg * 0.73 / weight_kg * 100  # ~73% of lean mass is water

    # BMR (Harris-Benedict Revised)
    if gender == "M":
        bmr = 88.362 + 13.397 * weight_kg + 4.799 * height_cm - 5.677 * age
    else:
        bmr = 447.593 + 9.247 * weight_kg + 3.098 * height_cm - 4.33 * age

    tdee = bmr * 1.2  # sedentary factor

    return {
        "body_fat_pct": round(bodyfat_pct, 2),
        "lean_mass_kg": round(lean_mass_kg, 2),
        "fat_mass_kg": round(fat_mass_kg, 2),
        "muscle_mass_kg": round(muscle_mass_kg, 2),
        "bone_mass_kg": round(bone_mass_kg, 2),
        "water_pct": round(water_pct, 2),
        "bmr": round(bmr, 2),
        "tdee": round(tdee, 2),
        "waist_cm": round(all_measurements["waist"], 2),
        "measurements": all_measurements,
        "calculated_metrics": stage_b_result["calculated_metrics"],
        "model_predictions": stage_b_result["model_predictions"],
        "ensemble_weights": stage_b_result["ensemble_weights"],
        "model_versions": {
            "stage_a": "v2.0",
            "stage_b": "v6.3_hybrid",
        },
    }
