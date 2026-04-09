"""Dependency injection for loaded ML models."""

from dataclasses import dataclass
from typing import Any

import torch.nn as nn


@dataclass
class LoadedModels:
    stage_a_model: nn.Module
    stage_a_scaler: Any
    stage_b_male: dict
    stage_b_female: dict


# Singleton — set during app lifespan
_models: LoadedModels | None = None


def set_models(models: LoadedModels) -> None:
    global _models
    _models = models


def get_models() -> LoadedModels:
    if _models is None:
        raise RuntimeError("Models not loaded yet. Is the app starting?")
    return _models
