"""
Image preprocessing for Stage A (silhouette -> 14 measurements).

WHY THIS MODULE EXISTS
----------------------
The original `preprocess_image` did two things that cause the waist
inconsistency users report:

  1. `Image.open(...).convert("L")` -> feeds the RAW grayscale photo to a
     model whose own class is `SilhouetteToMeasurements` and whose README
     documents the input as a *silhouette*. That is a train/serve skew:
     the model runs off-distribution (background, clothing, lighting leak in).

  2. `img.resize((224, 224))` -> squashes any aspect ratio into a square,
     NON-uniformly. Waist is read from width-in-frame; two photos of the
     same person at different crop/aspect ratios produce different waists.
     With no metric scale anchor, that distortion lands straight in cm.

IMPORTANT — MATCH TRAINING OR RETRAIN
-------------------------------------
The training pipeline lives in a separate repo (bodyscan-3d) and is NOT
visible here. Before enabling `letterbox` or `segment` in production you MUST
confirm the training preprocessing, because the model only behaves correctly
on inputs that match how it was trained:

  - Did training resize by SQUASH (cv2/PIL resize to 224x224) or LETTERBOX?
  - Were inputs real silhouette masks, and with what convention
    (person=white on black? grayscale person on black? filled vs edge)?
  - What padding color was used for any letterboxing?

If training squashed raw grayscale, keep mode="legacy". If training used
letterboxed silhouettes, set mode="letterbox" + a segmentation function that
reproduces the SAME mask convention. The cleanest fix is to retrain Stage A
with this exact module shared between train and serve.
"""

from __future__ import annotations

import io
from typing import Callable, Literal, Optional

import numpy as np
import torch
from PIL import Image

PreprocessMode = Literal["legacy", "letterbox"]

# A function that takes a PIL RGB image and returns a single-channel PIL mask
# (mode "L") with the SAME convention used at training time. Optional.
SegmentFn = Callable[[Image.Image], Image.Image]

TARGET = 224


def _legacy(img_l: Image.Image) -> Image.Image:
    """Exact reproduction of the original behavior: non-uniform squash."""
    return img_l.resize((TARGET, TARGET))


def _letterbox(img_l: Image.Image, pad: int = 0) -> Image.Image:
    """Aspect-preserving resize: scale longest side to TARGET, pad to square.

    `pad` is the fill value for the padded border (0 = black). The right value
    depends on the training silhouette convention — verify before enabling.
    """
    w, h = img_l.size
    scale = TARGET / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img_l.resize((nw, nh))
    canvas = Image.new("L", (TARGET, TARGET), color=pad)
    canvas.paste(resized, ((TARGET - nw) // 2, (TARGET - nh) // 2))
    return canvas


def preprocess_image(
    image_bytes: bytes,
    mode: PreprocessMode = "legacy",
    segment: Optional[SegmentFn] = None,
    pad: int = 0,
) -> torch.Tensor:
    """Convert raw image bytes to a [1, 1, 224, 224] float tensor in [0, 1].

    mode="legacy"    -> identical to the original pipeline (production default).
    mode="letterbox" -> aspect-preserving, removes the squash distortion.
    segment          -> optional callable producing a training-matched mask;
                        applied BEFORE resizing so the model sees a silhouette.
    """
    img = Image.open(io.BytesIO(image_bytes))

    if segment is not None:
        mask = segment(img.convert("RGB"))   # caller guarantees the convention
        img_l = mask.convert("L")
    else:
        img_l = img.convert("L")

    if mode == "letterbox":
        img_l = _letterbox(img_l, pad=pad)
    else:
        img_l = _legacy(img_l)

    arr = np.asarray(img_l, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, 224, 224]


# ---------------------------------------------------------------------------
# Optional segmentation backend (rembg). Not a dependency by default.
# Install with:  pip install rembg onnxruntime
# WARNING: rembg's mask convention (soft alpha, person=foreground) will very
# likely NOT match your training silhouettes out of the box. Treat this as a
# starting point and validate against bodyscan-3d before enabling.
# ---------------------------------------------------------------------------

_REMBG_SESSION = None


def rembg_segment(img_rgb: Image.Image) -> Image.Image:
    """Return a hard silhouette: person=white(255) on black(0), mode 'L'.

    Adjust the convention (invert / soft alpha / fill) to match training.
    """
    global _REMBG_SESSION
    from rembg import remove, new_session  # lazy: optional dependency

    if _REMBG_SESSION is None:
        _REMBG_SESSION = new_session("u2net_human_seg")

    cut = remove(img_rgb, session=_REMBG_SESSION)  # RGBA, transparent bg
    alpha = np.asarray(cut.split()[-1])            # 0..255 person mask
    binary = (alpha > 127).astype(np.uint8) * 255  # hard silhouette
    return Image.fromarray(binary, mode="L")
